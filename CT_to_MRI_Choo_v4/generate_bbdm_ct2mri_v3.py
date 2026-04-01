"""
Inference script v3 for 2.5D Brownian Bridge Diffusion Model (CT-to-MRI).

Uses v3 dataset (official Choo2024 MR histogram, no HistogramEncoder) and bbdm_model_v3.
Context shape: [D, 128, 3, 1] (avg_histogram expanded) — auto-rearranged to [D, 3, 128]
by attention.py's 4D branch.

The average MR histogram is loaded from the checkpoint (saved during v3 training)
or can be overridden via --avg_hist_path. If neither exists and --train_metadata
is provided, it is computed from the training set and cached.

Usage:
  # Normal mode:
  python generate_bbdm_ct2mri_v3.py --experiment_name EXP --checkpoint_ver epoch050 ...

  # ISTA mode:
  python generate_bbdm_ct2mri_v3.py --experiment_name EXP --checkpoint_ver epoch050 \\
      --use_ista --num_ISTA_step 1 --ISTA_step_size 0.5 --sub_batch_size 48 ...
"""
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.ct2mri_25d_v3 import CT2MRI_25D_Inference_Dataset
from model.BrownianBridge.bbdm_model_v3 import BrownianBridgeModel
from utils.common import set_random_seed


def to_0_1(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] to [0, 1]"""
    return (x + 1.0) / 2.0


def get_3_slices(volume: torch.Tensor, slice_idx: int) -> torch.Tensor:
    """
    Extract 3 consecutive slices centered at slice_idx.

    Args:
        volume: [D, H, W] tensor
        slice_idx: Center slice index

    Returns:
        [3, H, W] tensor
    """
    D = volume.shape[0]

    if slice_idx == 0:
        slices = [volume[0], volume[0], volume[1]]
    elif slice_idx == D - 1:
        slices = [volume[D - 2], volume[D - 1], volume[D - 1]]
    else:
        slices = [volume[slice_idx - 1], volume[slice_idx], volume[slice_idx + 1]]

    return torch.stack(slices, dim=0)


def pad_to_square(tensor: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    """Pad [C, H, W] tensor to [C, target_size, target_size] with symmetric zero padding."""
    C, H, W = tensor.shape

    if H == target_size and W == target_size:
        return tensor

    pad_h = target_size - H
    pad_w = target_size - W
    assert pad_h >= 0 and pad_w >= 0, f"Image size ({H}, {W}) larger than target ({target_size})"

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom),
                                   mode='constant', value=0)


def unpad_from_square(tensor: torch.Tensor, original_h: int, original_w: int,
                      target_size: int = 256) -> torch.Tensor:
    """Remove padding from [C, target_size, target_size] to recover [C, original_h, original_w]."""
    C, H, W = tensor.shape

    if H == original_h and W == original_w:
        return tensor

    pad_h = target_size - original_h
    pad_w = target_size - original_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return tensor[:, pad_top:pad_top + original_h, pad_left:pad_left + original_w]


def load_avg_histogram(ckpt: dict, avg_hist_path: Optional[str],
                       train_metadata: Optional[str], config: dict) -> torch.Tensor:
    """
    Load or compute the average MR histogram for inference.

    Priority:
    1. --avg_hist_path (if file exists) — load cached .pt file
    2. ckpt['avg_histogram']            — saved during v3 training
    3. --train_metadata                 — compute from training set and cache to avg_hist_path

    Args:
        ckpt: Loaded checkpoint dict
        avg_hist_path: Optional path to cached avg_histogram .pt file
        train_metadata: Optional path to training metadata CSV (for fallback computation)
        config: Model config from checkpoint

    Returns:
        avg_histogram: [128, 3, 1] tensor
    """
    # Option 1: load from cache file
    if avg_hist_path and Path(avg_hist_path).exists():
        avg_histogram = torch.load(avg_hist_path, map_location="cpu", weights_only=True)
        print(f"Loaded avg_histogram from cache: {avg_hist_path}")
        print(f"  Shape: {avg_histogram.shape}")
        return avg_histogram

    # Option 2: load from checkpoint (saved by train_bbdm_ct2mri_wandb_v3.py)
    if "avg_histogram" in ckpt:
        avg_histogram = ckpt["avg_histogram"]
        print(f"Loaded avg_histogram from checkpoint. Shape: {avg_histogram.shape}")
        if avg_hist_path:
            torch.save(avg_histogram, avg_hist_path)
            print(f"  Cached to: {avg_hist_path}")
        return avg_histogram

    # Option 3: compute from training set
    if train_metadata:
        print("Computing avg_histogram from training set (this may take a while)...")
        from datasets.ct2mri_25d_v3 import CT2MRI_25D_Dataset
        train_set = CT2MRI_25D_Dataset(
            metadata_csv=train_metadata,
        )
        avg_histogram = train_set.avg_histogram  # [128, 3, 1]
        print(f"Computed avg_histogram. Shape: {avg_histogram.shape}")
        if avg_hist_path:
            torch.save(avg_histogram, avg_hist_path)
            print(f"  Cached to: {avg_hist_path}")
        return avg_histogram

    raise ValueError(
        "Cannot load avg_histogram. Provide one of:\n"
        "  --avg_hist_path <path_to_cached.pt>  (if cached file exists)\n"
        "  Checkpoint with 'avg_histogram' key  (v3 training saves this)\n"
        "  --train_metadata <train_csv>          (compute from training set)"
    )


def reconstruct_3d_normal(model: BrownianBridgeModel,
                           ct_volume: torch.Tensor,
                           avg_histogram: torch.Tensor,
                           device: torch.device,
                           use_bf16: bool = True,
                           target_size: int = 256) -> torch.Tensor:
    """
    Reconstruct 3D MRI from CT by processing 2.5D slices independently (normal mode).

    Context: avg_histogram [128, 3, 1] → [1, 128, 3, 1] per slice.
    attention.py auto-rearranges [B, 128, 3, 1] → [B, 3, 128] for cross-attention.

    Args:
        model: Trained BrownianBridgeModel (v3)
        ct_volume: CT volume [D, H, W] in [-1, 1]
        avg_histogram: Average MR histogram [128, 3, 1]
        device: Device
        use_bf16: Use bfloat16
        target_size: Spatial size for model input

    Returns:
        mri_volume: [D, H, W] in [0, 1]
    """
    D, H, W = ct_volume.shape

    # Context: same avg_histogram for every slice, [1, 128, 3, 1]
    context = avg_histogram.unsqueeze(0).to(device)  # [1, 128, 3, 1]

    mri_slices = []

    for slice_idx in tqdm(range(D), desc="Processing slices", leave=False):
        ct_25d = get_3_slices(ct_volume, slice_idx)             # [3, H, W]
        ct_25d_padded = pad_to_square(ct_25d, target_size)      # [3, T, T]
        ct_25d_batch = ct_25d_padded.unsqueeze(0).to(device)    # [1, 3, T, T]

        with autocast(enabled=use_bf16, dtype=torch.bfloat16):
            mri_25d = model.sample(
                ct_25d_batch, context,
                clip_denoised=True,
                inference_type='normal'
            )  # [1, 3, T, T]

        mri_slice_padded = mri_25d[0, 1, :, :].unsqueeze(0)   # [1, T, T]
        mri_slice = unpad_from_square(mri_slice_padded, H, W, target_size)[0]  # [H, W]
        mri_slices.append(mri_slice.cpu())

    mri_volume = torch.stack(mri_slices, dim=0)  # [D, H, W]
    mri_volume = to_0_1(mri_volume)
    mri_volume = torch.clamp(mri_volume, 0, 1)

    return mri_volume


def reconstruct_3d_ista(model: BrownianBridgeModel,
                         ct_volume: torch.Tensor,
                         avg_histogram: torch.Tensor,
                         device: torch.device,
                         use_bf16: bool = True,
                         target_size: int = 256,
                         inference_type: str = 'ISTA_average',
                         num_ISTA_step: int = 1,
                         ISTA_step_size: float = 0.5,
                         sub_batch_size: int = 48) -> torch.Tensor:
    """
    Reconstruct 3D MRI from CT using ISTA (processes all D slices simultaneously).

    Context: avg_histogram [128, 3, 1] → expanded to [D, 128, 3, 1].
    attention.py auto-rearranges each slice's context [128, 3, 1] → [3, 128] for
    cross-attention (same for all D slices since avg_histogram is shared).

    Args:
        model: Trained BrownianBridgeModel (v3)
        ct_volume: CT volume [D, H, W] in [-1, 1]
        avg_histogram: Average MR histogram [128, 3, 1]
        device: Device
        use_bf16: Use bfloat16
        target_size: Spatial size for model input
        inference_type: 'ISTA_average' or 'ISTA_mid'
        num_ISTA_step: ISTA correction iterations per diffusion step
        ISTA_step_size: ISTA adaptive step size multiplier
        sub_batch_size: Max slices per UNet forward pass

    Returns:
        mri_volume: [D, H, W] in [0, 1]
    """
    D, H, W = ct_volume.shape

    # Expand avg_histogram to [D, 128, 3, 1] for full-volume batch
    context = avg_histogram.unsqueeze(0).expand(D, -1, -1, -1).contiguous().to(device)
    # [D, 128, 3, 1] — attention.py rearranges to [D, 3, 128] per slice

    # Build full CT batch [D, 3, target_size, target_size]
    ct_slices = []
    for slice_idx in tqdm(range(D), desc="Preparing CT slices", leave=False):
        ct_25d = get_3_slices(ct_volume, slice_idx)         # [3, H, W]
        ct_25d_padded = pad_to_square(ct_25d, target_size)  # [3, T, T]
        ct_slices.append(ct_25d_padded)

    ct_batch = torch.stack(ct_slices, dim=0).to(device)  # [D, 3, T, T]

    # Run ISTA inference
    with autocast(enabled=use_bf16, dtype=torch.bfloat16):
        mri_batch = model.sample(
            ct_batch, context,
            clip_denoised=True,
            inference_type=inference_type,
            num_ISTA_step=num_ISTA_step,
            ISTA_step_size=ISTA_step_size,
            sub_batch_size=sub_batch_size
        )  # [D, 3, T, T]

    # Extract center channel and unpad each slice
    mri_slices = []
    for slice_idx in range(D):
        mri_slice_padded = mri_batch[slice_idx, 1, :, :].unsqueeze(0)  # [1, T, T]
        mri_slice = unpad_from_square(mri_slice_padded, H, W, target_size)[0]  # [H, W]
        mri_slices.append(mri_slice.cpu())

    mri_volume = torch.stack(mri_slices, dim=0)  # [D, H, W]
    mri_volume = to_0_1(mri_volume)
    mri_volume = torch.clamp(mri_volume, 0, 1)

    return mri_volume


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MRI from CT using 2.5D BBDM (v3)")

    # Model checkpoint
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--checkpoint_ver", type=str, required=True,
                        help="Checkpoint version (e.g., 'epoch050')")

    # Data
    parser.add_argument("--test_metadata", type=str, required=True,
                        help="Path to test metadata CSV")
    parser.add_argument("--mni", action="store_true",
                        help="Use MNI-registered data")

    # Average histogram
    parser.add_argument("--avg_hist_path", type=str, default=None,
                        help="Path to cached avg_histogram .pt file (loads from ckpt if absent)")
    parser.add_argument("--train_metadata", type=str, default=None,
                        help="Path to training metadata CSV (used only if avg_histogram "
                             "is not in checkpoint and --avg_hist_path doesn't exist)")

    # Generation options
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)

    # ISTA options
    parser.add_argument("--use_ista", action="store_true",
                        help="Use ISTA_average inference (default: normal slice-by-slice)")
    parser.add_argument("--num_ISTA_step", type=int, default=1,
                        help="ISTA correction iterations per diffusion step")
    parser.add_argument("--ISTA_step_size", type=float, default=0.5,
                        help="ISTA adaptive step size multiplier")
    parser.add_argument("--sub_batch_size", type=int, default=6,
                        help="Max slices per UNet forward pass in ISTA mode")

    # Seed
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== Load Model ==========
    print("\n" + "=" * 60)
    print("Loading Model...")
    print("=" * 60)

    ckpt_dir = Path("/pscratch/sd/s/seojw/CT_to_MRI/checkpoints") / args.experiment_name
    ckpt_path = ckpt_dir / f"{args.experiment_name}_{args.checkpoint_ver}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

    target_size = config.get("image_size", 256)
    print(f"Target size for model: {target_size}")

    # Build model (v3: context_dim=128 still valid; shape [B,128,3,1]->[B,3,128] via attention.py)
    model = BrownianBridgeModel(
        image_size=target_size,
        in_channels=config.get("in_channels", 6),
        out_channels=config.get("out_channels", 3),
        model_channels=config.get("model_channels", 128),
        num_res_blocks=config.get("num_res_blocks", 2),
        attention_resolutions=tuple(config.get("attention_resolutions", [32, 16, 8])),
        channel_mult=tuple(config.get("channel_mult", [1, 4, 8])),
        num_heads=config.get("num_heads", 8),
        num_head_channels=config.get("num_head_channels", 64),
        num_timesteps=config.get("num_timesteps", 1000),
        mt_type=config.get("mt_type", "linear"),
        max_var=config.get("max_var", 1.0),
        eta=config.get("eta", 0.0),
        objective=config.get("objective", "grad"),
        loss_type=config.get("loss_type", "l1"),
        skip_sample=True,
        sample_type=config.get("sample_type", "linear"),
        sample_step=config.get("sample_step", 50),
        use_context=config.get("use_histogram", True),
        context_dim=config.get("context_dim", 128),
        condition_key='hist_context_y_concat'
    ).to(device)

    # Load weights (EMA preferred)
    if "ema" in ckpt:
        print("Loading EMA weights...")
        from train_bbdm_ct2mri_wandb_v3 import EMA
        ema = EMA(model.denoise_fn, decay=config.get("ema_decay", 0.995))
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
    else:
        print("Loading regular weights...")
        model.load_state_dict(ckpt["model"], strict=True)

    model.eval().to(device)
    print("Model loaded (no HistogramEncoder in v3)")

    # ========== Load avg_histogram ==========
    print("\n" + "=" * 60)
    print("Loading avg_histogram...")
    print("=" * 60)

    avg_histogram = load_avg_histogram(
        ckpt=ckpt,
        avg_hist_path=args.avg_hist_path,
        train_metadata=args.train_metadata,
        config=config
    )  # [128, 3, 1]

    assert avg_histogram.shape == (128, 3, 1), \
        f"Expected avg_histogram shape [128, 3, 1], got {avg_histogram.shape}"

    # ========== Load Dataset ==========
    print("\n" + "=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    test_set = CT2MRI_25D_Inference_Dataset(
        metadata_csv=args.test_metadata,
        mni=args.mni,
        avg_histogram=avg_histogram
    )

    print(f"Test subjects: {len(test_set)}")

    # ========== Prepare Output Directory ==========
    if args.use_ista:
        inference_suffix = f"ISTA_average_n{args.num_ISTA_step}_s{args.ISTA_step_size}"
    else:
        inference_suffix = "normal"

    output_dir = ckpt_dir / "syn" / f"{args.checkpoint_ver}_{inference_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    ct_dir = output_dir / "ct"
    ct_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inference type: {inference_suffix}")
    print(f"Output directory: {output_dir}")

    # ========== Generation Loop ==========
    print("\n" + "=" * 60)
    print("Starting Generation...")
    print("=" * 60)

    with torch.no_grad():
        for idx in tqdm(range(len(test_set)), desc="Generating"):
            sample = test_set[idx]
            subject_id = sample["id"]
            print(subject_id)

            ct = sample["ct"].to(device)       # [D, H, W] in [-1, 1]
            mri_gt = sample["mri"].to(device)  # [D, H, W] in [-1, 1]
            # hist_mr = sample["hist_mr"]      # [128, 3, 1] — same as avg_histogram, not needed here

            # Reconstruct 3D MRI
            if args.use_ista:
                mri_syn = reconstruct_3d_ista(
                    model, ct, avg_histogram, device, args.use_bf16, target_size,
                    inference_type='ISTA_average',
                    num_ISTA_step=args.num_ISTA_step,
                    ISTA_step_size=args.ISTA_step_size,
                    sub_batch_size=args.sub_batch_size
                )
            else:
                mri_syn = reconstruct_3d_normal(
                    model, ct, avg_histogram, device, args.use_bf16, target_size
                )
            # mri_syn: [D, H, W] in [0, 1]

            # Convert CT and GT MRI to [0, 1]
            ct_01 = torch.clamp(to_0_1(ct), 0, 1)
            mri_gt_01 = torch.clamp(to_0_1(mri_gt), 0, 1)

            # Save
            torch.save(mri_syn, output_dir / f"syn_t1_{subject_id}.pt")
            torch.save(mri_gt_01, original_dir / f"orig_t1_{subject_id}.pt")
            torch.save(ct_01, ct_dir / f"ct_{subject_id}.pt")

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(test_set)} subjects")

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Synthetic MRI saved to: {output_dir}")
    print(f"Original MRI saved to: {original_dir}")
    print(f"CT saved to: {ct_dir}")
    print(f"Total subjects: {len(test_set)}")


if __name__ == "__main__":
    main()
