"""
Deep Ensemble UQ generation script for 2.5D BBDM (CT-to-MRI) — v4.

Single-GPU workflow (sequential checkpoint loading):
  For each ensemble checkpoint (N=5 default):
    1. Load checkpoint → build model → send to GPU
    2. Generate synthetic MRI for all test subjects
    3. Save per-subject volumes to a per-ensemble intermediate directory
    4. Delete model, call gc.collect() + torch.cuda.empty_cache()  ← GPU freed
  After all N checkpoints:
    5. Load all intermediate .pt files, stack → [N, D, H, W]
    6. Compute mean and variance → save final UQ maps

This sequential approach prevents OOM on a single GPU: only one model
lives on the GPU at a time.

Usage:
  python generate_uq_ensemble_v4.py \\
      --ensemble_ckpt_paths \\
          "/path/ckpts/exp_seed1337/exp_seed1337_epoch050.pt" \\
          "/path/ckpts/exp_seed42/exp_seed42_epoch050.pt"    \\
          "/path/ckpts/exp_seed123/exp_seed123_epoch050.pt"  \\
          "/path/ckpts/exp_seed456/exp_seed456_epoch050.pt"  \\
          "/path/ckpts/exp_seed789/exp_seed789_epoch050.pt"  \\
      --test_metadata "/path/test_metadata.csv"              \\
      --output_dir "/path/ensemble_uq_results"               \\
      --use_ista --mni

Output files (in --output_dir):
  ensemble_mean_{subject_id}.pt    — pixel-wise mean  [D, H, W] in [0,1]
  ensemble_var_{subject_id}.pt     — pixel-wise variance [D, H, W]
  original/orig_t1_{subject_id}.pt — GT MRI
  ct/ct_{subject_id}.pt            — CT
  intermediates/member_{k}/{subject_id}.pt  — per-member outputs (kept for audit)
"""

import argparse
import gc
from pathlib import Path
from typing import List, Optional

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.ct2mri_25d_v3 import CT2MRI_25D_Inference_Dataset
from model.BrownianBridge.bbdm_model_v4 import BrownianBridgeModel
from utils.common import set_random_seed


# ---------------------------------------------------------------------------
# Helpers copied from generate_bbdm_ct2mri_v4.py (I/O unchanged from v3)
# ---------------------------------------------------------------------------

def to_0_1(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def get_3_slices(volume: torch.Tensor, slice_idx: int) -> torch.Tensor:
    D = volume.shape[0]
    if slice_idx == 0:
        slices = [volume[0], volume[0], volume[1]]
    elif slice_idx == D - 1:
        slices = [volume[D - 2], volume[D - 1], volume[D - 1]]
    else:
        slices = [volume[slice_idx - 1], volume[slice_idx], volume[slice_idx + 1]]
    return torch.stack(slices, dim=0)


def pad_to_square(tensor: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    C, H, W = tensor.shape
    if H == target_size and W == target_size:
        return tensor
    pad_h = target_size - H
    pad_w = target_size - W
    assert pad_h >= 0 and pad_w >= 0
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    return torch.nn.functional.pad(
        tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0
    )


def unpad_from_square(tensor: torch.Tensor, original_h: int, original_w: int,
                      target_size: int = 256) -> torch.Tensor:
    if tensor.shape[1] == original_h and tensor.shape[2] == original_w:
        return tensor
    pad_top  = (target_size - original_h) // 2
    pad_left = (target_size - original_w) // 2
    return tensor[:, pad_top:pad_top + original_h, pad_left:pad_left + original_w]


def reconstruct_3d_normal(model, ct_volume, avg_histogram, device,
                          use_bf16=True, target_size=256):
    D, H, W = ct_volume.shape
    context = avg_histogram.unsqueeze(0).to(device)  # [1, 128, 3, 1]
    mri_slices = []
    for slice_idx in tqdm(range(D), desc="Slices", leave=False):
        ct_25d = get_3_slices(ct_volume, slice_idx)
        ct_25d_padded = pad_to_square(ct_25d, target_size)
        ct_25d_batch  = ct_25d_padded.unsqueeze(0).to(device)
        with autocast(enabled=use_bf16, dtype=torch.bfloat16):
            mri_25d = model.sample(ct_25d_batch, context,
                                   clip_denoised=True, inference_type='normal')
        mri_slice = unpad_from_square(mri_25d[0, 1, :, :].unsqueeze(0), H, W, target_size)[0]
        mri_slices.append(mri_slice.cpu())
    mri_volume = torch.stack(mri_slices, dim=0)
    return torch.clamp(to_0_1(mri_volume), 0, 1)


def reconstruct_3d_ista(model, ct_volume, avg_histogram, device,
                        use_bf16=True, target_size=256,
                        inference_type='ISTA_average',
                        num_ISTA_step=1, ISTA_step_size=0.5, sub_batch_size=6):
    D, H, W = ct_volume.shape
    context = avg_histogram.unsqueeze(0).expand(D, -1, -1, -1).contiguous().to(device)
    ct_slices = []
    for slice_idx in tqdm(range(D), desc="Preparing CT", leave=False):
        ct_25d = get_3_slices(ct_volume, slice_idx)
        ct_slices.append(pad_to_square(ct_25d, target_size))
    ct_batch = torch.stack(ct_slices, dim=0).to(device)
    with autocast(enabled=use_bf16, dtype=torch.bfloat16):
        mri_batch = model.sample(ct_batch, context, clip_denoised=True,
                                 inference_type=inference_type,
                                 num_ISTA_step=num_ISTA_step,
                                 ISTA_step_size=ISTA_step_size,
                                 sub_batch_size=sub_batch_size)
    mri_slices = []
    for slice_idx in range(D):
        mri_slice = unpad_from_square(
            mri_batch[slice_idx, 1, :, :].unsqueeze(0), H, W, target_size)[0]
        mri_slices.append(mri_slice.cpu())
    mri_volume = torch.stack(mri_slices, dim=0)
    return torch.clamp(to_0_1(mri_volume), 0, 1)


# ---------------------------------------------------------------------------
# Model loading helper (I/O unchanged from v3)
# ---------------------------------------------------------------------------

def load_avg_histogram(ckpt: dict, avg_hist_path: Optional[str],
                       train_metadata: Optional[str]) -> torch.Tensor:
    """Load avg_histogram: from file → checkpoint → compute from train set."""
    if avg_hist_path and Path(avg_hist_path).exists():
        avg_histogram = torch.load(avg_hist_path, map_location="cpu", weights_only=True)
        print(f"  avg_histogram loaded from cache: {avg_hist_path}")
        return avg_histogram
    if "avg_histogram" in ckpt:
        avg_histogram = ckpt["avg_histogram"]
        print(f"  avg_histogram loaded from checkpoint. Shape: {avg_histogram.shape}")
        return avg_histogram
    if train_metadata:
        from datasets.ct2mri_25d_v3 import CT2MRI_25D_Dataset
        train_set = CT2MRI_25D_Dataset(metadata_csv=train_metadata)
        avg_histogram = train_set.avg_histogram
        print(f"  avg_histogram computed from train set. Shape: {avg_histogram.shape}")
        return avg_histogram
    raise ValueError(
        "Cannot load avg_histogram. Provide --avg_hist_path, a v4 checkpoint with "
        "'avg_histogram' key, or --train_metadata."
    )


def build_and_load_model(ckpt_path: str, device: torch.device) -> tuple:
    """
    Load a v4 checkpoint and return (model, ckpt_dict, config, target_size).

    The ckpt_dict is returned so the caller can extract avg_histogram without
    a second torch.load() call (avoids double I/O for large checkpoints).

    Checkpoint format is IDENTICAL to v3:
      {"epoch", "global_step", "model", "ema", "optim", "config", "avg_histogram"}
    No changes to I/O.
    """
    print(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    target_size = config.get("image_size", 256)

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
        condition_key='hist_context_y_concat',
        dropout_rate=config.get("dropout_rate", 0.0),
    ).to(device)

    # Load weights (EMA preferred) — same as v3
    if "ema" in ckpt:
        from train_bbdm_ct2mri_wandb_v4 import EMA
        ema = EMA(model.denoise_fn, decay=config.get("ema_decay", 0.995))
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
        print("  EMA weights applied.")
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        print("  Regular weights loaded.")

    model.eval()
    return model, ckpt, config, target_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deep Ensemble UQ generation for 2.5D BBDM (v4) — single-GPU sequential"
    )
    parser.add_argument("--ensemble_ckpt_paths", nargs="+", required=True,
                        help="Paths to N ensemble checkpoint .pt files (space-separated). "
                             "Each was trained with a different --seed.")
    parser.add_argument("--test_metadata", type=str, required=True)
    parser.add_argument("--output_dir",    type=str, required=True,
                        help="Root directory for all UQ outputs.")
    parser.add_argument("--mni",           action="store_true")
    parser.add_argument("--avg_hist_path", type=str, default=None)
    parser.add_argument("--train_metadata",type=str, default=None)
    parser.add_argument("--use_bf16",      action="store_true", default=True)
    parser.add_argument("--use_ista",      action="store_true")
    parser.add_argument("--num_ISTA_step", type=int,   default=1)
    parser.add_argument("--ISTA_step_size",type=float, default=0.5)
    parser.add_argument("--sub_batch_size",type=int,   default=6)
    parser.add_argument("--seed",          type=int,   default=1337,
                        help="Seed for diffusion noise during generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_ensemble = len(args.ensemble_ckpt_paths)
    print(f"Ensemble size: N={n_ensemble}")
    for k, p in enumerate(args.ensemble_ckpt_paths):
        print(f"  Member {k}: {p}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir / "original"
    original_dir.mkdir(exist_ok=True)
    ct_dir = output_dir / "ct"
    ct_dir.mkdir(exist_ok=True)
    inter_root = output_dir / "intermediates"
    inter_root.mkdir(exist_ok=True)

    # ---- Phase 1: Sequential per-member generation -------------------------
    # Load one checkpoint at a time, generate all subjects, save, clear GPU.

    # We'll collect subject IDs from the first member's run and reuse.
    subject_ids: List[str] = []

    for k, ckpt_path in enumerate(args.ensemble_ckpt_paths):
        print("\n" + "=" * 60)
        print(f"Ensemble member {k + 1}/{n_ensemble}")
        print("=" * 60)

        # ---- Load model (one at a time, sequential) ----
        # build_and_load_model returns the ckpt dict so we avoid loading it twice.
        model, ckpt, config, target_size = build_and_load_model(ckpt_path, device)

        # ---- Load avg_histogram (from the already-loaded ckpt dict) ----
        avg_histogram = load_avg_histogram(
            ckpt=ckpt,
            avg_hist_path=args.avg_hist_path,
            train_metadata=args.train_metadata
        )
        assert avg_histogram.shape == (128, 3, 1), \
            f"avg_histogram shape mismatch: {avg_histogram.shape}"

        # Free ckpt dict from CPU RAM (weights are already in model on GPU)
        del ckpt
        gc.collect()

        # ---- Load dataset ----
        test_set = CT2MRI_25D_Inference_Dataset(
            metadata_csv=args.test_metadata,
            mni=args.mni,
            avg_histogram=avg_histogram
        )
        print(f"Test subjects: {len(test_set)}")

        # ---- Intermediate directory for this member ----
        inter_dir = inter_root / f"member_{k:02d}"
        inter_dir.mkdir(exist_ok=True)

        # ---- Generate all subjects with this model ----
        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), desc=f"Member {k + 1}"):
                sample = test_set[idx]
                subject_id = sample["id"]

                ct    = sample["ct"].to(device)    # [D, H, W] in [-1, 1]
                mri_gt = sample["mri"].to(device)  # [D, H, W] in [-1, 1]

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
                # mri_syn: [D, H, W] in [0, 1], on CPU

                # Save intermediate for this member
                torch.save(mri_syn, inter_dir / f"{subject_id}.pt")

                # Save GT and CT once (on first member's pass)
                if k == 0:
                    subject_ids.append(subject_id)
                    torch.save(torch.clamp(to_0_1(mri_gt.cpu()), 0, 1),
                               original_dir / f"orig_t1_{subject_id}.pt")
                    torch.save(torch.clamp(to_0_1(ct.cpu()), 0, 1),
                               ct_dir / f"ct_{subject_id}.pt")

        # ---- Free GPU memory before loading next model ----
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Member {k + 1} done. GPU memory cleared.")

    # ---- Phase 2: Aggregate across ensemble members -------------------------
    print("\n" + "=" * 60)
    print("Aggregating ensemble members → mean + variance")
    print("=" * 60)

    for subject_id in tqdm(subject_ids, desc="Aggregating"):
        # Load all N intermediate volumes for this subject
        member_volumes = []
        for k in range(n_ensemble):
            vol = torch.load(
                inter_root / f"member_{k:02d}" / f"{subject_id}.pt",
                map_location="cpu",
                weights_only=True
            )
            member_volumes.append(vol)

        # Stack: [N, D, H, W]
        stack = torch.stack(member_volumes, dim=0)

        ensemble_mean = stack.mean(dim=0)    # [D, H, W]
        ensemble_var  = stack.var(dim=0)     # [D, H, W]  (Bessel's correction, N-1)

        torch.save(ensemble_mean, output_dir / f"ensemble_mean_{subject_id}.pt")
        torch.save(ensemble_var,  output_dir / f"ensemble_var_{subject_id}.pt")

    print("\n" + "=" * 60)
    print("Deep Ensemble UQ Complete!")
    print("=" * 60)
    print(f"  Ensemble members : {n_ensemble}")
    print(f"  Subjects processed: {len(subject_ids)}")
    print(f"  Outputs saved to : {output_dir}")
    print(f"    ensemble_mean_{{id}}.pt  — mean prediction [D, H, W] in [0,1]")
    print(f"    ensemble_var_{{id}}.pt   — epistemic uncertainty [D, H, W]")
    print(f"    original/              — GT MRI")
    print(f"    ct/                    — CT")
    print(f"    intermediates/         — per-member outputs (for audit/debug)")


if __name__ == "__main__":
    main()
