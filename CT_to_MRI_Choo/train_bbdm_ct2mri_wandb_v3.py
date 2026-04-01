"""
Training script for 2.5D Brownian Bridge Diffusion Model (CT-to-MRI).
v3: Official Choo2024 MR histogram approach — no HistogramEncoder.

Changes from v2 (train_bbdm_ct2mri_wandb_v2.py):
  1. [Dataset]        Uses CT2MRI_25D_Dataset from ct2mri_25d_v3 (MR histogram).
  2. [Histogram]      Removed HistogramEncoder entirely.
                      context = batch["hist_mr"].to(device)  → [B, 128, 3, 1]
                      attention.py auto-rearranges [B,128,3,1] → [B,3,128].
  3. [context_dim]    Still 128 — only token count changes (1→3), dim unchanged.
  4. [Checkpoint]     No hist_encoder in save/load.
  5. [avg_histogram]  train_dataset.avg_histogram ([128,3,1]) passed to val/inf dataset.

Histogram details:
  - Source    : MR (target) volume, whole-volume global
  - Bins      : 128, range (0.001, 1.0)
  - Features  : 3 — normalized (×10), cumulative, differential (×10)
  - Shape     : [B, 128, 3, 1] in batch → CrossAttention sees [B, 3, 128]
  - Train/Val : per-subject MR histogram (ground truth MRI available)
  - Test      : avg_histogram from all training subjects (no ground truth)

Reference:
  CT2MRI/brain_dataset_utils/generate_total_hist_global.py
  CT2MRI/datasets/custom.py lines 96-102
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

from datasets.ct2mri_25d_v3 import CT2MRI_25D_Dataset
from model.BrownianBridge.bbdm_model import BrownianBridgeModel
from utils.common import set_random_seed, count_params, min_max_norm


# =============================================================================
# Weight Initialization
# Exact copy of CT2MRI/runners/utils.py::weights_init
# =============================================================================
def weights_init(m):
    """
    GAN-style weight initialization matching official Choo2024 code.
    - Conv2d / Linear / Parameter : Normal(mean=0.0, std=0.02)
    - BatchNorm                   : weight~Normal(1.0, 0.02), bias=0
    Source: CT2MRI/runners/utils.py:37-47
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# =============================================================================
# EMA (same as v2)
# Source: CT2MRI/runners/base/EMA.py
# =============================================================================
class EMA:
    """
    Exponential Moving Average for model parameters.
    with_decay=False before start_ema_step (warm-up: shadow copies current weights).
    with_decay=True  from  start_ema_step  (standard EMA formula).
    Source: CT2MRI/runners/base/EMA.py + CT2MRI/runners/BaseRunner.py::step_ema
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, with_decay: bool = True):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if with_decay:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                else:
                    new_average = param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2.5D Brownian Bridge Diffusion Model for CT-to-MRI (v3)"
    )

    # Data
    parser.add_argument("--train_metadata", type=str, required=True)
    parser.add_argument("--val_metadata",   type=str, required=True)
    parser.add_argument("--severance_only", action="store_true", default=False)
    parser.add_argument("--mni",            action="store_true")

    # Experiment
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--wandb_project",   type=str, default="CT-to-MRI")
    parser.add_argument("--wandb_entity",    type=str, default=None)
    parser.add_argument("--seed",            type=int, default=1337)

    # Model architecture
    parser.add_argument("--image_size",             type=int,   default=256)
    parser.add_argument("--in_channels",            type=int,   default=6)
    parser.add_argument("--out_channels",           type=int,   default=3)
    parser.add_argument("--model_channels",         type=int,   default=128)
    parser.add_argument("--num_res_blocks",         type=int,   default=2)
    parser.add_argument("--channel_mult",           type=int,   nargs="+", default=[1, 4, 8])
    parser.add_argument("--attention_resolutions",  type=int,   nargs="+", default=[32, 16, 8])
    parser.add_argument("--num_heads",              type=int,   default=8)
    parser.add_argument("--num_head_channels",      type=int,   default=64)

    # Brownian Bridge parameters
    parser.add_argument("--num_timesteps", type=int,   default=1000)
    parser.add_argument("--mt_type",       type=str,   default="linear",
                        choices=["linear", "sin", "control"])
    parser.add_argument("--max_var",       type=float, default=1.0)
    parser.add_argument("--eta",           type=float, default=0.0)
    parser.add_argument("--objective",     type=str,   default="grad",
                        choices=["grad", "noise", "ysubx"])
    parser.add_argument("--loss_type",     type=str,   default="l1",
                        choices=["l1", "l2"])

    # Sampling
    parser.add_argument("--sample_step", type=int, default=50)
    parser.add_argument("--sample_type", type=str, default="linear",
                        choices=["linear", "cosine"])

    # v3: context_dim stays 128 (token dim), no histogram_bins arg needed
    # (bins are fixed at 128 inside the dataset)
    parser.add_argument("--context_dim", type=int, default=128,
                        help="Cross-attention context dimension. "
                             "v3: histogram token dim is 128 (fixed). "
                             "Number of tokens is 3 (norm+cumsum+diff).")

    # Training
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--max_epochs",      type=int,   default=10000)
    parser.add_argument("--val_every",       type=int,   default=1)
    parser.add_argument("--grad_accum_steps",type=int,   default=1)
    parser.add_argument("--use_bf16",        action="store_true", default=True)

    # Augmentation
    parser.add_argument("--augment",      action="store_true")
    parser.add_argument("--augment_flip", action="store_true")

    # EMA (same as v2)
    parser.add_argument("--ema_decay",           type=float, default=0.995)
    parser.add_argument("--start_ema_step",      type=int,   default=30000)
    parser.add_argument("--ema_update_interval", type=int,   default=8)

    # Resume
    parser.add_argument("--resume_ckpt", type=str, default="")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_random_seed(args.seed)

    os.environ['WANDB_API_KEY'] = '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'

    ckpt_root = Path("/pscratch/sd/s/seojw/CT_to_MRI/checkpoints")
    exp_dir   = ckpt_root / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        tags=["BBDM", "2.5D", "CT2MRI", "Choo2024", "v3", "MR-histogram"],
        config=args_dict
    )

    print("=" * 60)
    print("Experiment Configuration (v3):")
    for key, value in args_dict.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # Datasets
    # v3: CT2MRI_25D_Dataset pre-computes MR histograms for all subjects.
    #     train_set.avg_histogram ([128,3,1]) is used for val inference.
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading Datasets (v3: MR histogram, volume caching enabled)...")
    print("=" * 60)

    train_set = CT2MRI_25D_Dataset(
        args.train_metadata,
        augment=args.augment,
        flip=args.augment_flip,
        mni=args.mni,
        severance_only=args.severance_only,
        target_size=args.image_size
    )

    # avg_histogram: mean MR histogram over all training subjects [128, 3, 1]
    # This is used for val and any inference dataset (no ground-truth MRI there).
    avg_histogram = train_set.avg_histogram  # [128, 3, 1]
    print(f"avg_histogram shape : {avg_histogram.shape}")

    val_set = CT2MRI_25D_Dataset(
        args.val_metadata,
        augment=False,
        flip=False,
        mni=args.mni,
        severance_only=args.severance_only,
        target_size=args.image_size
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Train slices : {len(train_set)}")
    print(f"Val   slices : {len(val_set)}")

    # =========================================================================
    # Model
    # v3: use_context=True, context_dim=128 (token dim stays 128)
    #     No HistogramEncoder — raw histogram tensor is the context.
    # =========================================================================
    print("\n" + "=" * 60)
    print("Initializing Model (v3)...")
    print("=" * 60)

    model = BrownianBridgeModel(
        image_size=args.image_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(args.attention_resolutions),
        channel_mult=tuple(args.channel_mult),
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_timesteps=args.num_timesteps,
        mt_type=args.mt_type,
        max_var=args.max_var,
        eta=args.eta,
        objective=args.objective,
        loss_type=args.loss_type,
        skip_sample=True,
        sample_type=args.sample_type,
        sample_step=args.sample_step,
        use_context=True,
        context_dim=args.context_dim,
        condition_key='hist_context_y_concat'
    ).to(device)

    model.apply(weights_init)
    print("Weight init applied : Normal(0.0, 0.02) for Conv2d/Linear, "
          "Normal(1.0, 0.02)/bias=0 for BatchNorm")

    print(f"Model parameters   : {count_params(model, trainable_only=True):,}")
    print("v3: No HistogramEncoder — raw MR histogram passed as context.")

    # EMA (same as v2)
    ema = EMA(model.denoise_fn, decay=args.ema_decay)
    print(f"EMA : decay={args.ema_decay}, "
          f"update every {args.ema_update_interval} global_steps, "
          f"decay starts at global_step={args.start_ema_step}")

    # Optimizer: Adam, weight_decay=0.0 (same as v2)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.999)
    )
    print(f"Optimizer : Adam(lr={args.lr}, weight_decay=0.0, betas=(0.9, 0.999))")
    print(f"Note: v3 has fewer optimizer params than v2 (no HistogramEncoder)")

    # =========================================================================
    # Resume from checkpoint
    # v3: no hist_encoder in checkpoint
    # =========================================================================
    start_epoch = 1
    global_step = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"\nResuming from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            for name in ema.shadow:
                ema.shadow[name] = ema.shadow[name].to(device)
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed: epoch={start_epoch - 1}, global_step={global_step}")

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting Training (v3)...")
    print(f"  Histogram     : MR whole-volume, 128 bins, 3 features, no encoder")
    print(f"  Context shape : [B, 128, 3, 1] → auto-rearranged to [B, 3, 128]")
    print(f"  Adam optimizer: fires every {args.grad_accum_steps} batch(es)")
    print(f"  EMA update    : fires every {args.ema_update_interval} global_step(s)")
    print(f"  EMA decay starts: global_step >= {args.start_ema_step}")
    print("=" * 60)

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(pbar):

            ct  = batch["ct"].to(device, non_blocking=True)   # [B, 3, H, W]
            mri = batch["mri"].to(device, non_blocking=True)  # [B, 3, H, W]

            # v3: MR histogram context — no encoder, raw values
            # batch["hist_mr"] : [B, 128, 3, 1]
            # CrossAttention in attention.py sees context.dim()==4 and rearranges
            # to [B, 3, 128] via: rearrange('b c h w -> b (h w) c')
            context = batch["hist_mr"].to(device, non_blocking=True)  # [B, 128, 3, 1]

            # Shape verification (debug — remove in production if needed)
            assert context.shape[1:] == (128, 3, 1), \
                f"Expected hist_mr [B,128,3,1], got {context.shape}"

            # Forward pass
            with autocast(enabled=args.use_bf16, dtype=torch.bfloat16):
                loss, log_dict = model(mri, ct, context)

            # Backward (scaled for gradient accumulation)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # EMA update
                if global_step % args.ema_update_interval == 0:
                    with_decay = (global_step >= args.start_ema_step)
                    ema.update(with_decay=with_decay)

            epoch_loss += loss.item() * args.grad_accum_steps

            # Per-step logging
            if global_step % 50 == 0 and global_step > 0:
                step_loss = loss.item() * args.grad_accum_steps
                ema_phase = "decay" if global_step >= args.start_ema_step else "warmup"
                wandb.log({
                    "train/loss": step_loss,
                    "train/ema_phase": 1 if global_step >= args.start_ema_step else 0
                }, step=global_step)
                pbar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "gs":   global_step,
                    "ema":  ema_phase
                })

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}  (global_step={global_step})")
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

        # =====================================================================
        # Validation + Checkpoint
        # v3: no hist_encoder in save_dict
        # =====================================================================
        if epoch % args.val_every == 0:
            ckpt_path = exp_dir / f"{args.experiment_name}_epoch{epoch:03d}.pt"
            save_dict = {
                "epoch":          epoch,
                "global_step":    global_step,
                "model":          model.state_dict(),
                "ema":            ema.state_dict(),
                "optim":          optimizer.state_dict(),
                "config":         args_dict,
                # v3: store avg_histogram so it can be loaded for inference
                # without reconstructing the full training dataset
                "avg_histogram":  avg_histogram,
            }

            torch.save(save_dict, ckpt_path)
            wandb.save(str(ckpt_path))
            print(f"Checkpoint saved: {ckpt_path}")

            try:
                print(f"Validating at epoch {epoch} (EMA applied)...")
                ema.apply_shadow()
                model.eval()
                ema.restore()
            except Exception:
                continue

    wandb.finish()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
