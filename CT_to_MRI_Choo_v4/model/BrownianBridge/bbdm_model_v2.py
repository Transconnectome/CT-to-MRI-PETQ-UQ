"""
Brownian Bridge Diffusion Model v2 with ISTA (Inter-Slice Trajectory Alignment).

Based on bbdm_model.py with ISTA support ported from Choo et al. 2024 (official).
Designed for use with v2 training (CT histogram, HistogramEncoder, context [B, 1, 128]).

ISTA reduces inter-slice inconsistency in 3D MRI reconstruction by:
1. Averaging adjacent slice trajectories at each diffusion step
2. Applying a score-based correction to maintain per-slice fidelity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Tuple

from utils.common import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel


class BrownianBridgeModel(nn.Module):
    """
    Brownian Bridge Diffusion Model with optional ISTA inference.

    Key features:
    - Bridges from source (CT) to target (MRI) using Brownian Bridge process
    - 2.5D slices: uses 3 consecutive slices
    - Histogram context for style conditioning (v2: CT histogram via HistogramEncoder)
    - ISTA (Inter-Slice Trajectory Alignment) for 3D consistency at inference
    """

    def __init__(self,
                 # Model architecture
                 image_size: int = 160,
                 in_channels: int = 6,  # 3 CT slices + 3 conditioning
                 out_channels: int = 3,  # 3 MRI slices
                 model_channels: int = 128,
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int, ...] = (32, 16, 8),
                 channel_mult: Tuple[int, ...] = (1, 4, 8),
                 num_heads: int = 8,
                 num_head_channels: int = 64,
                 # Brownian Bridge parameters
                 num_timesteps: int = 1000,
                 mt_type: str = 'linear',
                 max_var: float = 1.0,
                 eta: float = 0.0,  # DDIM eta
                 objective: str = 'grad',  # 'grad', 'noise', 'ysubx'
                 loss_type: str = 'l1',
                 # Sampling
                 skip_sample: bool = True,
                 sample_type: str = 'linear',
                 sample_step: int = 50,
                 # Context
                 use_context: bool = True,
                 context_dim: int = 128,
                 condition_key: str = 'hist_context_y_concat',
                 # ISTA parameters (inference only)
                 num_ISTA_step: int = 1,
                 ISTA_step_size: float = 0.5):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        self.mt_type = mt_type
        self.max_var = max_var
        self.eta = eta
        self.objective = objective
        self.loss_type = loss_type
        self.skip_sample = skip_sample
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.use_context = use_context
        self.condition_key = condition_key
        self.num_ISTA_step = num_ISTA_step
        self.ISTA_step_size = ISTA_step_size

        # Register Brownian Bridge schedule
        self.register_schedule()

        # U-Net denoising network
        self.denoise_fn = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_spatial_transformer=use_context,
            context_dim=context_dim if use_context else None,
            condition_key='nocond' if use_context else 'concat',
            use_scale_shift_norm=True,
            resblock_updown=True,
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
        )

    def register_schedule(self):
        """Register Brownian Bridge diffusion schedule"""
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        elif self.mt_type == 'control':
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
            m_t = np.sin(m_t * np.pi / 2)
        else:
            raise NotImplementedError(f"mt_type {self.mt_type} not implemented")

        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps).long()
        else:
            self.steps = torch.arange(self.num_timesteps - 1, -1, -1)

    def q_sample(self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: sample from q(x_t | x0, y)."""
        noise = default(noise, lambda: torch.randn_like(x0))

        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        x_t = (1. - m_t) * x0 + m_t * y + sigma_t * noise

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError(f"Objective {self.objective} not implemented")

        return x_t, objective

    def predict_x0_from_objective(self, x_t: torch.Tensor, y: torch.Tensor,
                                   t: torch.Tensor, objective_recon: torch.Tensor) -> torch.Tensor:
        """Predict x0 from objective reconstruction."""
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    def p_losses(self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                 context: Optional[torch.Tensor] = None,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Training loss computation."""
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        x_in = torch.cat([x_t, y], dim=1)  # [B, 6, H, W]
        objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)

        if self.loss_type == 'l1':
            loss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {"loss": loss.item(), "x0_recon": x0_recon.detach()}

        return loss, log_dict

    def forward(self, x0: torch.Tensor, y: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for training."""
        b, c, h, w = x0.shape
        assert h == self.image_size and w == self.image_size, \
            f"Input size mismatch: expected ({self.image_size}, {self.image_size}), got ({h}, {w})."

        device = x0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x0, y, t, context)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, y: torch.Tensor, i: int,
                 context: Optional[torch.Tensor] = None,
                 clip_denoised: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single reverse diffusion step: sample from p(x_{t-1} | x_t, y)."""
        b, *_, device = *x_t.shape, x_t.device

        if self.steps[i] == 0:
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            x_in = torch.cat([x_t, y], dim=1)
            objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

            if clip_denoised:
                x0_recon = x0_recon.clamp(-1., 1.)

            return x0_recon, x0_recon
        else:
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            n_t = torch.full((b,), self.steps[i + 1], device=device, dtype=torch.long)

            x_in = torch.cat([x_t, y], dim=1)
            objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

            if clip_denoised:
                x0_recon = x0_recon.clamp(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)

            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + \
                            torch.sqrt((var_nt - sigma2_t) / var_t) * (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_sub_batch(self, x_t: torch.Tensor, y: torch.Tensor,
                            i: int, context: Optional[torch.Tensor],
                            clip_denoised: bool, sub_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient p_sample by splitting [D, C, H, W] into sub-batches.

        Used in ISTA mode where the full D-slice batch is needed for averaging
        but may be too large for the UNet in one forward pass.

        Args:
            x_t: [D, C, H, W] full volume batch
            y: [D, C, H, W] full CT volume batch
            i: Step index in self.steps
            context: [D, ...] histogram context
            clip_denoised: Clip x0 to [-1, 1]
            sub_batch_size: Max slices per UNet forward pass

        Returns:
            x_t_next: [D, C, H, W]
            x0_recon: [D, C, H, W]
        """
        full_batch = x_t.shape[0]

        if full_batch <= sub_batch_size:
            return self.p_sample(x_t, y, i, context, clip_denoised)

        x_t_recons = []
        x0_recons = []
        num_iter = int(np.ceil(full_batch / sub_batch_size))

        for n in range(num_iter):
            start_idx = n * sub_batch_size
            end_idx = min((n + 1) * sub_batch_size, full_batch)

            x_t_sub = x_t[start_idx:end_idx]
            y_sub = y[start_idx:end_idx]
            context_sub = context[start_idx:end_idx] if context is not None else None

            x_t_recon, x0_recon = self.p_sample(x_t_sub, y_sub, i, context_sub, clip_denoised)
            x_t_recons.append(x_t_recon)
            x0_recons.append(x0_recon)

        return torch.cat(x_t_recons, dim=0), torch.cat(x0_recons, dim=0)

    @torch.no_grad()
    def batch2avgvolume(self, batch_img: torch.Tensor, device: torch.device,
                        pad: bool = True) -> torch.Tensor:
        """
        Convert 2.5D batch to an overlap-averaged volume.

        For 3-channel 2.5D (prev/curr/next), each spatial position in the
        volume is covered by up to 3 consecutive slice windows. Averaging
        these overlapping estimates reduces inter-slice inconsistency.

        Args:
            batch_img: [D, C, H, W] batch (C=3 for 2.5D)
            device: Target device
            pad: If True, keep padded boundary; if False, crop to [D, H, W]

        Returns:
            averaged_volume: [D+2*radius, H, W] if pad else [D, H, W]
        """
        batch_size, ch_size, H, W = batch_img.shape
        radius = ch_size // 2  # = 1 for C=3

        padded_size = batch_size + (2 * radius)
        averaged_volume = torch.zeros((ch_size, padded_size, H, W), device=device)
        dup_slices = torch.ones(padded_size, dtype=torch.int32, device=device) * ch_size

        for ch in range(ch_size):
            averaged_volume[ch, ch:ch + batch_size] = batch_img[:, ch]
            dup_slices[ch] = ch + 1
            dup_slices[-ch - 1] = ch + 1

        averaged_volume = torch.sum(averaged_volume, dim=0, keepdim=False) / dup_slices[:, None, None]

        if not pad:
            averaged_volume = averaged_volume[radius:-radius]

        return averaged_volume  # [D+2, H, W] or [D, H, W]

    @torch.no_grad()
    def volume2batch(self, volume_img: torch.Tensor,
                     batch_img_shape: Tuple, device: torch.device) -> torch.Tensor:
        """
        Convert averaged volume back to 2.5D batch format.

        Inverse of batch2avgvolume (with pad=True). Each output slice [d, :, h, w]
        receives the volume value as its center channel and adjacent slices as neighbors.

        Args:
            volume_img: [D+2*radius, H, W] averaged volume
            batch_img_shape: Original batch shape (D, C, H, W)
            device: Target device

        Returns:
            batch_img: [D, C, H, W]
        """
        batch_size, ch_size, H, W = batch_img_shape
        radius = ch_size // 2

        padded_size = batch_size + (2 * radius)
        double_padded_size = batch_size + (4 * radius)

        batch_img = torch.zeros((ch_size, double_padded_size, H, W), device=device)

        for ch in range(ch_size - 1, -1, -1):
            batch_img[ch_size - 1 - ch, ch:ch + padded_size] = volume_img

        batch_img = batch_img[:, ch_size - 1:ch_size - 1 + batch_size].permute(1, 0, 2, 3)

        return batch_img  # [D, C, H, W]

    @torch.no_grad()
    def cal_score(self, x0: torch.Tensor, x_t: torch.Tensor,
                  y: torch.Tensor, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute score function for ISTA correction step.

        The score of the Brownian Bridge marginal q(x_t | x0, y) is:
          score_t = -( x_t - (1-m_t)*x0 - m_t*y ) / var_t

        Args:
            x0: Predicted x0 [D, C, H, W]
            x_t: Current noisy sample [D, C, H, W]
            y: Source CT [D, C, H, W]
            i: Step index in self.steps

        Returns:
            score_t: Score tensor [D, C, H, W]
            var_t_nt: Variance at step i, shape [1, 1, 1] (scalar-like for broadcasting)
        """
        t = torch.full((x0.shape[0],), self.steps[i], device=x0.device, dtype=torch.long)
        m_t = extract(self.m_t, t, x0.shape)           # [D, 1, 1, 1]
        var_t = extract(self.variance_t, t, x0.shape)  # [D, 1, 1, 1]
        var_t_nt = extract(self.variance_t_tminus, t, x0.shape)  # [D, 1, 1, 1]

        score_t = -((x_t - (1. - m_t) * x0 - m_t * y) / var_t)

        return score_t, var_t_nt[0]  # var_t_nt[0]: [1, 1, 1]

    @torch.no_grad()
    def p_sample_loop(self, y: torch.Tensor,
                      context: Optional[torch.Tensor] = None,
                      clip_denoised: bool = True,
                      inference_type: str = 'normal',
                      num_ISTA_step: int = 1,
                      ISTA_step_size: float = 0.5,
                      sub_batch_size: int = 48,
                      return_all: bool = False):
        """
        Full reverse diffusion loop with optional ISTA.

        inference_type options:
          'normal'       — independent slice-by-slice (default, backward-compatible)
          'average'      — passive averaging of adjacent slice trajectories
          'ISTA_average' — averaging + score-gradient correction (full averaged score)
          'ISTA_mid'     — averaging + score-gradient correction (center channel only)

        In ISTA modes, y must contain ALL D slices simultaneously [D, C, H, W].
        Sub-batching handles UNet memory via sub_batch_size.

        Args:
            y: Source CT [B_or_D, C, H, W] — D slices for ISTA, B slices for normal
            context: Histogram context [B_or_D, ...]
            clip_denoised: Clip intermediate x0 to [-1, 1]
            inference_type: Inference mode (see above)
            num_ISTA_step: ISTA correction iterations per diffusion step
            ISTA_step_size: ISTA step size (gamma multiplier)
            sub_batch_size: Max slices per UNet forward pass in ISTA mode
            return_all: If True, also return all intermediate steps

        Returns:
            img: Generated MRI [B_or_D, C, H, W]
        """
        device = y.device
        img = y.clone()

        all_steps = [img] if return_all else None

        for i in tqdm(range(len(self.steps)), desc='Sampling', leave=False):

            # ---- Forward step ----
            if inference_type == 'normal':
                # Slice-by-slice: standard p_sample (no sub-batching needed)
                img, x0_recon = self.p_sample(img, y, i, context, clip_denoised)
            else:
                # ISTA modes: full volume batch, use sub-batching for memory
                img, x0_recon = self.p_sample_sub_batch(img, y, i, context, clip_denoised, sub_batch_size)

            # ---- Post-processing step ----
            if inference_type == 'normal':
                pass

            elif inference_type == 'average':
                averaged_volume = self.batch2avgvolume(img, device, pad=True)
                img = self.volume2batch(averaged_volume, img.shape, device)

            elif inference_type in ('ISTA_average', 'ISTA_mid'):
                # Step 1: averaging
                averaged_volume = self.batch2avgvolume(img, device, pad=True)
                img = self.volume2batch(averaged_volume, img.shape, device)

                # Step 2: score-gradient correction (skipped on final step)
                if i < len(self.steps) - 1:
                    batch_size, ch_size, H, W = img.shape
                    radius = ch_size // 2
                    dim = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=device))

                    for _ in range(num_ISTA_step):
                        # Predict x0 at next timestep (i+1)
                        _, x0_recon = self.p_sample_sub_batch(
                            img, y, i + 1, context, clip_denoised, sub_batch_size
                        )
                        score, var_t_nt = self.cal_score(x0=x0_recon, x_t=img, y=y, i=i + 1)
                        # score: [D, C, H, W]   var_t_nt: [1, 1, 1]

                        # Compute score volume for averaging
                        if inference_type == 'ISTA_average':
                            score_volume = self.batch2avgvolume(score, device, pad=True)
                            # score_volume: [D+2, H, W]
                        else:  # ISTA_mid: use center channel only
                            score_volume = score[:, radius]  # [D, H, W]
                            score_volume = torch.cat((
                                score[0, :radius],          # [radius, H, W]
                                score_volume,               # [D, H, W]
                                score[-1, radius + 1:]      # [radius, H, W]
                            ), dim=0)                       # [D+2, H, W]

                        # Adaptive step size: gamma = step_size * var * (||img|| / ||score||^2)
                        score_l2_norm_squared = torch.sum(
                            torch.pow(score_volume, 2), dim=(1, 2), keepdim=True
                        )  # [D+2, 1, 1]
                        gamma = ISTA_step_size * var_t_nt * (dim / score_l2_norm_squared)
                        # gamma: [D+2, 1, 1]  (var_t_nt [1,1,1] broadcasts)

                        # Apply correction
                        gamma_score_batch = self.volume2batch(
                            gamma * score_volume, img.shape, device
                        )  # [D, C, H, W]
                        img = img + gamma_score_batch

            else:
                raise ValueError(f"Unknown inference_type: '{inference_type}'. "
                                 f"Choose from: 'normal', 'average', 'ISTA_average', 'ISTA_mid'")

            if return_all:
                all_steps.append(img)

        if return_all:
            return img, all_steps
        return img

    @torch.no_grad()
    def sample(self, y: torch.Tensor,
               context: Optional[torch.Tensor] = None,
               clip_denoised: bool = True,
               inference_type: str = 'normal',
               num_ISTA_step: int = 1,
               ISTA_step_size: float = 0.5,
               sub_batch_size: int = 48) -> torch.Tensor:
        """
        Sample from model (inference entry point).

        Args:
            y: Source CT — [B, C, H, W] for 'normal', [D, C, H, W] for ISTA modes
            context: Histogram context (same leading dim as y)
            clip_denoised: Clip intermediate x0 reconstructions
            inference_type: 'normal' | 'average' | 'ISTA_average' | 'ISTA_mid'
            num_ISTA_step: ISTA correction iterations per step
            ISTA_step_size: ISTA adaptive step size multiplier
            sub_batch_size: UNet sub-batch size for ISTA memory management

        Returns:
            Generated MRI (same shape as y)
        """
        return self.p_sample_loop(
            y, context, clip_denoised,
            inference_type, num_ISTA_step, ISTA_step_size, sub_batch_size
        )
