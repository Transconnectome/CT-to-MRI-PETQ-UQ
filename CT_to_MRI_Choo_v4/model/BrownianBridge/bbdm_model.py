"""
Brownian Bridge Diffusion Model for 2.5D CT-to-MRI Translation.
Based on Choo et al. 2024 MICCAI.
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
    Brownian Bridge Diffusion Model for image-to-image translation.

    Key features:
    - Bridges from source (CT) to target (MRI) using Brownian Bridge process
    - 2.5D slices: uses 3 consecutive slices
    - Histogram context for style conditioning
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
                 condition_key: str = 'hist_context_y_concat'):
        """
        Args:
            image_size: Input image size
            in_channels: Input channels (6 for 2.5D: 3 source + 3 conditioning)
            out_channels: Output channels (3 for 2.5D MRI slices)
            model_channels: Base channel width for U-Net
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions to apply attention
            channel_mult: Channel multiplier per resolution
            num_heads: Number of attention heads
            num_head_channels: Channels per attention head
            num_timesteps: Number of diffusion timesteps
            mt_type: Schedule type ('linear', 'sin', 'control')
            max_var: Maximum variance for Brownian Bridge
            eta: DDIM eta parameter (0 = deterministic)
            objective: Training objective
            loss_type: Loss function type
            skip_sample: Use DDIM sampling (skip timesteps)
            sample_type: DDIM sampling schedule
            sample_step: Number of sampling steps
            use_context: Use histogram context
            context_dim: Context embedding dimension
            condition_key: Conditioning method
        """
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
            condition_key='nocond' if use_context else 'concat',  # Use cross-attention only, no concat
            use_scale_shift_norm=True,
            resblock_updown=True,
            conv_resample=True,
            dims=2,  # 2D U-Net for 2.5D slices
            use_checkpoint=False,  # Disable gradient checkpointing to avoid BFloat16 issues
        )

    def register_schedule(self):
        """Register Brownian Bridge diffusion schedule"""
        T = self.num_timesteps

        # Create m_t schedule (bridge coefficient from source to target)
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

        # Variance schedule for Brownian Bridge
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        # Convert to torch tensors
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        # DDIM sampling steps
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
        """
        Forward diffusion process: sample from q(x_t | x0, y).
        Brownian Bridge from x0 (target MRI) to y (source CT).

        Args:
            x0: Target (MRI) [B, 3, H, W]
            y: Source (CT) [B, 3, H, W]
            t: Timestep [B]
            noise: Optional noise

        Returns:
            x_t: Noisy sample at timestep t
            objective: Training objective (grad, noise, or ysubx)
        """
        noise = default(noise, lambda: torch.randn_like(x0))

        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        # Brownian Bridge forward process
        x_t = (1. - m_t) * x0 + m_t * y + sigma_t * noise

        # Compute objective based on parameterization
        if self.objective == 'grad':
            # Predict gradient direction
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            # Predict noise
            objective = noise
        elif self.objective == 'ysubx':
            # Predict y - x0
            objective = y - x0
        else:
            raise NotImplementedError(f"Objective {self.objective} not implemented")

        return x_t, objective

    def predict_x0_from_objective(self, x_t: torch.Tensor, y: torch.Tensor,
                                   t: torch.Tensor, objective_recon: torch.Tensor) -> torch.Tensor:
        """
        Predict x0 from objective reconstruction.

        Args:
            x_t: Noisy sample
            y: Source (CT)
            t: Timestep
            objective_recon: Reconstructed objective

        Returns:
            x0_recon: Reconstructed x0 (MRI)
        """
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
        """
        Training loss computation.

        Args:
            x0: Target MRI [B, 3, H, W]
            y: Source CT [B, 3, H, W]
            t: Timestep [B]
            context: Histogram context [B, 1, context_dim]
            noise: Optional noise

        Returns:
            loss: Training loss
            log_dict: Additional logging info
        """
        noise = default(noise, lambda: torch.randn_like(x0))

        # Forward process
        x_t, objective = self.q_sample(x0, y, t, noise)

        # Concatenate x_t with conditioning y
        x_in = torch.cat([x_t, y], dim=1)  # [B, 6, H, W]

        # Predict objective
        objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)

        # Compute loss
        if self.loss_type == 'l1':
            loss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

        # Reconstruct x0 for monitoring
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

        log_dict = {
            "loss": loss.item(),
            "x0_recon": x0_recon.detach()
        }

        return loss, log_dict

    def forward(self, x0: torch.Tensor, y: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for training.

        Args:
            x0: Target MRI [B, 3, H, W]
            y: Source CT [B, 3, H, W]
            context: Histogram context

        Returns:
            loss: Training loss
            log_dict: Additional info
        """
        b, c, h, w = x0.shape
        assert h == self.image_size and w == self.image_size, \
            f"Input size mismatch: expected ({self.image_size}, {self.image_size}), got ({h}, {w}). " \
            f"Make sure dataset target_size matches model image_size."

        # Sample random timesteps
        device = x0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x0, y, t, context)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, y: torch.Tensor, i: int,
                 context: Optional[torch.Tensor] = None,
                 clip_denoised: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single reverse diffusion step: sample from p(x_{t-1} | x_t, y).

        Args:
            x_t: Current noisy sample [B, 3, H, W]
            y: Source CT [B, 3, H, W]
            i: Step index in self.steps
            context: Histogram context
            clip_denoised: Clip reconstructed x0 to [-1, 1]

        Returns:
            x_tminus: Next sample
            x0_recon: Reconstructed x0
        """
        b, *_, device = *x_t.shape, x_t.device

        # Get current and next timestep
        if self.steps[i] == 0:
            # Final step: directly return x0
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            x_in = torch.cat([x_t, y], dim=1)
            objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

            if clip_denoised:
                x0_recon = x0_recon.clamp(-1., 1.)

            return x0_recon, x0_recon
        else:
            # Intermediate step
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            n_t = torch.full((b,), self.steps[i + 1], device=device, dtype=torch.long)

            # Predict objective and reconstruct x0
            x_in = torch.cat([x_t, y], dim=1)
            objective_recon = self.denoise_fn(x_in, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

            if clip_denoised:
                x0_recon = x0_recon.clamp(-1., 1.)

            # Compute posterior mean and variance
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)

            # DDIM formula
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + \
                            torch.sqrt((var_nt - sigma2_t) / var_t) * (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y: torch.Tensor,
                      context: Optional[torch.Tensor] = None,
                      clip_denoised: bool = True,
                      return_all: bool = False) -> torch.Tensor:
        """
        Full reverse diffusion loop.

        Args:
            y: Source CT [B, 3, H, W]
            context: Histogram context
            clip_denoised: Clip intermediate results
            return_all: Return all intermediate steps

        Returns:
            x0: Generated MRI [B, 3, H, W]
            (Optional) all_steps: List of intermediate steps
        """
        device = y.device
        b, c, h, w = y.shape

        # Start from pure noise (or from y for Brownian Bridge)
        img = y.clone()  # Start from source

        all_steps = [img] if return_all else None

        for i in tqdm(range(len(self.steps)), desc='Sampling', leave=False):
            img, x0_recon = self.p_sample(img, y, i, context, clip_denoised)

            if return_all:
                all_steps.append(img)

        if return_all:
            return img, all_steps
        return img

    @torch.no_grad()
    def sample(self, y: torch.Tensor, context: Optional[torch.Tensor] = None,
               clip_denoised: bool = True) -> torch.Tensor:
        """
        Sample from model (inference).

        Args:
            y: Source CT [B, 3, H, W]
            context: Histogram context

        Returns:
            Generated MRI [B, 3, H, W]
        """
        return self.p_sample_loop(y, context, clip_denoised)
