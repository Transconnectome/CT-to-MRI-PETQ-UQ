"""
Common utility functions for CT2MRI Brownian Bridge implementation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """
    Extract values from tensor a at indices t and reshape to match x_shape.

    Args:
        a: Source tensor (1D)
        t: Indices (batch)
        x_shape: Target shape

    Returns:
        Extracted and reshaped tensor
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    """Return val if it exists, otherwise return d (can be a lambda)"""
    if val is not None:
        return val
    return d() if callable(d) else d


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        except AttributeError:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def make_beta_schedule(schedule: str = "linear",
                       n_timestep: int = 1000,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2,
                       cosine_s: float = 8e-3) -> np.ndarray:
    """
    Create beta schedule for diffusion process.

    Args:
        schedule: Schedule type ('linear', 'cosine')
        n_timestep: Number of timesteps
        linear_start: Starting beta for linear schedule
        linear_end: Ending beta for linear schedule
        cosine_s: Small constant for cosine schedule

    Returns:
        Beta schedule array
    """
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'cosine':
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError(f"Schedule {schedule} not implemented")
    return betas


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_random_seed(seed: int = 1234):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For performance


def min_max_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Min-max normalize tensor to [0, 1]"""
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min > 1e-8:
        return (tensor - t_min) / (t_max - t_min)
    return tensor


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] to [-1, 1]"""
    return x * 2.0 - 1.0


def to_0_1(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] to [0, 1]"""
    return (x + 1.0) / 2.0
