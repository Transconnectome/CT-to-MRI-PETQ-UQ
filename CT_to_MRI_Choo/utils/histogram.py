"""
Histogram-based Style Key Conditioning utilities.
Implements the histogram context feature extraction for style conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class HistogramExtractor:
    """
    Extract histogram features from medical images for style conditioning.
    Based on Choo et al. 2024 Style Key Conditioning (SKC).
    """

    def __init__(self,
                 num_bins: int = 256,
                 value_range: Tuple[float, float] = (-1.0, 1.0),
                 normalize: bool = True):
        """
        Args:
            num_bins: Number of histogram bins
            value_range: Value range for histogram computation
            normalize: Whether to normalize histogram to sum to 1
        """
        self.num_bins = num_bins
        self.value_range = value_range
        self.normalize = normalize

    def compute_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute histogram for a batch of images.

        Args:
            x: Input tensor [B, C, H, W] or [B, C, D, H, W]

        Returns:
            Histogram features [B, num_bins]
        """
        # Flatten spatial dimensions
        if x.dim() == 5:  # 3D volume
            x_flat = x.flatten(2)  # [B, C, D*H*W]
        else:  # 2D image
            x_flat = x.flatten(2)  # [B, C, H*W]

        # Average across channels
        x_flat = x_flat.mean(dim=1)  # [B, D*H*W] or [B, H*W]

        # Compute histogram for each sample in batch
        batch_size = x_flat.size(0)
        histograms = []

        for i in range(batch_size):
            hist = torch.histc(
                x_flat[i],
                bins=self.num_bins,
                min=self.value_range[0],
                max=self.value_range[1]
            )
            if self.normalize:
                hist = hist / (hist.sum() + 1e-8)
            histograms.append(hist)

        return torch.stack(histograms, dim=0)  # [B, num_bins]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method"""
        return self.compute_histogram(x)


class HistogramEncoder(nn.Module):
    """
    Encode histogram features to context embeddings.
    Maps histogram → embedding suitable for cross-attention.
    """

    def __init__(self,
                 num_bins: int = 256,
                 embed_dim: int = 128,
                 hidden_dim: int = 512):
        """
        Args:
            num_bins: Number of histogram bins (input dim)
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_bins = num_bins
        self.embed_dim = embed_dim

        # MLP to encode histogram to embedding
        self.encoder = nn.Sequential(
            nn.Linear(num_bins, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Encode histogram to embedding.

        Args:
            hist: Histogram features [B, num_bins]

        Returns:
            Embeddings [B, 1, embed_dim] for cross-attention
        """
        emb = self.encoder(hist)  # [B, embed_dim]
        return emb.unsqueeze(1)  # [B, 1, embed_dim]


def create_global_histogram_reference(
    dataset,
    num_bins: int = 256,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    num_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Create a global histogram reference from dataset.
    This can be used as a canonical style reference.

    Args:
        dataset: PyTorch dataset
        num_bins: Number of histogram bins
        value_range: Value range
        num_samples: Number of samples to use (None = all)

    Returns:
        Global histogram [num_bins]
    """
    extractor = HistogramExtractor(num_bins, value_range, normalize=False)

    global_hist = torch.zeros(num_bins)
    count = 0

    indices = range(len(dataset)) if num_samples is None else range(min(num_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        # Handle different dataset return formats
        if isinstance(sample, dict):
            x = sample['ct'] if 'ct' in sample else sample['x']
        else:
            x = sample[0]

        if x.dim() == 3:  # [D, H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif x.dim() == 4:  # [C, D, H, W] or [1, D, H, W]
            x = x.unsqueeze(0)  # [1, C, D, H, W]

        hist = extractor(x)[0]  # [num_bins]
        global_hist += hist
        count += 1

    # Normalize
    global_hist = global_hist / (global_hist.sum() + 1e-8)

    return global_hist
