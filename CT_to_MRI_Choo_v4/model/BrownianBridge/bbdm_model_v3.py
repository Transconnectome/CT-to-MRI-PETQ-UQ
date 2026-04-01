"""
Brownian Bridge Diffusion Model v3 with ISTA (Inter-Slice Trajectory Alignment).

Functionally identical to bbdm_model_v2. Re-exported here for use with v3 training
(MR histogram, no HistogramEncoder, context shape [B, 128, 3, 1] -> [B, 3, 128]).

The model architecture is context-shape-agnostic: attention.py automatically
rearranges 4D context [B, C, H, W] -> [B, H*W, C] before cross-attention.
"""
from model.BrownianBridge.bbdm_model_v2 import BrownianBridgeModel  # noqa: F401

__all__ = ['BrownianBridgeModel']
