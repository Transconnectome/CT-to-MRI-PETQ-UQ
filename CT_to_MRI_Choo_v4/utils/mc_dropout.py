"""
MC Dropout utilities for Uncertainty Quantification (v4).

Provides:
  - MCDropout  : nn.Dropout subclass that stays active in model.eval() mode.
  - enable_mc_dropout : Recursively replaces all nn.Dropout instances in a
                        model with MCDropout, enabling test-time stochasticity.

Usage:
    from utils.mc_dropout import MCDropout, enable_mc_dropout

    # After loading and calling model.eval():
    enable_mc_dropout(model)

    # Now run model.sample() N times → different outputs each time.
    samples = [model.sample(y, context) for _ in range(num_mc_samples)]
    mean    = torch.stack(samples).mean(0)
    variance = torch.stack(samples).var(0)
"""

import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Dropout):
    """
    Dropout that remains ACTIVE regardless of model.train() / model.eval() state.

    Standard nn.Dropout is disabled when the model is in eval mode (training=False).
    MCDropout overrides forward() to always call F.dropout with training=True,
    bypassing the eval-mode disable behaviour.

    This is required for MC Dropout inference: we call model.eval() to disable
    BatchNorm / LayerNorm running-stat updates (if any), while keeping dropout
    active to sample from the approximate posterior.

    Note: the model uses GroupNorm32 (a GroupNorm variant), which is unaffected
    by train/eval mode. Therefore, calling model.eval() + enable_mc_dropout()
    is fully correct for this architecture.
    """

    def forward(self, x):
        # training=True is forced regardless of the module's self.training flag.
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Recursively replace every nn.Dropout instance in `model` with MCDropout.

    Only instances whose type is exactly nn.Dropout are replaced (not subclasses,
    to avoid accidentally replacing already-patched layers on repeated calls).

    Args:
        model: The nn.Module to patch in-place.

    Returns:
        The same model with all nn.Dropout layers replaced by MCDropout.
    """
    for name, child in model.named_children():
        if type(child) is nn.Dropout:
            # Preserve probability and inplace settings.
            setattr(model, name, MCDropout(p=child.p, inplace=child.inplace))
        else:
            # Recurse into submodules (handles Sequential, ModuleList, etc.).
            enable_mc_dropout(child)
    return model
