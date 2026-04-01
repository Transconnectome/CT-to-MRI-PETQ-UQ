"""
Utility functions for the Brownian Bridge model.
Adapted from LDM and other diffusion model codebases.
"""

import importlib
import torch
import numpy as np


def instantiate_from_config(config):
    """
    Instantiate an object from a config dictionary.

    Args:
        config: Dictionary or OmegaConf config with 'target' key specifying the class
                and optional 'params' key for initialization parameters.

    Returns:
        Instantiated object
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)

    params = config.get("params", dict())
    return cls(**params)


def get_obj_from_str(string, reload=False):
    """
    Get object from string path.

    Args:
        string: String path to object (e.g., "torch.nn.Conv2d")
        reload: Whether to reload the module

    Returns:
        Object referenced by the string
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt, verbose=False):
    """
    Load model from config and checkpoint.

    Args:
        config: Model configuration
        ckpt: Path to checkpoint file
        verbose: Whether to print loading information

    Returns:
        Loaded model
    """
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def count_params(model, verbose=False):
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        verbose: Whether to print detailed information

    Returns:
        Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def ismap(x):
    """Check if input is a map (dict-like)."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    """Check if input is an image tensor."""
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    """Check if variable exists and is not None."""
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d."""
    if exists(val):
        return val
    return d() if callable(d) else d


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
