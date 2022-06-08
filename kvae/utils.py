import numpy as np
import torch
import torch.nn.functional as F

def nll_gaussian_var_fixed(preds, target, variance, add_const=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))