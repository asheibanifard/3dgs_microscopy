"""
Evaluation metrics for 3D volume reconstruction.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute PSNR from MSE.
    
    PSNR = 10 * log10((data_range^2) / MSE)
    
    Args:
        mse: Mean squared error (tensor or scalar)
        data_range: Dynamic range of the data (default: 1.0 for normalized data)
        
    Returns:
        PSNR in dB
    """
    return float(10.0 * torch.log10((data_range * data_range) / (mse + 1e-12)))


def ssim_3d_slicewise(
    pred: torch.Tensor, 
    tgt: torch.Tensor, 
    data_range: float = 1.0
) -> float:
    """
    Compute slice-wise SSIM averaged over Z dimension.
    
    Computes 2D SSIM for each Z slice and averages, which is more
    stable than full 3D SSIM for volumetric data.
    
    Args:
        pred: Predicted volume [B, Z, X, Y, C]
        tgt: Target volume [B, Z, X, Y, C]
        data_range: Dynamic range of the data
        
    Returns:
        Average SSIM across all slices
    """
    pred_np = pred[0].detach().cpu().numpy()  # [Z, X, Y, C]
    tgt_np = tgt[0].detach().cpu().numpy()

    if pred_np.shape[-1] == 1:
        # Single channel: squeeze out channel dimension
        pred_np = pred_np[..., 0]  # [Z, X, Y]
        tgt_np = tgt_np[..., 0]
        scores = [
            compare_ssim(pred_np[z], tgt_np[z], data_range=data_range) 
            for z in range(pred_np.shape[0])
        ]
    else:
        # Multi-channel
        scores = [
            compare_ssim(pred_np[z], tgt_np[z], data_range=data_range, channel_axis=-1)
            for z in range(pred_np.shape[0])
        ]
    
    return float(np.mean(scores))
