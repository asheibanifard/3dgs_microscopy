"""
Loss functions for 3D Gaussian Splatting training.
Includes background suppression, sparsity, concentration, and multi-scale losses.
"""

import torch
import torch.nn.functional as F

from .downsampling import antialias_downsample_3d


def background_suppression_loss(
    output: torch.Tensor, 
    target: torch.Tensor, 
    t0: float = 0.03, 
    sharp: float = 80.0
) -> torch.Tensor:
    """
    Softly penalize non-zero output where target is near-zero (background).
    
    Uses a soft sigmoid mask to avoid hard thresholding that could
    accidentally remove dim foreground signal.
    
    Args:
        output: Rendered output volume [B, Z, X, Y, C]
        target: Target volume [B, Z, X, Y, C]
        t0: Threshold for background detection
        sharp: Sharpness of the sigmoid transition
        
    Returns:
        Background suppression loss (scalar)
    """
    # bg ~ 1 in background, ~0 in foreground
    bg = torch.sigmoid((t0 - target) * sharp)
    return (output.pow(2) * bg).mean() + 0.2 * (output.abs() * bg).mean()


def sparsity_loss(
    output: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.02
) -> torch.Tensor:
    """
    Encourage output sparsity to match target sparsity.
    
    Penalizes when output has more "active" voxels than target.
    Uses abs(output) to handle potential negative values.
    
    Args:
        output: Rendered output volume
        target: Target volume
        threshold: Threshold for counting active voxels
        
    Returns:
        Sparsity loss (scalar, only positive when output is less sparse)
    """
    output_active = (output.abs() > threshold).float().mean()
    target_active = (target > threshold).float().mean()
    return F.relu(output_active - target_active)


def concentration_loss(gaussians) -> torch.Tensor:
    """
    Penalize overly large Gaussians that cause diffuse haze.
    
    Encourages tight, focused Gaussians by penalizing scales
    above a threshold.
    
    Args:
        gaussians: GaussianModel instance with get_scaling property/method
        
    Returns:
        Concentration loss (scalar)
    """
    # Handle both property and callable
    scales = gaussians.get_scaling() if callable(getattr(gaussians, "get_scaling", None)) else gaussians.get_scaling
    # scales: [N, 3]
    max_scales = scales.max(dim=1)[0]
    scale_threshold = 0.03  # ~3% of volume
    return F.relu(max_scales - scale_threshold).mean()


def multi_scale_mse_loss(
    output: torch.Tensor, 
    target: torch.Tensor, 
    weights: tuple = (1.0, 0.3), 
    kernel: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute MSE at multiple scales to damp residual aliasing/checkerboard.
    
    Lower scales average out grid artifacts, providing smoother gradients.
    
    Args:
        output: Rendered output [B, Z, X, Y, C]
        target: Target volume [B, Z, X, Y, C]
        weights: Weights for each scale level
        kernel: Pre-computed Gaussian kernel for downsampling
        
    Returns:
        Weighted multi-scale MSE loss
    """
    total = 0.0
    out, tgt = output, target
    wsum = float(sum(weights))
    
    for i, w in enumerate(weights):
        total = total + float(w) * F.mse_loss(out, tgt)
        if i < len(weights) - 1:
            out = antialias_downsample_3d(out, factor=2, kernel=kernel)
            tgt = antialias_downsample_3d(tgt, factor=2, kernel=kernel)
    
    return total / wsum


@torch.jit.script
def tv_regularization(image: torch.Tensor) -> torch.Tensor:
    """
    Total Variation regularization for 3D volumes.
    
    Encourages smooth transitions in the output, reducing noise.
    
    Args:
        image: Input volume [B, Z, X, Y, C]
        
    Returns:
        TV loss (scalar)
    """
    diff_z = image[:, 1:, :, :, :] - image[:, :-1, :, :, :]
    diff_x = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
    diff_y = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
    return diff_z.abs().mean() + diff_x.abs().mean() + diff_y.abs().mean()
