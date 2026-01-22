"""
Gaussian pruning utilities for removing background/noisy Gaussians.
"""

import torch
import torch.nn.functional as F


def prune_background_gaussians(
    gaussians,
    target_volume: torch.Tensor,
    intensity_threshold: float = 0.01,
    target_threshold_interior: float = 0.05,
    target_threshold_boundary: float = 0.10,
    boundary_margin: float = 0.05,
) -> int:
    """
    Remove Gaussians that sit in background regions according to target samples.
    
    Uses stricter criteria for Gaussians near volume boundaries to eliminate
    edge artifacts visible at grazing angles.
    
    Args:
        gaussians: GaussianModel instance
        target_volume: Target volume [B, Z, X, Y, C]
        intensity_threshold: Minimum intensity threshold (unused currently)
        target_threshold_interior: Target brightness threshold for interior Gaussians
        target_threshold_boundary: Target brightness threshold for boundary Gaussians
        boundary_margin: Fraction of volume considered "boundary" (0.05 = 5%)
        
    Returns:
        Number of pruned Gaussians
    """
    with torch.no_grad():
        xyz = gaussians.get_xyz  # [N, 3] in [0, 1]
        if xyz.numel() == 0:
            return 0

        # grid_sample expects coords in [-1, 1] and layout [B, D_out, H_out, W_out, 3]
        sample_coords = (xyz * 2 - 1).view(1, 1, 1, -1, 3)  # [1, 1, 1, N, 3]

        # target_volume: [B, Z, X, Y, C] -> [B, C, Z, X, Y]
        tgt = target_volume.permute(0, 4, 1, 2, 3)

        samples = F.grid_sample(
            tgt, sample_coords,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )  # [B, C, 1, 1, N]

        samples = samples[0, :, 0, 0, :]       # [C, N]
        target_at_gaussians = samples.mean(0)  # [N]

        # Get Gaussian intensities
        intensities = gaussians.get_intensity.squeeze()
        if intensities.ndim > 1:
            intensities = intensities.mean(dim=-1)

        # Check if near boundary (within margin of edges)
        near_boundary = ((xyz < boundary_margin) | (xyz > 1 - boundary_margin)).any(dim=1)

        # Keep criteria:
        # - Interior: target is bright OR Gaussian has decent intensity
        # - Boundary: target must be bright AND Gaussian has decent intensity (stricter)
        keep_interior = (target_at_gaussians > target_threshold_interior) | (intensities.abs() > 0.3)
        keep_boundary = (target_at_gaussians > target_threshold_boundary) & (intensities.abs() > 0.2)
        keep_mask = torch.where(near_boundary, keep_boundary, keep_interior)

        n_pruned = int((~keep_mask).sum().item())
        if n_pruned > 0:
            gaussians.prune_points(~keep_mask)
        
        return n_pruned
