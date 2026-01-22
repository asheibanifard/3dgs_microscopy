"""
Anti-aliased downsampling utilities for 3D volumes.
Prevents grid/checkerboard artifacts during coarse-to-fine training.
"""

import torch
import torch.nn.functional as F


def create_gaussian_kernel_3d(kernel_size: int = 5, sigma: float = 1.0, device='cuda'):
    """
    Create a 3D Gaussian kernel for anti-aliasing.
    
    Args:
        kernel_size: Size of the kernel (should be odd)
        sigma: Standard deviation of the Gaussian
        device: Device to create the kernel on
        
    Returns:
        Gaussian kernel of shape [1, 1, K, K, K]
    """
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-coords**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_3d = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    return gauss_3d[None, None, ...]


def antialias_downsample_3d(
    volume: torch.Tensor,
    factor: int = 2,
    kernel: torch.Tensor = None,
    kernel_size: int = 5,
    sigma: float = None,
):
    """
    Properly downsample a 3D volume with Gaussian blur to prevent aliasing.
    
    This REPLACES the problematic strided subsampling:
        image[:, i::2, j::2, k::2, :]  # Creates grid artifacts!
    
    Args:
        volume: Input tensor of shape [B, Z, X, Y, C]
        factor: Downsampling factor
        kernel: Pre-computed Gaussian kernel (optional)
        kernel_size: Size of Gaussian kernel if not provided
        sigma: Sigma for Gaussian kernel (default: 0.35 * factor)
        
    Returns:
        Downsampled volume of shape [B, Z', X', Y', C]
    """
    B, Z, X, Y, C = volume.shape
    vol = volume.permute(0, 4, 1, 2, 3)  # [B, C, Z, X, Y]

    if sigma is None:
        # Gentler than factor/2, better for thin bright filaments
        sigma = 0.35 * factor

    if kernel is None:
        kernel = create_gaussian_kernel_3d(kernel_size=kernel_size, sigma=sigma, device=volume.device)

    kernel_expanded = kernel.expand(C, 1, -1, -1, -1)  # [C, 1, K, K, K]
    pad = kernel.shape[-1] // 2

    blurred = F.conv3d(vol, kernel_expanded, padding=pad, groups=C)
    down = blurred[:, :, ::factor, ::factor, ::factor]
    return down.permute(0, 2, 3, 4, 1)  # [B, Z', X', Y', C]


def antialias_downsample_grid(grid: torch.Tensor, factor: int = 2, target_size: tuple = None):
    """
    Downsample coordinate grid to match target size.
    
    Args:
        grid: Coordinate grid of shape [B, Z, X, Y, 3]
        factor: Downsampling factor
        target_size: Optional target size (Z', X', Y') to ensure exact match
        
    Returns:
        Downsampled grid of shape [B, Z', X', Y', 3]
    """
    B, Z, X, Y, C = grid.shape
    grid_permuted = grid.permute(0, 4, 1, 2, 3)  # [B, 3, Z, X, Y]
    
    if target_size is not None:
        down = F.interpolate(grid_permuted, size=target_size, mode='trilinear', align_corners=True)
    else:
        down = F.avg_pool3d(grid_permuted, factor, stride=factor)
    
    return down.permute(0, 2, 3, 4, 1)  # [B, Z', X', Y', 3]
