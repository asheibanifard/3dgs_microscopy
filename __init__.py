# Training modules for 3D Gaussian Splatting CT Reconstruction
from .downsampling import (
    create_gaussian_kernel_3d,
    antialias_downsample_3d,
    antialias_downsample_grid,
)
from .losses import (
    background_suppression_loss,
    sparsity_loss,
    concentration_loss,
    multi_scale_mse_loss,
    tv_regularization,
)
from .pruning import prune_background_gaussians
from .scheduling import get_resolution_blend_weights
from .metrics import psnr_from_mse, ssim_3d_slicewise
from .optimization import OptimizationParams
from .renderer import (
    GaussianRenderer,
    CameraParams,
    create_camera_from_orbit,
    create_camera_intrinsics,
    create_view_matrix,
    create_projection_matrix,
    render_multiple_views,
    render_mip_projection,
    normalize_gaussian_positions,
    denormalize_gaussian_positions,
    # MIP and Alpha Blending
    render_mip_orthographic,
    render_alpha_blending_orthographic,
    render_gaussians_orthographic,
    render_sum_orthographic,
    render_all_views,
    render_from_checkpoint,
)

__all__ = [
    # Downsampling
    'create_gaussian_kernel_3d',
    'antialias_downsample_3d',
    'antialias_downsample_grid',
    # Losses
    'background_suppression_loss',
    'sparsity_loss',
    'concentration_loss',
    'multi_scale_mse_loss',
    'tv_regularization',
    # Pruning
    'prune_background_gaussians',
    # Scheduling
    'get_resolution_blend_weights',
    # Metrics
    'psnr_from_mse',
    'ssim_3d_slicewise',
    # Optimization
    'OptimizationParams',
    # Renderer
    'GaussianRenderer',
    'CameraParams',
    'create_camera_from_orbit',
    'create_camera_intrinsics',
    'create_view_matrix',
    'create_projection_matrix',
    'render_multiple_views',
    'render_mip_projection',
    'normalize_gaussian_positions',
    'denormalize_gaussian_positions',
    # MIP and Alpha Blending
    'render_mip_orthographic',
    'render_alpha_blending_orthographic',
    'render_gaussians_orthographic',
    'render_sum_orthographic',
    'render_all_views',
    'render_from_checkpoint',
]
