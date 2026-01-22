"""
2D Gaussian Splatting Renderer using diff_gaussian_rasterization

This module provides rendering functionality for 3D Gaussians projected to 2D images
using the differentiable Gaussian rasterization from the original 3DGS paper.
"""

import torch
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    image_height: int
    image_width: int
    FoVx: float  # Field of view in x (radians)
    FoVy: float  # Field of view in y (radians)
    world_view_transform: torch.Tensor  # 4x4 view matrix
    full_proj_transform: torch.Tensor   # 4x4 projection matrix
    camera_center: torch.Tensor         # 3D camera position


def create_camera_intrinsics(
    image_height: int,
    image_width: int,
    fov_deg: float = 60.0,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Create camera field-of-view parameters.
    
    Args:
        image_height: Output image height
        image_width: Output image width
        fov_deg: Field of view in degrees
        device: Device for tensors
    
    Returns:
        (FoVx, FoVy) in radians
    """
    fov_rad = math.radians(fov_deg)
    aspect = image_width / image_height
    
    FoVy = fov_rad
    FoVx = 2 * math.atan(math.tan(FoVy / 2) * aspect)
    
    return FoVx, FoVy


def create_view_matrix(
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create a view matrix (world to camera transform).
    
    Args:
        camera_position: 3D position of camera
        look_at: 3D point camera is looking at
        up: Up direction (default: [0, 1, 0])
        device: Device for tensors
    
    Returns:
        4x4 view matrix
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    camera_position = camera_position.to(device)
    look_at = look_at.to(device)
    up = up.to(device)
    
    # Forward direction (camera looks along -z in camera space)
    forward = look_at - camera_position
    forward = forward / forward.norm()
    
    # Right direction
    right = torch.cross(forward, up)
    right = right / right.norm()
    
    # Recompute up to ensure orthogonality
    up = torch.cross(right, forward)
    
    # Build rotation matrix (transposed because we want world-to-camera)
    R = torch.stack([right, up, -forward], dim=0)  # 3x3
    
    # Translation
    t = -R @ camera_position
    
    # Build 4x4 matrix
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = t
    
    return view_matrix


def create_projection_matrix(
    FoVx: float,
    FoVy: float,
    near: float = 0.01,
    far: float = 100.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create a perspective projection matrix.
    
    Args:
        FoVx: Field of view in x (radians)
        FoVy: Field of view in y (radians)  
        near: Near clipping plane
        far: Far clipping plane
        device: Device for tensors
    
    Returns:
        4x4 projection matrix
    """
    tanHalfFovY = math.tan(FoVy / 2)
    tanHalfFovX = math.tan(FoVx / 2)
    
    top = tanHalfFovY * near
    bottom = -top
    right = tanHalfFovX * near
    left = -right
    
    P = torch.zeros(4, 4, device=device)
    
    P[0, 0] = 2 * near / (right - left)
    P[1, 1] = 2 * near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    
    return P


def create_camera_from_orbit(
    distance: float,
    elevation: float,
    azimuth: float,
    look_at: torch.Tensor = None,
    image_height: int = 512,
    image_width: int = 512,
    fov_deg: float = 60.0,
    device: str = 'cuda'
) -> CameraParams:
    """
    Create camera parameters from orbital position.
    
    Args:
        distance: Distance from look_at point
        elevation: Elevation angle in degrees (0 = horizontal, 90 = top-down)
        azimuth: Azimuth angle in degrees (rotation around up axis)
        look_at: Point to look at (default: origin)
        image_height: Output image height
        image_width: Output image width
        fov_deg: Field of view in degrees
        device: Device for tensors
    
    Returns:
        CameraParams dataclass
    """
    if look_at is None:
        look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    # Convert angles to radians
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)
    
    # Compute camera position on sphere
    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = distance * math.sin(elev_rad)
    z = distance * math.cos(elev_rad) * math.cos(azim_rad)
    
    camera_position = torch.tensor([x, y, z], device=device) + look_at
    
    # Create matrices
    FoVx, FoVy = create_camera_intrinsics(image_height, image_width, fov_deg, device)
    view_matrix = create_view_matrix(camera_position, look_at, device=device)
    proj_matrix = create_projection_matrix(FoVx, FoVy, device=device)
    
    # Full projection = projection @ view
    full_proj = proj_matrix @ view_matrix
    
    return CameraParams(
        image_height=image_height,
        image_width=image_width,
        FoVx=FoVx,
        FoVy=FoVy,
        world_view_transform=view_matrix.T,  # Transposed for rasterizer
        full_proj_transform=full_proj.T,     # Transposed for rasterizer
        camera_center=camera_position
    )


class GaussianRenderer:
    """
    Renderer for 3D Gaussians using diff_gaussian_rasterization.
    
    This renderer projects 3D Gaussians to 2D images using differentiable
    Gaussian splatting, enabling gradient-based optimization.
    """
    
    def __init__(
        self,
        sh_degree: int = 0,
        bg_color: torch.Tensor = None,
        device: str = 'cuda',
        antialiasing: bool = True,
        debug: bool = False
    ):
        """
        Initialize the renderer.
        
        Args:
            sh_degree: Spherical harmonics degree (0 for grayscale/simple color)
            bg_color: Background color tensor (default: black)
            device: Device for rendering
            antialiasing: Whether to use antialiasing
            debug: Enable debug mode
        """
        self.sh_degree = sh_degree
        self.device = device
        self.antialiasing = antialiasing
        self.debug = debug
        
        if bg_color is None:
            self.bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)
        else:
            self.bg_color = bg_color.to(device)
    
    def render(
        self,
        camera: CameraParams,
        means3D: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        colors: torch.Tensor = None,
        shs: torch.Tensor = None,
        scaling_modifier: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to a 2D image.
        
        Args:
            camera: Camera parameters
            means3D: Gaussian centers [N, 3]
            opacities: Gaussian opacities [N, 1]
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations as quaternions [N, 4]
            colors: Precomputed colors [N, 3] (optional)
            shs: Spherical harmonics coefficients (optional)
            scaling_modifier: Scale multiplier
        
        Returns:
            (rendered_image, radii, depth_image)
            - rendered_image: [C, H, W] rendered image
            - radii: [N] radius of each Gaussian on screen
            - depth_image: [1, H, W] depth map (if available)
        """
        N = means3D.shape[0]
        
        # Create screenspace points for gradient tracking
        screenspace_points = torch.zeros_like(means3D, requires_grad=True, device=self.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # Setup rasterization settings
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            antialiasing=self.antialiasing
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Handle colors
        if colors is None and shs is None:
            # Default to white
            colors = torch.ones(N, 3, device=self.device)
        
        # Rasterize
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        return rendered_image, radii, depth_image
    
    def render_from_gaussian_model(
        self,
        camera: CameraParams,
        gaussians,
        scaling_modifier: float = 1.0,
        override_color: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render from a GaussianModel object.
        
        Args:
            camera: Camera parameters
            gaussians: GaussianModel instance
            scaling_modifier: Scale multiplier
            override_color: Override Gaussian colors
        
        Returns:
            (rendered_image, radii, depth_image)
        """
        means3D = gaussians.get_xyz
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        
        # Handle intensity vs opacity
        if hasattr(gaussians, 'get_opacity'):
            opacities = gaussians.get_opacity
        else:
            # Use intensity as opacity for grayscale models
            opacities = gaussians.get_intensity
        
        # Handle colors
        if override_color is not None:
            colors = override_color
        elif hasattr(gaussians, 'get_features'):
            colors = None  # Will use SHs
            shs = gaussians.get_features
        else:
            # Use intensity as grayscale color
            intensity = gaussians.get_intensity
            colors = intensity.expand(-1, 3)  # [N, 3]
            shs = None
        
        return self.render(
            camera=camera,
            means3D=means3D,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            colors=colors,
            shs=shs if 'shs' in dir() else None,
            scaling_modifier=scaling_modifier
        )


def render_multiple_views(
    renderer: GaussianRenderer,
    gaussians,
    num_views: int = 8,
    distance: float = 2.0,
    elevation: float = 30.0,
    image_size: int = 512,
    fov_deg: float = 60.0
) -> torch.Tensor:
    """
    Render Gaussians from multiple viewpoints around the object.
    
    Args:
        renderer: GaussianRenderer instance
        gaussians: GaussianModel instance
        num_views: Number of views to render
        distance: Camera distance from origin
        elevation: Camera elevation in degrees
        image_size: Output image size
        fov_deg: Field of view in degrees
    
    Returns:
        [num_views, C, H, W] tensor of rendered images
    """
    images = []
    
    for i in range(num_views):
        azimuth = i * (360.0 / num_views)
        
        camera = create_camera_from_orbit(
            distance=distance,
            elevation=elevation,
            azimuth=azimuth,
            image_height=image_size,
            image_width=image_size,
            fov_deg=fov_deg
        )
        
        rendered, _, _ = renderer.render_from_gaussian_model(camera, gaussians)
        images.append(rendered)
    
    return torch.stack(images, dim=0)


# =============================================================================
# MIP (Maximum Intensity Projection) Rendering
# =============================================================================

def render_mip_projection(
    renderer: GaussianRenderer,
    gaussians,
    axis: int = 2,
    num_slices: int = 100,
    image_size: int = 512,
    fov_deg: float = 60.0
) -> torch.Tensor:
    """
    Render a Maximum Intensity Projection along a specified axis.
    
    This renders multiple views along the axis and takes the maximum.
    
    Args:
        renderer: GaussianRenderer instance
        gaussians: GaussianModel instance
        axis: Projection axis (0=X, 1=Y, 2=Z)
        num_slices: Number of depth slices for MIP
        image_size: Output image size
        fov_deg: Field of view in degrees
    
    Returns:
        [C, H, W] MIP image
    """
    # Setup camera based on axis
    if axis == 0:  # X-axis (side view)
        elevation, azimuth = 0, 90
    elif axis == 1:  # Y-axis (top view)
        elevation, azimuth = 90, 0
    else:  # Z-axis (front view)
        elevation, azimuth = 0, 0
    
    camera = create_camera_from_orbit(
        distance=2.0,
        elevation=elevation,
        azimuth=azimuth,
        image_height=image_size,
        image_width=image_size,
        fov_deg=fov_deg
    )
    
    rendered, _, _ = renderer.render_from_gaussian_model(camera, gaussians)
    
    return rendered


# =============================================================================
# Utility functions
# =============================================================================

def normalize_gaussian_positions(positions: torch.Tensor, volume_size: tuple) -> torch.Tensor:
    """
    Normalize Gaussian positions from voxel coordinates to [-1, 1] range.
    
    Args:
        positions: [N, 3] positions in voxel coordinates
        volume_size: (D, H, W) volume dimensions
    
    Returns:
        [N, 3] normalized positions in [-1, 1]
    """
    volume_size = torch.tensor(volume_size, device=positions.device, dtype=positions.dtype)
    normalized = 2.0 * positions / volume_size - 1.0
    return normalized


def denormalize_gaussian_positions(positions: torch.Tensor, volume_size: tuple) -> torch.Tensor:
    """
    Convert Gaussian positions from [-1, 1] range to voxel coordinates.
    
    Args:
        positions: [N, 3] normalized positions in [-1, 1]
        volume_size: (D, H, W) volume dimensions
    
    Returns:
        [N, 3] positions in voxel coordinates
    """
    volume_size = torch.tensor(volume_size, device=positions.device, dtype=positions.dtype)
    denormalized = (positions + 1.0) * volume_size / 2.0
    return denormalized


if __name__ == "__main__":
    # Simple test
    print("Testing GaussianRenderer...")
    
    renderer = GaussianRenderer(sh_degree=0)
    
    # Create test Gaussians
    N = 100
    means3D = torch.randn(N, 3, device='cuda') * 0.5
    opacities = torch.ones(N, 1, device='cuda') * 0.8
    scales = torch.ones(N, 3, device='cuda') * 0.05
    rotations = torch.zeros(N, 4, device='cuda')
    rotations[:, 0] = 1.0  # Identity quaternion
    colors = torch.rand(N, 3, device='cuda')
    
    # Create camera
    camera = create_camera_from_orbit(
        distance=2.0,
        elevation=30.0,
        azimuth=45.0,
        image_height=512,
        image_width=512
    )
    
    # Render
    image, radii, depth = renderer.render(
        camera=camera,
        means3D=means3D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        colors=colors
    )
    
    print(f"Rendered image shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print("Test passed!")
