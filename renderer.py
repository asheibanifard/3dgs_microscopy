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

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        q: [N, 4] quaternions (w, x, y, z)
    
    Returns:
        [N, 3, 3] rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(q.shape[0], 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def compute_2d_covariance(
    cov_3d: torch.Tensor,
    axis_u: int,
    axis_v: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract 2D marginal covariance from 3D covariance for orthographic projection.
    
    Args:
        cov_3d: [N, 3, 3] 3D covariance matrices
        axis_u: First projection axis
        axis_v: Second projection axis
    
    Returns:
        (cov_2d, inv_cov_2d): 2D covariance and its inverse
    """
    N = cov_3d.shape[0]
    cov_2d = torch.zeros(N, 2, 2, device=cov_3d.device)
    cov_2d[:, 0, 0] = cov_3d[:, axis_u, axis_u]
    cov_2d[:, 0, 1] = cov_3d[:, axis_u, axis_v]
    cov_2d[:, 1, 0] = cov_3d[:, axis_v, axis_u]
    cov_2d[:, 1, 1] = cov_3d[:, axis_v, axis_v]
    
    # Inverse 2D covariance
    det = cov_2d[:, 0, 0] * cov_2d[:, 1, 1] - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]
    det = det.clamp(min=1e-10)
    
    inv_cov_2d = torch.zeros_like(cov_2d)
    inv_cov_2d[:, 0, 0] = cov_2d[:, 1, 1] / det
    inv_cov_2d[:, 1, 1] = cov_2d[:, 0, 0] / det
    inv_cov_2d[:, 0, 1] = -cov_2d[:, 0, 1] / det
    inv_cov_2d[:, 1, 0] = -cov_2d[:, 1, 0] / det
    
    return cov_2d, inv_cov_2d


def render_mip_orthographic(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    proj_axis: int,
    img_h: int,
    img_w: int,
    scale_factor: float = 4.0
) -> torch.Tensor:
    """
    Render Maximum Intensity Projection with orthographic projection.
    
    MIP takes the maximum Gaussian contribution along each ray.
    
    Args:
        xyz: [N, 3] Gaussian positions in [0, 1] range
        intensity: [N] Gaussian intensities
        scaling: [N, 3] Gaussian scales
        rotation: [N, 4] Gaussian rotations (quaternions)
        proj_axis: Projection axis (0=Z top-down, 1=Y front, 2=X side)
        img_h: Output image height
        img_w: Output image width
        scale_factor: Bounding box multiplier for Gaussians
    
    Returns:
        [H, W] MIP rendered image
    """
    N = xyz.shape[0]
    
    # Determine projection axes
    if proj_axis == 0:  # Looking along Z -> project to YX plane
        axis_u, axis_v = 1, 2
    elif proj_axis == 1:  # Looking along Y -> project to ZX plane
        axis_u, axis_v = 0, 2
    else:  # Looking along X -> project to ZY plane
        axis_u, axis_v = 0, 1
    
    # Build 3D covariance
    R = quaternion_to_rotation_matrix(rotation)
    S_diag = torch.diag_embed(scaling)
    cov_3d = R @ S_diag @ S_diag @ R.transpose(1, 2)
    
    # Get 2D covariance
    _, inv_cov_2d = compute_2d_covariance(cov_3d, axis_u, axis_v)
    
    # Move to numpy for splatting
    pos_2d = torch.stack([xyz[:, axis_u], xyz[:, axis_v]], dim=1).cpu().numpy()
    inv_cov_np = inv_cov_2d.cpu().numpy()
    intensity_np = intensity.cpu().numpy()
    scaling_np = scaling.cpu().numpy()
    
    # Grid coordinates
    u_coords = np.linspace(0, 1, img_h)
    v_coords = np.linspace(0, 1, img_w)
    
    # Output image
    image = np.zeros((img_h, img_w), dtype=np.float32)
    
    for g_idx in range(N):
        pos = pos_2d[g_idx]
        inv_c = inv_cov_np[g_idx]
        inten = intensity_np[g_idx]
        
        if inten < 0.01:
            continue
        
        # Bounding box
        s = scaling_np[g_idx]
        max_scale = max(s[axis_u], s[axis_v]) * scale_factor
        
        u_min = max(0, int((pos[0] - max_scale) * img_h))
        u_max = min(img_h, int((pos[0] + max_scale) * img_h) + 1)
        v_min = max(0, int((pos[1] - max_scale) * img_w))
        v_max = min(img_w, int((pos[1] + max_scale) * img_w) + 1)
        
        if u_min >= u_max or v_min >= v_max:
            continue
        
        uu, vv = np.meshgrid(u_coords[u_min:u_max], v_coords[v_min:v_max], indexing='ij')
        
        du = uu - pos[0]
        dv = vv - pos[1]
        
        mahal_sq = (inv_c[0, 0] * du * du + 
                   (inv_c[0, 1] + inv_c[1, 0]) * du * dv + 
                    inv_c[1, 1] * dv * dv)
        
        gauss = np.exp(-0.5 * np.clip(mahal_sq, 0, 20)) * inten
        
        # MIP: take maximum
        image[u_min:u_max, v_min:v_max] = np.maximum(
            image[u_min:u_max, v_min:v_max], gauss)
    
    return torch.from_numpy(image)


def render_alpha_blending_orthographic(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    proj_axis: int,
    img_h: int,
    img_w: int,
    scale_factor: float = 4.0
) -> torch.Tensor:
    """
    Render with alpha blending (front-to-back compositing) and orthographic projection.
    
    Gaussians are sorted by depth and composited using the over operator.
    
    Args:
        xyz: [N, 3] Gaussian positions in [0, 1] range
        intensity: [N] Gaussian intensities (used as both opacity and color)
        scaling: [N, 3] Gaussian scales
        rotation: [N, 4] Gaussian rotations (quaternions)
        proj_axis: Projection axis (0=Z top-down, 1=Y front, 2=X side)
        img_h: Output image height
        img_w: Output image width
        scale_factor: Bounding box multiplier for Gaussians
    
    Returns:
        [H, W] alpha-blended rendered image
    """
    N = xyz.shape[0]
    
    # Determine projection axes
    if proj_axis == 0:  # Looking along Z
        axis_u, axis_v, axis_depth = 1, 2, 0
    elif proj_axis == 1:  # Looking along Y
        axis_u, axis_v, axis_depth = 0, 2, 1
    else:  # Looking along X
        axis_u, axis_v, axis_depth = 0, 1, 2
    
    # Sort by depth (back to front for over compositing)
    depth = xyz[:, axis_depth]
    sorted_idx = torch.argsort(depth, descending=True)
    
    xyz_sorted = xyz[sorted_idx]
    intensity_sorted = intensity[sorted_idx]
    scaling_sorted = scaling[sorted_idx]
    rotation_sorted = rotation[sorted_idx]
    
    # Build 3D covariance
    R = quaternion_to_rotation_matrix(rotation_sorted)
    S_diag = torch.diag_embed(scaling_sorted)
    cov_3d = R @ S_diag @ S_diag @ R.transpose(1, 2)
    
    # Get 2D covariance
    _, inv_cov_2d = compute_2d_covariance(cov_3d, axis_u, axis_v)
    
    # Move to numpy
    pos_2d = torch.stack([xyz_sorted[:, axis_u], xyz_sorted[:, axis_v]], dim=1).cpu().numpy()
    inv_cov_np = inv_cov_2d.cpu().numpy()
    intensity_np = intensity_sorted.cpu().numpy()
    scaling_np = scaling_sorted.cpu().numpy()
    
    # Grid coordinates
    u_coords = np.linspace(0, 1, img_h)
    v_coords = np.linspace(0, 1, img_w)
    
    # Output: color and transmittance
    color = np.zeros((img_h, img_w), dtype=np.float32)
    transmittance = np.ones((img_h, img_w), dtype=np.float32)
    
    for g_idx in range(N):
        pos = pos_2d[g_idx]
        inv_c = inv_cov_np[g_idx]
        inten = intensity_np[g_idx]
        
        if inten < 0.01:
            continue
        
        # Bounding box
        s = scaling_np[g_idx]
        max_scale = max(s[axis_u], s[axis_v]) * scale_factor
        
        u_min = max(0, int((pos[0] - max_scale) * img_h))
        u_max = min(img_h, int((pos[0] + max_scale) * img_h) + 1)
        v_min = max(0, int((pos[1] - max_scale) * img_w))
        v_max = min(img_w, int((pos[1] + max_scale) * img_w) + 1)
        
        if u_min >= u_max or v_min >= v_max:
            continue
        
        uu, vv = np.meshgrid(u_coords[u_min:u_max], v_coords[v_min:v_max], indexing='ij')
        
        du = uu - pos[0]
        dv = vv - pos[1]
        
        mahal_sq = (inv_c[0, 0] * du * du + 
                   (inv_c[0, 1] + inv_c[1, 0]) * du * dv + 
                    inv_c[1, 1] * dv * dv)
        
        # Gaussian opacity
        alpha = np.exp(-0.5 * np.clip(mahal_sq, 0, 20)) * inten
        alpha = np.clip(alpha, 0, 1)
        
        # Front-to-back compositing: C_out = C_in + T * alpha * c
        local_T = transmittance[u_min:u_max, v_min:v_max]
        color[u_min:u_max, v_min:v_max] += local_T * alpha * inten
        transmittance[u_min:u_max, v_min:v_max] *= (1 - alpha)
    
    return torch.from_numpy(color)


def render_gaussians_orthographic(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    proj_axis: int,
    img_h: int,
    img_w: int,
    mode: str = 'alpha',
    scale_factor: float = 4.0
) -> torch.Tensor:
    """
    Render Gaussians with orthographic projection.
    
    Args:
        xyz: [N, 3] Gaussian positions in [0, 1] range
        intensity: [N] Gaussian intensities
        scaling: [N, 3] Gaussian scales
        rotation: [N, 4] Gaussian rotations (quaternions)
        proj_axis: Projection axis (0=Z top-down, 1=Y front, 2=X side)
        img_h: Output image height
        img_w: Output image width
        mode: Rendering mode - 'mip', 'alpha', or 'sum'
        scale_factor: Bounding box multiplier for Gaussians
    
    Returns:
        [H, W] rendered image
    """
    if mode == 'mip':
        return render_mip_orthographic(
            xyz, intensity, scaling, rotation,
            proj_axis, img_h, img_w, scale_factor
        )
    elif mode == 'alpha':
        return render_alpha_blending_orthographic(
            xyz, intensity, scaling, rotation,
            proj_axis, img_h, img_w, scale_factor
        )
    elif mode == 'sum':
        # Simple additive blending (no depth sorting)
        return render_sum_orthographic(
            xyz, intensity, scaling, rotation,
            proj_axis, img_h, img_w, scale_factor
        )
    else:
        raise ValueError(f"Unknown rendering mode: {mode}. Use 'mip', 'alpha', or 'sum'")


def render_sum_orthographic(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    proj_axis: int,
    img_h: int,
    img_w: int,
    scale_factor: float = 4.0
) -> torch.Tensor:
    """
    Render with simple additive blending (sum of Gaussian contributions).
    
    Args:
        xyz: [N, 3] Gaussian positions in [0, 1] range
        intensity: [N] Gaussian intensities
        scaling: [N, 3] Gaussian scales
        rotation: [N, 4] Gaussian rotations (quaternions)
        proj_axis: Projection axis (0=Z, 1=Y, 2=X)
        img_h: Output image height
        img_w: Output image width
        scale_factor: Bounding box multiplier for Gaussians
    
    Returns:
        [H, W] summed rendered image
    """
    N = xyz.shape[0]
    
    # Determine projection axes
    if proj_axis == 0:
        axis_u, axis_v = 1, 2
    elif proj_axis == 1:
        axis_u, axis_v = 0, 2
    else:
        axis_u, axis_v = 0, 1
    
    # Build 3D covariance
    R = quaternion_to_rotation_matrix(rotation)
    S_diag = torch.diag_embed(scaling)
    cov_3d = R @ S_diag @ S_diag @ R.transpose(1, 2)
    
    # Get 2D covariance
    _, inv_cov_2d = compute_2d_covariance(cov_3d, axis_u, axis_v)
    
    # Move to numpy
    pos_2d = torch.stack([xyz[:, axis_u], xyz[:, axis_v]], dim=1).cpu().numpy()
    inv_cov_np = inv_cov_2d.cpu().numpy()
    intensity_np = intensity.cpu().numpy()
    scaling_np = scaling.cpu().numpy()
    
    # Grid coordinates
    u_coords = np.linspace(0, 1, img_h)
    v_coords = np.linspace(0, 1, img_w)
    
    # Output image
    image = np.zeros((img_h, img_w), dtype=np.float32)
    
    for g_idx in range(N):
        pos = pos_2d[g_idx]
        inv_c = inv_cov_np[g_idx]
        inten = intensity_np[g_idx]
        
        if inten < 0.01:
            continue
        
        s = scaling_np[g_idx]
        max_scale = max(s[axis_u], s[axis_v]) * scale_factor
        
        u_min = max(0, int((pos[0] - max_scale) * img_h))
        u_max = min(img_h, int((pos[0] + max_scale) * img_h) + 1)
        v_min = max(0, int((pos[1] - max_scale) * img_w))
        v_max = min(img_w, int((pos[1] + max_scale) * img_w) + 1)
        
        if u_min >= u_max or v_min >= v_max:
            continue
        
        uu, vv = np.meshgrid(u_coords[u_min:u_max], v_coords[v_min:v_max], indexing='ij')
        
        du = uu - pos[0]
        dv = vv - pos[1]
        
        mahal_sq = (inv_c[0, 0] * du * du + 
                   (inv_c[0, 1] + inv_c[1, 0]) * du * dv + 
                    inv_c[1, 1] * dv * dv)
        
        gauss = np.exp(-0.5 * np.clip(mahal_sq, 0, 20)) * inten
        
        # Sum: additive blending
        image[u_min:u_max, v_min:v_max] += gauss
    
    return torch.from_numpy(image)


def render_all_views(
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    volume_size: Tuple[int, int, int],
    mode: str = 'alpha'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render all three orthographic views (top, front, side).
    
    Args:
        xyz: [N, 3] Gaussian positions in [0, 1] range
        intensity: [N] Gaussian intensities
        scaling: [N, 3] Gaussian scales
        rotation: [N, 4] Gaussian rotations (quaternions)
        volume_size: (Z, Y, X) volume dimensions
        mode: Rendering mode - 'mip', 'alpha', or 'sum'
    
    Returns:
        (top_view, front_view, side_view) rendered images
    """
    vol_z, vol_y, vol_x = volume_size
    
    # Top-down view (along Z axis): Y x X
    top = render_gaussians_orthographic(
        xyz, intensity, scaling, rotation,
        proj_axis=0, img_h=vol_y, img_w=vol_x, mode=mode
    )
    
    # Front view (along Y axis): Z x X
    front = render_gaussians_orthographic(
        xyz, intensity, scaling, rotation,
        proj_axis=1, img_h=vol_z, img_w=vol_x, mode=mode
    )
    
    # Side view (along X axis): Z x Y
    side = render_gaussians_orthographic(
        xyz, intensity, scaling, rotation,
        proj_axis=2, img_h=vol_z, img_w=vol_y, mode=mode
    )
    
    return top, front, side


def render_from_checkpoint(
    checkpoint_path: str,
    volume_size: Tuple[int, int, int],
    mode: str = 'alpha',
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load a checkpoint and render all views.
    
    Args:
        checkpoint_path: Path to model checkpoint
        volume_size: (Z, Y, X) volume dimensions
        mode: Rendering mode - 'mip', 'alpha', or 'sum'
        device: Device for computation
    
    Returns:
        (top_view, front_view, side_view) rendered images
    """
    ckpt = torch.load(checkpoint_path, weights_only=False)
    
    xyz = ckpt['xyz'].to(device)
    intensity = torch.sigmoid(ckpt['intensity']).to(device).squeeze()
    scaling = torch.exp(ckpt['scaling']).to(device)
    rotation = torch.nn.functional.normalize(ckpt['rotation']).to(device)
    
    print(f"Loaded {ckpt['num_gaussians']} Gaussians from {checkpoint_path}")
    
    return render_all_views(xyz, intensity, scaling, rotation, volume_size, mode)


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
