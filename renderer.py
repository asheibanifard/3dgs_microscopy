"""
Gaussian Rendering Utilities (Rewritten, Corrected, and Unified)

This module provides:
1) Perspective splatting via diff_gaussian_rasterization (3DGS-style alpha compositing).
2) Orthographic rendering for volumetric-style projections with:
   - Field MIP (depth-binned approximation to voxel/field MIP)
   - Primitive MIP (max over projected Gaussians; mainly for debugging)
   - Alpha compositing (with separate opacity control; no intensity-squaring)
   - Simple additive sum

Key fixes vs your original:
- Correct orthographic axis mapping (XYZ -> projection plane + depth).
- Correct alpha blending model (no "intensity squared" artifacts).
- Adds a depth-binned Field MIP that much better matches voxel/field MIP than max-over-Gaussians.
- Removes mixed conventions in view definitions and clarifies units.

Assumptions:
- xyz is in [0, 1] range in each axis for orthographic modes.
- scaling is per-axis standard deviation (sigma) in the same coordinate system as xyz.
- rotation is quaternion [w, x, y, z], normalized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch

# Optional dependency (keep perspective splatting functionality)
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    _HAS_DIFF_GAUSSIAN = True
except Exception:
    _HAS_DIFF_GAUSSIAN = False


# =============================================================================
# Camera utilities (Perspective / 3DGS-style)
# =============================================================================

@dataclass
class CameraParams:
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor  # 4x4 (transposed for rasterizer)
    full_proj_transform: torch.Tensor   # 4x4 (transposed for rasterizer)
    camera_center: torch.Tensor         # [3]


def create_camera_intrinsics(
    image_height: int,
    image_width: int,
    fov_deg: float = 60.0,
) -> Tuple[float, float]:
    fov_rad = math.radians(fov_deg)
    aspect = float(image_width) / float(image_height)
    FoVy = fov_rad
    FoVx = 2.0 * math.atan(math.tan(FoVy / 2.0) * aspect)
    return FoVx, FoVy


def create_view_matrix(
    camera_position: torch.Tensor,
    look_at: torch.Tensor,
    up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Right-handed look-at view matrix, mapping world -> camera.
    Camera looks along -Z in camera space.
    """
    device = camera_position.device
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=camera_position.dtype)

    cam = camera_position
    tgt = look_at

    forward = (tgt - cam)
    forward = forward / (forward.norm() + 1e-12)

    # right = forward x up
    right = torch.cross(forward, up, dim=0)
    right = right / (right.norm() + 1e-12)

    # recompute orthonormal up
    up_ortho = torch.cross(right, forward, dim=0)

    # World->camera rotation rows
    R = torch.stack([right, up_ortho, -forward], dim=0)  # [3,3]
    t = -R @ cam

    view = torch.eye(4, device=device, dtype=camera_position.dtype)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def create_projection_matrix(
    FoVx: float,
    FoVy: float,
    near: float = 0.01,
    far: float = 100.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Standard perspective projection (OpenGL-like, right-handed camera looking -Z).
    """
    tanHalfFovY = math.tan(FoVy / 2.0)
    tanHalfFovX = math.tan(FoVx / 2.0)

    top = tanHalfFovY * near
    bottom = -top
    right = tanHalfFovX * near
    left = -right

    P = torch.zeros(4, 4, device=device, dtype=dtype)
    P[0, 0] = 2.0 * near / (right - left)
    P[1, 1] = 2.0 * near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def create_camera_from_orbit(
    distance: float,
    elevation_deg: float,
    azimuth_deg: float,
    look_at: Optional[torch.Tensor] = None,
    image_height: int = 512,
    image_width: int = 512,
    fov_deg: float = 60.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    near: float = 0.01,
    far: float = 100.0,
) -> CameraParams:
    if look_at is None:
        look_at = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)

    x = distance * math.cos(elev) * math.sin(azim)
    y = distance * math.sin(elev)
    z = distance * math.cos(elev) * math.cos(azim)
    camera_position = torch.tensor([x, y, z], device=device, dtype=dtype) + look_at

    FoVx, FoVy = create_camera_intrinsics(image_height, image_width, fov_deg)
    view = create_view_matrix(camera_position, look_at)
    proj = create_projection_matrix(FoVx, FoVy, near=near, far=far, device=device, dtype=dtype)

    full_proj = proj @ view  # world -> clip
    # Rasterizer in 3DGS expects transposed matrices.
    return CameraParams(
        image_height=image_height,
        image_width=image_width,
        FoVx=FoVx,
        FoVy=FoVy,
        world_view_transform=view.T.contiguous(),
        full_proj_transform=full_proj.T.contiguous(),
        camera_center=camera_position,
    )


# =============================================================================
# Perspective splatting renderer (diff_gaussian_rasterization)
# =============================================================================

class GaussianRenderer:
    """
    Perspective renderer for 3D Gaussians using diff_gaussian_rasterization.
    This uses alpha blending (3DGS-style). It is NOT MIP.
    """

    def __init__(
        self,
        sh_degree: int = 0,
        bg_color: Optional[torch.Tensor] = None,
        device: str = "cuda",
        antialiasing: bool = True,
        debug: bool = False,
    ):
        if not _HAS_DIFF_GAUSSIAN:
            raise ImportError("diff_gaussian_rasterization is not available in this environment.")

        self.sh_degree = sh_degree
        self.device = device
        self.antialiasing = antialiasing
        self.debug = debug
        self.bg_color = (torch.tensor([0.0, 0.0, 0.0], device=device)
                         if bg_color is None else bg_color.to(device))

    @torch.no_grad()
    def _validate_shapes(
        self,
        means3D: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
    ) -> None:
        assert means3D.ndim == 2 and means3D.shape[1] == 3
        assert scales.ndim == 2 and scales.shape[1] == 3
        assert rotations.ndim == 2 and rotations.shape[1] == 4
        assert opacities.ndim == 2 and opacities.shape[1] == 1

    def render(
        self,
        camera: CameraParams,
        means3D: torch.Tensor,      # [N,3]
        opacities: torch.Tensor,    # [N,1]
        scales: torch.Tensor,       # [N,3]
        rotations: torch.Tensor,    # [N,4] quaternion
        colors: Optional[torch.Tensor] = None,  # [N,3]
        shs: Optional[torch.Tensor] = None,
        scaling_modifier: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._validate_shapes(means3D, opacities, scales, rotations)

        N = means3D.shape[0]
        screenspace_points = torch.zeros_like(means3D, requires_grad=True, device=self.device)
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

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
            antialiasing=self.antialiasing,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if colors is None and shs is None:
            colors = torch.ones(N, 3, device=self.device)

        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return rendered_image, radii, depth_image

    def render_from_gaussian_model(
        self,
        camera: CameraParams,
        gaussians,
        scaling_modifier: float = 1.0,
        override_color: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means3D = gaussians.get_xyz
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        if hasattr(gaussians, "get_opacity"):
            opacities = gaussians.get_opacity
        else:
            opacities = gaussians.get_intensity

        if override_color is not None:
            colors = override_color
            shs = None
        elif hasattr(gaussians, "get_features"):
            colors = None
            shs = gaussians.get_features
        else:
            inten = gaussians.get_intensity
            colors = inten.expand(-1, 3)
            shs = None

        return self.render(
            camera=camera,
            means3D=means3D,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            colors=colors,
            shs=shs,
            scaling_modifier=scaling_modifier,
        )


# =============================================================================
# Orthographic volumetric-style rendering
# =============================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    q: [N,4] quaternion in [w, x, y, z]
    returns: [N,3,3]
    """
    assert q.ndim == 2 and q.shape[1] == 4
    q = torch.nn.functional.normalize(q, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_cov3d_from_scale_rotation(
    scaling: torch.Tensor,   # [N,3] sigma per axis
    rotation: torch.Tensor,  # [N,4] quaternion
) -> torch.Tensor:
    """
    cov3d = R * diag(sigma^2) * R^T
    """
    assert scaling.ndim == 2 and scaling.shape[1] == 3
    R = quaternion_to_rotation_matrix(rotation)  # [N,3,3]
    sigma2 = scaling * scaling
    S2 = torch.diag_embed(sigma2)  # [N,3,3]
    cov3d = R @ S2 @ R.transpose(1, 2)
    return cov3d


def _axes_for_ortho(
    view_axis: Literal["x", "y", "z"]
) -> Tuple[int, int, int]:
    """
    Returns (axis_u, axis_v, axis_depth) given xyz indexing [0:x,1:y,2:z]

    Look along:
      - z: project to (x,y), depth=z
      - y: project to (x,z), depth=y
      - x: project to (y,z), depth=x
    """
    if view_axis == "z":
        return 0, 1, 2
    if view_axis == "y":
        return 0, 2, 1
    if view_axis == "x":
        return 1, 2, 0
    raise ValueError("view_axis must be one of {'x','y','z'}")


def _extract_cov2d_inv(
    cov3d: torch.Tensor, axis_u: int, axis_v: int
) -> torch.Tensor:
    """
    Extract 2x2 marginal covariance for orthographic projection and return inverse.
    cov2d = [[cov[u,u], cov[u,v]],
             [cov[v,u], cov[v,v]]]
    """
    N = cov3d.shape[0]
    cov2d = torch.empty((N, 2, 2), device=cov3d.device, dtype=cov3d.dtype)
    cov2d[:, 0, 0] = cov3d[:, axis_u, axis_u]
    cov2d[:, 0, 1] = cov3d[:, axis_u, axis_v]
    cov2d[:, 1, 0] = cov3d[:, axis_v, axis_u]
    cov2d[:, 1, 1] = cov3d[:, axis_v, axis_v]

    # Invert 2x2
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = det.clamp_min(1e-12)

    inv = torch.empty_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1] / det
    inv[:, 1, 1] = cov2d[:, 0, 0] / det
    inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv[:, 1, 0] = -cov2d[:, 1, 0] / det
    return inv


def _bbox_from_sigma(
    pos_uv_01: torch.Tensor,       # [2] in [0,1]
    sigma_uv_01: torch.Tensor,     # [2] sigma in [0,1] units (approx)
    H: int,
    W: int,
    sigma_extent: float = 3.0,     # render to 3-sigma by default
) -> Tuple[int, int, int, int]:
    """
    Convert a UV-space sigma extent into pixel bounding box.
    """
    u, v = float(pos_uv_01[0]), float(pos_uv_01[1])
    su, sv = float(sigma_uv_01[0]), float(sigma_uv_01[1])
    du = sigma_extent * su
    dv = sigma_extent * sv

    u0 = max(0, int((u - du) * (H - 1)))
    u1 = min(H, int((u + du) * (H - 1)) + 1)
    v0 = max(0, int((v - dv) * (W - 1)))
    v1 = min(W, int((v + dv) * (W - 1)) + 1)
    return u0, u1, v0, v1


def render_orthographic_sum(
    xyz_01: torch.Tensor,          # [N,3] in [0,1]
    intensity: torch.Tensor,       # [N] in [0,1] (or any positive)
    scaling: torch.Tensor,         # [N,3] sigma in [0,1] units
    rotation: torch.Tensor,        # [N,4]
    view_axis: Literal["x", "y", "z"],
    H: int,
    W: int,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> torch.Tensor:
    """
    Additive sum: image += intensity_i * exp(-0.5 * maha^2)
    """
    device = xyz_01.device
    dtype = xyz_01.dtype

    axis_u, axis_v, _ = _axes_for_ortho(view_axis)
    cov3d = build_cov3d_from_scale_rotation(scaling, rotation)
    inv2d = _extract_cov2d_inv(cov3d, axis_u, axis_v)  # [N,2,2]

    img = torch.zeros((H, W), device=device, dtype=torch.float32)

    # Precompute grid coords in [0,1]
    u_coords = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
    v_coords = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)

    intensity = intensity.to(torch.float32).view(-1)
    xyz = xyz_01.to(torch.float32)

    for i in range(xyz.shape[0]):
        inten = float(intensity[i].item())
        if inten <= intensity_min:
            continue

        pos_uv = torch.tensor([xyz[i, axis_u], xyz[i, axis_v]], device=device, dtype=torch.float32)
        # approximate sigma for bbox using diagonal sigmas in projection axes
        sigma_uv = torch.tensor([scaling[i, axis_u], scaling[i, axis_v]], device=device, dtype=torch.float32)

        u0, u1, v0, v1 = _bbox_from_sigma(pos_uv, sigma_uv, H, W, sigma_extent=sigma_extent)
        if u0 >= u1 or v0 >= v1:
            continue

        uu = u_coords[u0:u1].unsqueeze(1)  # [hu,1]
        vv = v_coords[v0:v1].unsqueeze(0)  # [1,wv]
        du = uu - pos_uv[0]
        dv = vv - pos_uv[1]

        inv = inv2d[i].to(torch.float32)
        # maha^2 = [du dv] * inv * [du dv]^T
        maha2 = (
            inv[0, 0] * du * du
            + (inv[0, 1] + inv[1, 0]) * du * dv
            + inv[1, 1] * dv * dv
        ).clamp_(0.0, 50.0)

        w = torch.exp(-0.5 * maha2) * inten
        img[u0:u1, v0:v1] += w

    return img.to(dtype)


def render_orthographic_primitive_mip(
    xyz_01: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    view_axis: Literal["x", "y", "z"],
    H: int,
    W: int,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> torch.Tensor:
    """
    Primitive MIP (debug/visualization):
      image = max_i intensity_i * exp(-0.5 * maha^2)

    This is NOT equivalent to voxel/field MIP and will show "beads/balls" when primitives are exposed.
    """
    device = xyz_01.device
    dtype = xyz_01.dtype

    axis_u, axis_v, _ = _axes_for_ortho(view_axis)
    cov3d = build_cov3d_from_scale_rotation(scaling, rotation)
    inv2d = _extract_cov2d_inv(cov3d, axis_u, axis_v)

    img = torch.zeros((H, W), device=device, dtype=torch.float32)

    u_coords = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
    v_coords = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)

    intensity = intensity.to(torch.float32).view(-1)
    xyz = xyz_01.to(torch.float32)

    for i in range(xyz.shape[0]):
        inten = float(intensity[i].item())
        if inten <= intensity_min:
            continue

        pos_uv = torch.tensor([xyz[i, axis_u], xyz[i, axis_v]], device=device, dtype=torch.float32)
        sigma_uv = torch.tensor([scaling[i, axis_u], scaling[i, axis_v]], device=device, dtype=torch.float32)

        u0, u1, v0, v1 = _bbox_from_sigma(pos_uv, sigma_uv, H, W, sigma_extent=sigma_extent)
        if u0 >= u1 or v0 >= v1:
            continue

        uu = u_coords[u0:u1].unsqueeze(1)
        vv = v_coords[v0:v1].unsqueeze(0)
        du = uu - pos_uv[0]
        dv = vv - pos_uv[1]

        inv = inv2d[i].to(torch.float32)
        maha2 = (
            inv[0, 0] * du * du
            + (inv[0, 1] + inv[1, 0]) * du * dv
            + inv[1, 1] * dv * dv
        ).clamp_(0.0, 50.0)

        w = torch.exp(-0.5 * maha2) * inten
        img[u0:u1, v0:v1] = torch.maximum(img[u0:u1, v0:v1], w)

    return img.to(dtype)


def render_orthographic_alpha(
    xyz_01: torch.Tensor,
    intensity: torch.Tensor,        # treated as emitted grayscale "color"
    opacity: torch.Tensor,          # [N] separate opacity control (recommended)
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    view_axis: Literal["x", "y", "z"],
    H: int,
    W: int,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> torch.Tensor:
    """
    Orthographic alpha compositing (front-to-back).

    color += T * alpha * intensity
    T *= (1 - alpha)

    alpha is derived from (opacity_i * exp(-0.5*maha^2)), then clamped to [0,1].
    """
    device = xyz_01.device
    dtype = xyz_01.dtype

    axis_u, axis_v, axis_d = _axes_for_ortho(view_axis)
    cov3d = build_cov3d_from_scale_rotation(scaling, rotation)
    inv2d = _extract_cov2d_inv(cov3d, axis_u, axis_v)

    xyz = xyz_01.to(torch.float32)
    intensity = intensity.to(torch.float32).view(-1)
    opacity = opacity.to(torch.float32).view(-1)

    # sort front-to-back: smaller depth first (closer) for standard front-to-back compositing
    depth = xyz[:, axis_d]
    order = torch.argsort(depth, descending=False)

    img = torch.zeros((H, W), device=device, dtype=torch.float32)
    T = torch.ones((H, W), device=device, dtype=torch.float32)

    u_coords = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
    v_coords = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)

    for idx in order.tolist():
        inten = float(intensity[idx].item())
        if inten <= intensity_min:
            continue

        op = float(opacity[idx].item())
        if op <= 0.0:
            continue

        pos_uv = torch.tensor([xyz[idx, axis_u], xyz[idx, axis_v]], device=device, dtype=torch.float32)
        sigma_uv = torch.tensor([scaling[idx, axis_u], scaling[idx, axis_v]], device=device, dtype=torch.float32)

        u0, u1, v0, v1 = _bbox_from_sigma(pos_uv, sigma_uv, H, W, sigma_extent=sigma_extent)
        if u0 >= u1 or v0 >= v1:
            continue

        uu = u_coords[u0:u1].unsqueeze(1)
        vv = v_coords[v0:v1].unsqueeze(0)
        du = uu - pos_uv[0]
        dv = vv - pos_uv[1]

        inv = inv2d[idx].to(torch.float32)
        maha2 = (
            inv[0, 0] * du * du
            + (inv[0, 1] + inv[1, 0]) * du * dv
            + inv[1, 1] * dv * dv
        ).clamp_(0.0, 50.0)

        # local alpha from opacity * gaussian footprint
        a = torch.exp(-0.5 * maha2) * op
        a = a.clamp_(0.0, 1.0)

        localT = T[u0:u1, v0:v1]
        img[u0:u1, v0:v1] += localT * a * inten
        T[u0:u1, v0:v1] = localT * (1.0 - a)

    return img.to(dtype)


def render_orthographic_field_mip(
    xyz_01: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    view_axis: Literal["x", "y", "z"],
    H: int,
    W: int,
    num_depth_bins: int = 64,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> torch.Tensor:
    """
    Field MIP approximation (recommended for matching voxel/GT MIP):

    1) Bin Gaussians by depth into B bins.
    2) For each bin, accumulate the *field* in that layer: layer += intensity_i * footprint_i
    3) MIP across depth bins: image = max_b layer[b]

    This approximates: I(u,v) = max_t sum_i intensity_i * G_i( r(t) )
    """
    device = xyz_01.device
    dtype = xyz_01.dtype

    axis_u, axis_v, axis_d = _axes_for_ortho(view_axis)

    xyz = xyz_01.to(torch.float32)
    intensity = intensity.to(torch.float32).view(-1)

    cov3d = build_cov3d_from_scale_rotation(scaling, rotation)
    inv2d = _extract_cov2d_inv(cov3d, axis_u, axis_v).to(torch.float32)

    # Precompute depth bins in [0,1]
    depth = xyz[:, axis_d].clamp(0.0, 1.0)
    bin_idx = torch.clamp((depth * num_depth_bins).long(), 0, num_depth_bins - 1)

    # Layers
    layers = torch.zeros((num_depth_bins, H, W), device=device, dtype=torch.float32)

    u_coords = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)
    v_coords = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)

    for i in range(xyz.shape[0]):
        inten = float(intensity[i].item())
        if inten <= intensity_min:
            continue

        b = int(bin_idx[i].item())
        pos_uv = torch.tensor([xyz[i, axis_u], xyz[i, axis_v]], device=device, dtype=torch.float32)
        sigma_uv = torch.tensor([scaling[i, axis_u], scaling[i, axis_v]], device=device, dtype=torch.float32)

        u0, u1, v0, v1 = _bbox_from_sigma(pos_uv, sigma_uv, H, W, sigma_extent=sigma_extent)
        if u0 >= u1 or v0 >= v1:
            continue

        uu = u_coords[u0:u1].unsqueeze(1)
        vv = v_coords[v0:v1].unsqueeze(0)
        du = uu - pos_uv[0]
        dv = vv - pos_uv[1]

        inv = inv2d[i]
        maha2 = (
            inv[0, 0] * du * du
            + (inv[0, 1] + inv[1, 0]) * du * dv
            + inv[1, 1] * dv * dv
        ).clamp_(0.0, 50.0)

        w = torch.exp(-0.5 * maha2) * inten
        layers[b, u0:u1, v0:v1] += w

    img = torch.amax(layers, dim=0)
    return img.to(dtype)


# =============================================================================
# Convenience wrappers
# =============================================================================

OrthoMode = Literal["field_mip", "primitive_mip", "alpha", "sum"]


def render_gaussians_orthographic(
    xyz_01: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    view_axis: Literal["x", "y", "z"],
    H: int,
    W: int,
    mode: OrthoMode = "field_mip",
    *,
    opacity: Optional[torch.Tensor] = None,
    num_depth_bins: int = 64,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> torch.Tensor:
    if mode == "sum":
        return render_orthographic_sum(
            xyz_01, intensity, scaling, rotation,
            view_axis=view_axis, H=H, W=W,
            sigma_extent=sigma_extent, intensity_min=intensity_min,
        )

    if mode == "primitive_mip":
        return render_orthographic_primitive_mip(
            xyz_01, intensity, scaling, rotation,
            view_axis=view_axis, H=H, W=W,
            sigma_extent=sigma_extent, intensity_min=intensity_min,
        )

    if mode == "field_mip":
        return render_orthographic_field_mip(
            xyz_01, intensity, scaling, rotation,
            view_axis=view_axis, H=H, W=W,
            num_depth_bins=num_depth_bins,
            sigma_extent=sigma_extent, intensity_min=intensity_min,
        )

    if mode == "alpha":
        if opacity is None:
            # Default: derive opacity from intensity (linear), but keep it separate.
            # You should tune opacity_scale for your data.
            opacity = (intensity.detach().clone()).clamp(0.0, 1.0)
        return render_orthographic_alpha(
            xyz_01, intensity, opacity, scaling, rotation,
            view_axis=view_axis, H=H, W=W,
            sigma_extent=sigma_extent, intensity_min=intensity_min,
        )

    raise ValueError(f"Unknown mode: {mode}")


def render_all_ortho_views(
    xyz_01: torch.Tensor,
    intensity: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    volume_size_zyx: Tuple[int, int, int],
    mode: OrthoMode = "field_mip",
    *,
    opacity: Optional[torch.Tensor] = None,
    num_depth_bins: int = 64,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    volume_size_zyx: (Z, Y, X) used to size images consistently with your volume indexing.

    Outputs:
      - view_axis='z' -> image (Y,X)
      - view_axis='y' -> image (Z,X)
      - view_axis='x' -> image (Z,Y)
    """
    Z, Y, X = volume_size_zyx

    top = render_gaussians_orthographic(
        xyz_01, intensity, scaling, rotation,
        view_axis="z", H=Y, W=X, mode=mode,
        opacity=opacity, num_depth_bins=num_depth_bins,
        sigma_extent=sigma_extent, intensity_min=intensity_min,
    )
    front = render_gaussians_orthographic(
        xyz_01, intensity, scaling, rotation,
        view_axis="y", H=Z, W=X, mode=mode,
        opacity=opacity, num_depth_bins=num_depth_bins,
        sigma_extent=sigma_extent, intensity_min=intensity_min,
    )
    side = render_gaussians_orthographic(
        xyz_01, intensity, scaling, rotation,
        view_axis="x", H=Z, W=Y, mode=mode,
        opacity=opacity, num_depth_bins=num_depth_bins,
        sigma_extent=sigma_extent, intensity_min=intensity_min,
    )
    return top, front, side


# =============================================================================
# Checkpoint loader (your format)
# =============================================================================

def render_from_checkpoint(
    checkpoint_path: str,
    volume_size_zyx: Tuple[int, int, int],
    mode: OrthoMode = "field_mip",
    device: str = "cuda",
    *,
    num_depth_bins: int = 64,
    sigma_extent: float = 3.0,
    intensity_min: float = 0.0,
    opacity_from_intensity: bool = True,
    opacity_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expects ckpt keys:
      - 'xyz'         [N,3] (assumed in [0,1])
      - 'intensity'   [N,1] or [N]
      - 'scaling'     [N,3] (log-space in your original)
      - 'rotation'    [N,4] (unnormalized quaternion)
      - 'num_gaussians'
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    xyz = ckpt["xyz"].to(device)
    intensity = ckpt["intensity"].to(device).view(-1)
    intensity = torch.sigmoid(intensity).clamp(0.0, 1.0)

    scaling = ckpt["scaling"].to(device)
    scaling = torch.exp(scaling)  # your stored scaling is log-scale

    rotation = ckpt["rotation"].to(device)
    rotation = torch.nn.functional.normalize(rotation, dim=1)

    if "num_gaussians" in ckpt:
        print(f"Loaded {ckpt['num_gaussians']} Gaussians from {checkpoint_path}")
    else:
        print(f"Loaded {xyz.shape[0]} Gaussians from {checkpoint_path}")

    opacity = None
    if mode == "alpha":
        if opacity_from_intensity:
            opacity = (intensity * opacity_scale).clamp(0.0, 1.0)
        else:
            opacity = torch.full_like(intensity, 0.1)  # conservative default

    return render_all_ortho_views(
        xyz_01=xyz,
        intensity=intensity,
        scaling=scaling,
        rotation=rotation,
        volume_size_zyx=volume_size_zyx,
        mode=mode,
        opacity=opacity,
        num_depth_bins=num_depth_bins,
        sigma_extent=sigma_extent,
        intensity_min=intensity_min,
    )


# =============================================================================
# Quick self-test (orthographic)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic Gaussians in [0,1]
    N = 200
    xyz = torch.rand(N, 3, device=device)
    intensity = torch.rand(N, device=device)
    scaling = torch.rand(N, 3, device=device) * 0.03 + 0.01  # small sigmas
    rotation = torch.zeros(N, 4, device=device)
    rotation[:, 0] = 1.0

    Z, Y, X = 128, 128, 128

    top_fmip, front_fmip, side_fmip = render_all_ortho_views(
        xyz, intensity, scaling, rotation, (Z, Y, X),
        mode="field_mip", num_depth_bins=64
    )
    top_pmip, _, _ = render_all_ortho_views(
        xyz, intensity, scaling, rotation, (Z, Y, X),
        mode="primitive_mip"
    )
    top_sum, _, _ = render_all_ortho_views(
        xyz, intensity, scaling, rotation, (Z, Y, X),
        mode="sum"
    )
    top_alpha, _, _ = render_all_ortho_views(
        xyz, intensity, scaling, rotation, (Z, Y, X),
        mode="alpha",
        opacity=intensity.clamp(0.0, 1.0) * 0.2  # conservative opacity
    )

    print("Orthographic test done.")
    print("Field MIP:", float(top_fmip.min()), float(top_fmip.max()))
    print("Primitive MIP:", float(top_pmip.min()), float(top_pmip.max()))
    print("Sum:", float(top_sum.min()), float(top_sum.max()))
    print("Alpha:", float(top_alpha.min()), float(top_alpha.max()))
