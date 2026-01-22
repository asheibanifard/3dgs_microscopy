"""
Skeleton constraint loss for 3DGR-CT

Penalizes Gaussians that drift too far from the SWC skeleton.
"""

import torch
import numpy as np
from swc_utils import parse_swc_file, swc_to_arrays, get_skeleton_bounds, densify_skeleton


class SkeletonConstraint:
    """
    Computes distance from Gaussians to the nearest skeleton point.
    Used to penalize Gaussians that drift away from the neuron structure.
    """
    
    def __init__(self, swc_path: str, volume_size: tuple, device='cuda', 
                 densify_points=True, points_per_unit=2.0, max_skeleton_points=10000):
        """
        Args:
            swc_path: Path to SWC file
            volume_size: (D, H, W) volume dimensions
            device: torch device
            densify_points: Whether to densify skeleton for better coverage
            points_per_unit: Density of skeleton points for distance computation
            max_skeleton_points: Maximum skeleton points (subsample if needed to save memory)
        """
        self.device = device
        self.volume_size = volume_size
        D, H, W = volume_size
        
        # Parse SWC
        nodes = parse_swc_file(swc_path)
        positions, radii, parent_ids = swc_to_arrays(nodes)
        
        # Densify skeleton for better distance computation
        if densify_points:
            positions, radii = densify_skeleton(positions, radii, parent_ids, 
                                                points_per_unit=points_per_unit)
        
        # Subsample skeleton points if too many (to save GPU memory)
        if len(positions) > max_skeleton_points:
            indices = np.linspace(0, len(positions)-1, max_skeleton_points, dtype=int)
            positions = positions[indices]
            radii = radii[indices]
            print(f"  Subsampled skeleton to {max_skeleton_points} points (from {len(indices)*points_per_unit:.0f})")
        
        # Get bounds for normalization (same as swc_utils.py)
        min_bounds, max_bounds = get_skeleton_bounds(positions)
        extent = max_bounds - min_bounds
        extent = np.where(extent < 1e-6, 1.0, extent)
        
        # Convert SWC (X, Y, Z) to volume (D, H, W) = (Z, Y, X) coordinate order
        # This must match the coordinate transform in swc_utils.py
        positions_dhw = positions[:, [2, 1, 0]]  # Reorder: X,Y,Z -> Z,Y,X
        min_bounds_dhw = min_bounds[[2, 1, 0]]
        extent_dhw = extent[[2, 1, 0]]
        
        # Normalize to [0, 1] then apply margin (same as swc_utils.normalize_positions)
        margin = 0.05
        normalized = (positions_dhw - min_bounds_dhw) / extent_dhw
        skeleton_coords = normalized * (1 - 2 * margin) + margin  # Apply same margin as swc_utils
        
        self.skeleton_points = torch.tensor(skeleton_coords, dtype=torch.float32, device=device)
        self.skeleton_radii = torch.tensor(radii, dtype=torch.float32, device=device)
        
        # Normalize radii to volume scale
        max_dim = max(D, H, W)
        self.skeleton_radii = self.skeleton_radii / max_dim
        
        print(f"SkeletonConstraint: {len(self.skeleton_points)} skeleton points")
        print(f"  Position range: [{self.skeleton_points.min():.4f}, {self.skeleton_points.max():.4f}]")
    
    def compute_distance_loss(
        self,
        xyz: torch.Tensor,                 # [N,3] in [0,1]
        margin: float = 0.0,
        use_radius: bool = True,
        radius_mult: float = 2.0,
        chunk_xyz: int = 2048,
        chunk_skel: int = 8192,
    ) -> torch.Tensor:
        """
        Compute min distance from each Gaussian center to the skeleton.
        Uses squared distances for efficiency, sqrt only after min.
        """

        xyz = xyz.clamp(0.0, 1.0)
        N = xyz.shape[0]
        skel = self.skeleton_points          # [M,3]
        radii = self.skeleton_radii          # [M]

        min_dist_sq = torch.full((N,), float("inf"), device=xyz.device, dtype=xyz.dtype)

        # Chunk over xyz
        for i in range(0, N, chunk_xyz):
            end_i = min(i + chunk_xyz, N)
            x = xyz[i:end_i]  # [C,3]
            best = torch.full((end_i - i,), float("inf"), device=xyz.device, dtype=xyz.dtype)

            # Chunk over skeleton points
            for j in range(0, skel.shape[0], chunk_skel):
                end_j = min(j + chunk_skel, skel.shape[0])
                s = skel[j:end_j]  # [K,3]

                # squared distances: [C,K]
                # (x[:,None,:] - s[None,:,:])^2 summed on last dim
                d2 = (x[:, None, :] - s[None, :, :]).pow(2).sum(dim=-1)

                if use_radius:
                    # Convert radius to squared allowance approximately:
                    # We compare sqrt(d2) - r*mult, but avoid sqrt by using:
                    # d = sqrt(d2) - a  => penalty on positive. We keep d2 for min selection,
                    # then apply radius after selecting best point index.
                    # So: keep pure geometry here, handle radius after argmin.
                    pass

                best = torch.minimum(best, d2.min(dim=1).values)

            min_dist_sq[i:end_i] = best

        # Now apply radius allowance more correctly: we need the nearest skeleton point index.
        # If you need radius-aware nearest, compute argmin too; simplest is to skip radii or approximate.
        # A practical compromise: apply a global allowance radius (median skeleton radius).
        dist = torch.sqrt(min_dist_sq + 1e-12)

        if use_radius:
            # Conservative global allowance (stable, cheap):
            allowance = float(self.skeleton_radii.median().item()) * radius_mult
            dist = dist - allowance

        if margin > 0:
            dist = dist - margin

        dist = F.relu(dist)
        return (dist ** 2).mean()

    
    def compute_soft_constraint(self, xyz: torch.Tensor, 
                                 sigma: float = 0.05) -> torch.Tensor:
        """
        Soft constraint using Gaussian proximity to skeleton.
        Higher values = closer to skeleton = less penalty.
        
        Args:
            xyz: Gaussian positions [N, 3]
            sigma: Softness of constraint
            
        Returns:
            loss: Mean soft constraint loss
        """
        N = xyz.shape[0]
        chunk_size = 10000
        
        proximity_scores = torch.zeros(N, device=self.device)
        
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            xyz_chunk = xyz[i:end]
            
            # Distance to all skeleton points
            diff = xyz_chunk.unsqueeze(1) - self.skeleton_points.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=-1)  # [C, M]
            
            # Gaussian proximity score
            scores = torch.exp(-dist_sq / (2 * sigma ** 2))  # [C, M]
            
            # Max proximity to any skeleton point
            max_scores, _ = scores.max(dim=1)  # [C]
            proximity_scores[i:end] = max_scores
        
        # Loss = 1 - proximity (want high proximity)
        loss = (1.0 - proximity_scores).mean()
        
        return loss


def skeleton_distance_loss(xyz: torch.Tensor, 
                           skeleton_constraint: SkeletonConstraint,
                           weight: float = 1.0,
                           use_soft: bool = False) -> torch.Tensor:
    """
    Convenience function for computing skeleton constraint loss.
    
    Args:
        xyz: Gaussian positions [N, 3]
        skeleton_constraint: SkeletonConstraint object
        weight: Loss weight
        use_soft: Use soft Gaussian constraint instead of hard distance
        
    Returns:
        Weighted skeleton loss
    """
    if use_soft:
        loss = skeleton_constraint.compute_soft_constraint(xyz)
    else:
        loss = skeleton_constraint.compute_distance_loss(xyz)
    
    return weight * loss
