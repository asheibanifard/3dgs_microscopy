# 3DGS Microscopy - Method 1

3D Gaussian Splatting for Volumetric Microscopy Reconstruction

## Overview

This module provides a modular implementation for reconstructing 3D volumetric microscopy data using 3D Gaussian Splatting. It includes training utilities, multiple rendering modes (MIP, alpha blending), and skeleton-based constraints for neuron reconstruction.

## Features

- **3D Gaussian Splatting** for volumetric reconstruction
- **Multiple rendering modes**: MIP (Maximum Intensity Projection), Alpha Blending, Sum
- **Skeleton-based initialization** from SWC neuron files
- **Anti-aliased downsampling** for coarse-to-fine training
- **Custom loss functions** for background suppression, sparsity, and concentration

## Installation

```bash
# Clone the repository
git clone https://github.com/asheibanifard/3dgs_microscopy.git
cd 3dgs_microscopy

# Install dependencies (requires CUDA)
pip install torch torchvision
pip install diff-gaussian-rasterization  # From 3DGS repo
pip install simple-knn                    # From 3DGS submodules
```

## Project Structure

```
Method1/
├── __init__.py          # Module exports
├── renderer.py          # Gaussian rendering (MIP, Alpha, Sum modes)
├── render_ckpt.py       # Render from saved checkpoints
├── losses.py            # Loss functions (MSE, TV, background, sparsity)
├── metrics.py           # PSNR, SSIM metrics
├── downsampling.py      # Anti-aliased 3D downsampling
├── pruning.py           # Gaussian pruning utilities
├── scheduling.py        # Learning rate scheduling
├── optimization.py      # Optimizer parameters
├── skeleton_loss.py     # SWC skeleton constraint loss
├── train_modular.py     # Modular training script
└── gs_utils/            # CUDA utilities
    ├── discretize_grid.cu
    ├── general_utils.py
    └── Compute_intensity.py
```

## Usage

### Rendering from Checkpoint

```python
from renderer import render_from_checkpoint

# Render with different modes
top, front, side = render_from_checkpoint(
    checkpoint_path="model.pth",
    volume_size_zyx=(100, 647, 813),
    mode="field_mip",  # Options: "field_mip", "primitive_mip", "alpha"
    device="cuda"
)
```

### Rendering Modes

| Mode | Description |
|------|-------------|
| `field_mip` | Voxel-like MIP - renders to volume grid, then takes max projection |
| `primitive_mip` | Primitive MIP - projects 2D Gaussians, takes max per pixel |
| `alpha` | Alpha blending - front-to-back compositing with depth sorting |
| `sum` | Additive blending - simple sum of Gaussian contributions |

### Training

```python
from Method1 import (
    OptimizationParams,
    create_gaussian_kernel_3d,
    antialias_downsample_3d,
    background_suppression_loss,
    psnr_from_mse
)

# Setup training
op = OptimizationParams()
kernel = create_gaussian_kernel_3d(kernel_size=5, sigma=0.7)

# During training loop
loss_bg = background_suppression_loss(output, target)
loss_sparse = sparsity_loss(output, target)
psnr = psnr_from_mse(mse_loss)
```

### Skeleton-based Initialization

```python
from skeleton_loss import SkeletonConstraint, skeleton_distance_loss

# Load skeleton constraint
skeleton = SkeletonConstraint(
    swc_path="neuron.swc",
    volume_size=(50, 324, 407),
    device='cuda'
)

# Compute skeleton loss during training
skel_loss = skeleton_distance_loss(gaussian_xyz, skeleton, weight=0.5)
```

## API Reference

### Rendering Functions

- `render_from_checkpoint()` - Load checkpoint and render all views
- `render_gaussians_orthographic()` - Unified rendering with mode selection
- `render_mip_orthographic()` - Maximum Intensity Projection
- `render_alpha_blending_orthographic()` - Alpha compositing
- `render_all_views()` - Render top/front/side views

### Loss Functions

- `background_suppression_loss()` - Penalize background intensity
- `sparsity_loss()` - Match output sparsity to target
- `concentration_loss()` - Penalize large Gaussians
- `multi_scale_mse_loss()` - Multi-scale MSE
- `tv_regularization()` - Total variation smoothness

### Metrics

- `psnr_from_mse()` - Compute PSNR from MSE
- `ssim_3d_slicewise()` - Slice-wise 3D SSIM

## Configuration

Key parameters in training config:

```yaml
# Gaussian initialization
num_gaussian: 50000
ini_intensity: 0.05
ini_sigma: 0.05

# Training
max_iter: 20000
low_reso_stage: 3000

# Densification
do_density_control: true
max_scale: 0.10
min_scale: 0.012

# Loss weights
skeleton_weight: 0.5
background_weight: 3.0
sparsity_weight: 0.5
overlap_weight: 0.3
```

## Output Examples

The renderer produces orthographic projections:
- `top_*.png` - Top-down view (along Z axis)
- `front_*.png` - Front view (along Y axis)
- `side_*.png` - Side view (along X axis)

## Citation

If you use this code, please cite:

```bibtex
@article{3dgs_microscopy,
  title={3D Gaussian Splatting for Volumetric Microscopy},
  author={Sheibanifard, A.},
  year={2026}
}
```

## License

MIT License
