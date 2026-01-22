"""
Artifact-Free 3D Gaussian Splatting Training (Modular Version)

This is the main training script that uses modular components from the training/ folder.
"""

import os
import argparse
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from Method1.skeleton_loss import SkeletonConstraint, skeleton_distance_loss
from models.gaussian_model import GaussianModel

# Import modular training components
from Method1 import (
    # Downsampling
    create_gaussian_kernel_3d,
    antialias_downsample_3d,
    antialias_downsample_grid,
    # Losses
    background_suppression_loss,
    sparsity_loss,
    concentration_loss,
    multi_scale_mse_loss,
    tv_regularization,
    # Pruning
    prune_background_gaussians,
    # Scheduling
    get_resolution_blend_weights,
    # Metrics
    psnr_from_mse,
    ssim_3d_slicewise,
    # Optimization
    OptimizationParams,
)

import wandb

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--swc_path', type=str, default=None, help="Path to SWC skeleton file")
    return parser.parse_args()


def setup_output_directory(opts, config):
    """Setup output folders and copy config."""
    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    output_subfolder = config['data']
    model_name = os.path.join(output_folder, output_subfolder)
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
    return checkpoint_directory, image_directory


def prepare_data(config):
    """Load data and create low-resolution versions."""
    print(f'Loading image: {config["img_path"]}')
    data_loader = get_data_loader(
        config['data'], config['img_path'], config['img_size'],
        img_slice=None, train=True, batch_size=config['batch_size']
    )
    
    config['img_size'] = (config['img_size'],) * 3 if isinstance(config['img_size'], int) else tuple(config['img_size'])
    slice_idx = list(range(0, config['img_size'][0], max(1, config['img_size'][0] // config['display_image_num'])))
    
    return data_loader, slice_idx


def setup_gaussians(opts, config):
    """Initialize Gaussian model from SWC skeleton."""
    swc_path = opts.swc_path or config.get('swc_path', None)
    if swc_path is None:
        raise ValueError("SWC path required. Use --swc_path or set in config.")
    
    op = OptimizationParams()
    gaussians = GaussianModel()
    
    volume_size = tuple(s // 2 for s in config['img_size'])
    gaussians.create_from_swc(
        swc_path=swc_path,
        volume_size=volume_size,
        num_samples=config['num_gaussian'],
        ini_intensity=config['ini_intensity'],
        spatial_lr_scale=config['spatial_lr_scale'],
        densify=config.get('swc_densify', True),
        points_per_unit=config.get('swc_points_per_unit', 5.0),
        radius_based_density=config.get('swc_radius_density', True)
    )
    
    skeleton_constraint = SkeletonConstraint(
        swc_path=swc_path,
        volume_size=volume_size,
        device='cuda',
        densify_points=True,
        points_per_unit=1.0,
        max_skeleton_points=2000
    )
    
    gaussians.training_setup(op)
    
    return gaussians, skeleton_constraint


def train_step(
    iteration, gaussians, grid, image, grid_low_reso, image_low_reso,
    gaussian_kernel, config, skeleton_constraint
):
    """Perform a single training step."""
    low_reso_stage = config['low_reso_stage']
    blend_window = config.get('blend_window', 500)
    tv_weight = config['tv_weight']
    background_weight = config.get('background_weight', 1.0)
    sparsity_weight = config.get('sparsity_weight', 0.5)
    concentration_weight = config.get('concentration_weight', 0.1)
    skeleton_weight = config.get('skeleton_weight', 0.5)
    use_multiscale = config.get('use_multiscale_loss', True)
    
    gaussians.update_learning_rate(iteration)
    low_weight, high_weight = get_resolution_blend_weights(iteration, low_reso_stage, blend_window)

    # Forward + base MSE losses
    if high_weight == 0.0:
        train_output = gaussians.grid_sample(grid_low_reso)
        target_image = image_low_reso
        if use_multiscale:
            loss_mse = multi_scale_mse_loss(train_output, target_image, weights=(1.0, 0.3), kernel=gaussian_kernel)
        else:
            loss_mse = F.mse_loss(train_output, target_image)

    elif low_weight == 0.0:
        train_output = gaussians.grid_sample(grid)
        target_image = image
        if use_multiscale:
            loss_mse = multi_scale_mse_loss(train_output, target_image, weights=(1.0, 0.3), kernel=gaussian_kernel)
        else:
            loss_mse = F.mse_loss(train_output, target_image)

    else:
        # Blend window
        out_low = gaussians.grid_sample(grid_low_reso)
        out_high = gaussians.grid_sample(grid)
        loss_low = F.mse_loss(out_low, image_low_reso)
        loss_high = F.mse_loss(out_high, image)
        loss_mse = low_weight * loss_low + high_weight * loss_high
        train_output = out_high
        target_image = image

    # Regularizers (weighted during blend)
    reg_w = float(high_weight) if (0.0 < high_weight < 1.0) else 1.0
    reg_w = 0.2 + 0.8 * reg_w

    loss_tv = (reg_w * tv_weight * tv_regularization(train_output)) if tv_weight > 0 else torch.tensor(0.0, device='cuda')
    loss_bg = reg_w * background_weight * background_suppression_loss(train_output, target_image)
    loss_sparse = reg_w * sparsity_weight * sparsity_loss(train_output, target_image)
    loss_conc = concentration_weight * concentration_loss(gaussians)
    skel_loss = skeleton_distance_loss(gaussians._xyz, skeleton_constraint, weight=skeleton_weight)

    loss = loss_mse + loss_tv + loss_bg + loss_sparse + loss_conc + skel_loss

    return loss, loss_mse, loss_bg, loss_sparse, train_output, target_image, high_weight


def main():
    opts = parse_args()
    config = get_config(opts.config)
    max_iter = config['max_iter']
    
    checkpoint_directory, image_directory = setup_output_directory(opts, config)
    
    wandb.init(project="3dgr-ct-skeleton", name="modular_training", config=config, mode="disabled")
    
    print("\n" + "=" * 60)
    print("ARTIFACT-FREE VOLUMETRIC TRAINING (MODULAR)")
    print("  - Anti-aliased downsampling (no grid artifacts)")
    print("  - Background suppression (soft; reduces haze)")
    print("  - Smooth coarse-to-fine blending")
    print("=" * 60 + "\n")
    
    data_loader, slice_idx = prepare_data(config)
    
    for it, (grid, image) in enumerate(data_loader):
        grid = grid.cuda()
        image = image.cuda()
        
        # Create anti-aliased low-resolution data
        print("Creating anti-aliased low-resolution data...")
        gaussian_kernel = create_gaussian_kernel_3d(kernel_size=5, sigma=0.7, device='cuda')
        
        image_low_reso = antialias_downsample_3d(image, factor=2, kernel=gaussian_kernel)
        low_reso_size = image_low_reso.shape[1:4]
        grid_low_reso = antialias_downsample_grid(grid, factor=2, target_size=low_reso_size)
        
        print(f"  High-res: {image.shape}")
        print(f"  Low-res:  {image_low_reso.shape}")
        
        test_data = (grid, image)
        save_image_3d(test_data[1], slice_idx, os.path.join(image_directory, "test.png"))
        
        # Setup Gaussians
        gaussians, skeleton_constraint = setup_gaussians(opts, config)
        
        # Density control setup
        do_density_control = config.get('do_density_control', False)
        max_scale = config.get('max_scale', None)
        max_log_scale = torch.log(torch.tensor(max_scale, device='cuda')) if max_scale else None
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            # Training step
            loss, loss_mse, loss_bg, loss_sparse, train_output, target_image, high_weight = train_step(
                iteration, gaussians, grid, image, grid_low_reso, image_low_reso,
                gaussian_kernel, config, skeleton_constraint
            )
            
            loss.backward()
            
            if do_density_control:
                gaussians.add_densification_stats()
            
            gaussians.optimizer.step()
            
            # Clamp scales
            if max_log_scale is not None:
                gaussians._scaling.data.clamp_(max=max_log_scale)
            
            # Density control
            if do_density_control:
                with torch.no_grad():
                    n_gauss = gaussians.get_xyz.shape[0]
                    if (n_gauss < config['max_gaussians'] and
                        iteration < config['densify_until_iter'] and
                        iteration > config['densify_from_iter'] and
                        iteration % config['densification_interval'] == 0):
                        gaussians.densify_and_prune(
                            config['max_grad'], config['min_intensity'],
                            sigma_extent=config['sigma_extent'], max_scale=max_scale
                        )
                        gaussians.reset_densification_stats()
            
            # Prune background Gaussians
            if iteration > 0 and iteration % 200 == 0:
                n_pruned = prune_background_gaussians(gaussians, image)
                if n_pruned > 0:
                    print(f"  [Pruned {n_pruned} background Gaussians]")
            
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if (iteration + 1) % config['log_iter'] == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (iteration + 1)
                eta = (max_iter - iteration - 1) * avg_time
                
                mse_for_psnr = F.mse_loss(train_output.detach(), target_image)
                train_psnr = psnr_from_mse(mse_for_psnr, data_range=1.0)
                
                print(
                    f"[Iter: {iteration + 1}/{max_iter}] Loss: {loss.item():.4g} | "
                    f"MSE: {loss_mse.item():.4g} | BG: {loss_bg.item():.4g} | "
                    f"PSNR: {train_psnr:.2f}dB | #G: {gaussians.get_xyz.shape[0]} | "
                    f"Time: {avg_time:.2f}s | ETA: {eta:.1f}s"
                )
            
            # Checkpoints
            if (iteration + 1) % 1000 == 0:
                checkpoint_path = os.path.join(checkpoint_directory, f"model_iter_{iteration + 1}.pth")
                torch.save({
                    'iteration': iteration + 1,
                    'xyz': gaussians._xyz.data,
                    'intensity': gaussians._intensity.data,
                    'scaling': gaussians._scaling.data,
                    'rotation': gaussians._rotation.data,
                    'num_gaussians': gaussians._xyz.shape[0],
                    'config': config,
                }, checkpoint_path)
                print(f"  [Checkpoint: {checkpoint_path}]")
            
            # Validation
            if iteration == 0 or (iteration + 1) % config['val_iter'] == 0:
                with torch.no_grad():
                    test_output = gaussians.grid_sample(test_data[0])
                    mse = F.mse_loss(test_output, test_data[1])
                    test_psnr = psnr_from_mse(mse, data_range=1.0)
                    test_ssim = ssim_3d_slicewise(test_output, test_data[1], data_range=1.0)
                
                save_image_3d(
                    test_output, slice_idx,
                    os.path.join(image_directory, f"recon_{iteration + 1}_{test_psnr:.3f}dB_ssim{test_ssim:.4f}.png")
                )
        
        # Final save
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        model_path = os.path.join(checkpoint_directory, "final_model.pth")
        torch.save({
            'xyz': gaussians._xyz.data,
            'intensity': gaussians._intensity.data,
            'scaling': gaussians._scaling.data,
            'rotation': gaussians._rotation.data,
            'num_gaussians': gaussians._xyz.shape[0],
            'config': config,
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Final Gaussians: {gaussians._xyz.shape[0]}")
        print(f"Final Val PSNR: {test_psnr:.2f} dB")
        print(f"Final Val SSIM: {test_ssim:.4f}")
        
        break  # Single volume training


if __name__ == "__main__":
    main()
