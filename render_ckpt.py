import torch
from renderer import render_from_checkpoint

ckpt_path = "../outputs/gaussian_swc/tiff/checkpoints/model_iter_7000.pth"

# Your volume size in (Z, Y, X)
volume_size_zyx = (100, 647, 813)

# Best match to voxel/GT MIP:
top, front, side = render_from_checkpoint(
    checkpoint_path=ckpt_path,
    volume_size_zyx=volume_size_zyx,
    mode="field_mip",          # <- use this for voxel-like MIP
    device="cuda",
    num_depth_bins=64,         # 64 or 128 are typical
    sigma_extent=3.0,
    intensity_min=0.01
)

# Debug: max over projected primitives (your "MIP splatting")
top_p, front_p, side_p = render_from_checkpoint(
    checkpoint_path=ckpt_path,
    volume_size_zyx=volume_size_zyx,
    mode="primitive_mip",
    device="cuda",
    sigma_extent=3.0,
    intensity_min=0.01
)

# Alpha blending (need opacity; default uses intensity unless you override)
top_a, front_a, side_a = render_from_checkpoint(
    checkpoint_path=ckpt_path,
    volume_size_zyx=volume_size_zyx,
    mode="alpha",
    device="cuda",
    opacity_scale=0.2,         # tune this to reduce haze
    sigma_extent=3.0,
    intensity_min=0.01
)

# Save results (simple Torch save; you can also convert to PNG with torchvision/PIL)
torch.save({"top": top, "front": front, "side": side}, "field_mip_views.pt")
torch.save({"top": top_p, "front": front_p, "side": side_p}, "primitive_mip_views.pt")
torch.save({"top": top_a, "front": front_a, "side": side_a}, "alpha_views.pt")

print("Done.")

from torchvision.utils import save_image

def save_gray_png(img_hw: torch.Tensor, path: str):
    # img_hw: [H, W]
    x = img_hw.float()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # normalize for display
    x = x.unsqueeze(0)  # [1, H, W]
    save_image(x, path)  # saves exact HxW pixels

Z, Y, X = 647, 813, 100
top, front, side = render_from_checkpoint(checkpoint_path=ckpt_path, volume_size_zyx=(Z,Y,X), mode="field_mip")
save_gray_png(top, "top_field_mip.png")
save_gray_png(front, "front_field_mip.png")
save_gray_png(side, "side_field_mip.png")