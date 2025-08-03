import os
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet                 # Model architecture used in Real-ESRGAN
from realesrgan import RealESRGANer                            # The real upscaling engine

# Choose GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#image_path = "img/5x5_test.jpg"
image_path = "img/5x5_t_upscaled_realersgan_another_4_4.jpg"


# Define model architecture manually to match the pretrained weights
# - num_in_ch: number of input image channels (3 = RGB)
# - num_out_ch: output channels
# - num_feat: number of feature maps
# - num_block: number of RRDB blocks
# - num_grow_ch: growth channels in dense layers
# This must match the settings of the pretrained model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

# Path to the pretrained weights (you must have downloaded this manually)
model_path = 'weights/RealESRGAN_x4plus.pth'

# Create the RealESRGAN upscaler
upscaler = RealESRGANer(
    scale=4,               # 4× upscaling
    model_path=model_path,
    model=model,
    tile=0,                # Use tiling if your GPU runs out of memory
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),  # Use half precision if on GPU
    device=device
)

# Load the input image
img = Image.open(image_path).convert("RGB")
img_np = np.array(img)

# Perform super-resolution
output, _ = upscaler.enhance(img_np, outscale=4)

# Save the result
Image.fromarray(output).save(image_path[:-7]+"_upscaled_realersgan.jpg")
