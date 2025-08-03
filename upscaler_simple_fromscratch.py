import os
import subprocess
import numpy as np
from scipy.ndimage import zoom
from PIL import Image


def convert_to_ppm(input_path, ppm_path):
    """Converts any input image to binary PPM using FFmpeg."""
    print(f"[FFmpeg] Converting {input_path} to {ppm_path}...")
    subprocess.run(['ffmpeg', '-y', '-i', input_path, '-frames:v', '1', '-update', '1', '-pix_fmt', 'rgb24', ppm_path], check=True)


def convert_from_ppm(ppm_path, output_path):
    """Converts back to original format using FFmpeg."""
    print(f"[FFmpeg] Converting {ppm_path} to {output_path}...")
    subprocess.run(['ffmpeg', '-y', '-i', ppm_path, output_path], check=True)


def load_ppm(filename):
    """Loads binary PPM (P6 format) into a NumPy array."""
    with open(filename, 'rb') as f:
        # Read magic number
        assert f.readline().strip() == b'P6'

        # Skip comments and get width, height
        while True:
            line = f.readline()
            if not line.startswith(b'#'):
                width, height = map(int, line.strip().split())
                break

        # Read max color value (usually 255)
        maxval = int(f.readline().strip())
        assert maxval == 255

        # Read the binary image data
        raw_data = f.read(width * height * 3)

    # Convert to NumPy array outside the with-block
    img = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
    return img


def generate_5x5_test_image_ppm(output_path):
    """Generates a 5x5 test image with distinct colors and saves as binary PPM."""
    
    # Define a 5x5 image with RGB values
    # Each row has a different color gradient for variety
    img = np.array([
        [[255,   0,   0], [255, 128,   0], [255, 255,   0], [128, 255,   0], [  0, 255,   0]],
        [[  0, 255, 128], [  0, 255, 255], [  0, 128, 255], [  0,   0, 255], [128,   0, 255]],
        [[255,   0, 255], [255,   0, 128], [128, 128, 128], [ 64,  64,  64], [  0,   0,   0]],
        [[255, 255, 255], [192, 192, 192], [128, 128,   0], [128,   0, 128], [  0, 128, 128]],
        [[ 64,   0,   0], [  0, 64,   0], [  0,   0, 64], [ 64, 64,   0], [  0, 64, 64]]
    ], dtype=np.uint8)

    height, width, _ = img.shape

    # Write as binary PPM (P6)
    with open(output_path, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode('ascii'))
        f.write(img.tobytes())


def save_ppm(filename, img):
    """Saves a Numpy image array as binary PPM(P6)"""
    height, width, _ = img.shape
    with open(filename, 'wb') as f:
        f.write(b'P6\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(img.tobytes())





def box_filter_interpolate(img, scale):
    """Upscale using simple box filtering (mean of 4 nearest neighbors)."""

    h, w, c = img.shape
    new_h, new_w = int(h * scale), int(w * scale)

    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Map back to original coordinates
            x = i / scale
            y = j / scale

            # Get the 4 nearest neighbors (floor and ceil)
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, h - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, w - 1)

            # Average their RGB values
            pixel = (
                img[x0, y0].astype(np.uint16) +
                img[x0, y1].astype(np.uint16) +
                img[x1, y0].astype(np.uint16) +
                img[x1, y1].astype(np.uint16)
            ) // 4

            result[i, j] = pixel.astype(np.uint8)

    return result


def nearest_neighbor_interpolate_special(img, scale):
    """Upscales an image using nearest neighbor interpolation.

    For each pixel in the output image, this method finds the closest pixel 
    in the input image and simply copies its color value.

    This is the simplest method. No blending or smoothing. Uses round() to create special pattern since it always rounds towards even.
    """

    h, w, c = img.shape
    new_h, new_w = int(h * scale), int(w * scale)

    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Map output pixel (i,j) back to input coordinates
            x = int(round(i / scale))   # Row in input
            y = int(round(j / scale))   # Column in input

            # Clamp coordinates to stay in bounds
            x = min(x, h - 1)
            y = min(y, w - 1)

            # Copy nearest pixel value directly
            result[i, j] = img[x, y]

    return result

def nearest_neighbor_interpolate(img, scale):
    """Upscales an image using nearest neighbor interpolation.

    For each pixel in the output image, this method finds the closest pixel 
    in the input image and simply copies its color value.

    This is the simplest method. No blending or smoothing.
    """

    h, w, c = img.shape
    new_h, new_w = int(h * scale), int(w * scale)

    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Map output pixel (i,j) back to input coordinates
            x = int(np.floor(i / scale))   # Row in input
            y = int(np.floor(j / scale))   # Column in input

            # Clamp coordinates to stay in bounds
            x = min(x, h - 1)
            y = min(y, w - 1)

            # Copy nearest pixel value directly
            result[i, j] = img[x, y]

    return result


def bilinear_interpolate(img, scale):
    """Upscales an image using bilinear interpolation.

    For each output pixel, this method:
    - Maps it to a floating-point coordinate in the input image
    - Identifies the 4 surrounding input pixels
    - Linearly blends them based on how close the target point is to each
    """
 
    h, w, c = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    result = np.zeros((new_h, new_w, c), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale
            y = j / scale

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, h - 1)
            y1 = min(y0 + 1, w - 1)

            dx = x - x0
            dy = y - y0

            for k in range(c):
                a = img[x0, y0, k]
                b = img[x0, y1, k]
                c_ = img[x1, y0, k]
                d = img[x1, y1, k]

                value = (
                    a * (1 - dx) * (1 - dy) +
                    b * (1 - dx) * dy +
                    c_ * dx * (1 - dy) +
                    d * dx * dy
                )

                result[i, j, k] = int(value)
            
    return result


def bicubic_interpolate(img, scale):
    """Upscales image using bicubic interpolation.

    This method uses 4x4 (16) nearby pixels and cubic functions to interpolate
    each new pixel value. It produces smoother results than bilinear.

    We're using SciPy's built-in `zoom()` with `order=3`:
    - order=0: Nearest
    - order=1: Bilinear
    - order=3: Bicubic
    """

    return zoom(img, (scale, scale, 1), order=3).astype(np.uint8)


def lanczos_interpolate(img, scale):
    """Upscales using Lanczos resampling (high-quality).

    Lanczos uses a sinc-based kernel to interpolate pixel values,
    looking at a wide window of surrounding pixels (usually 8).

    It preserves edge sharpness better than bicubic and is often
    used in professional photo editing tools.

    This function uses PIL's built-in Lanczos mode.
    """

    pil_img = Image.fromarray(img)
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    resized = pil_img.resize(new_size, resample=Image.LANCZOS)
    return np.array(resized)


def upscale_image(input_file, output_file, scale, method='bilinear'):
    """
    Full Pipeline:
    1. Convert input image to PPM
    2. Load as NumPy array
    3. Apply selected upscaling method
    4. Save PPM output
    5. Convert back to image
    6. Clean temp files

    Parameters:
        input_file (str): Path to input image
        output_file (str): Path to save final upscaled image
        scale (float): Upscaling factor (e.g., 2.0, 4.0)
        method (str): Upscaling method ('nearest', 'bilinear', 'bicubic', 'lanczos')
    """

    tmp_input_ppm = 'temp_input.ppm'
    tmp_output_ppm = 'temp_output.ppm'

    convert_to_ppm(input_file, tmp_input_ppm)
    img = load_ppm(tmp_input_ppm)

    #generate_5x5_test_image_ppm('5x5_test.ppm')
    #tmp_input_ppm = '5x5_test.ppm'
    #img = load_ppm(tmp_input_ppm)


    # Select the method
    if method == 'nearest':
        upscaled = nearest_neighbor_interpolate(img, scale)
    elif method == 'box':
        upscaled = box_filter_interpolate(img, scale)
    elif method == 'bilinear':
        upscaled = bilinear_interpolate(img, scale)
    elif method == 'bicubic':
        upscaled = bicubic_interpolate(img, scale)
    elif method == 'lanczos':
        upscaled = lanczos_interpolate(img, scale)
    else:
        raise ValueError(f"Unknown upscaling method: '{method}'")

    save_ppm(tmp_output_ppm, upscaled)
    convert_from_ppm(tmp_output_ppm, output_file)

    #os.remove(tmp_input_ppm)
    os.remove(tmp_output_ppm)

    print(f"Done. Upscaled image saved to '{output_file}' using '{method}' interpolation.")




def main():
    scale_factor = 32.0
    #input_path = "img/mountain_moon_or.jpg"
    input_path = "img/5x5_test.jpg"


    # List of available methods
    methods = ['nearest', 'box', 'bilinear', 'bicubic', 'lanczos']

    for method in methods:
        # Construct output path for each method
        output_path = input_path[:-7] + f"_exp_upscaled_{int(scale_factor)}x_{method}.jpg"
        upscale_image(input_path, output_path, scale_factor, method=method)
    
    #scale_factor = 8.0
    #method = 'bilinear'
    #output_path = input_path[:-7] + f"_upscaled_{int(scale_factor)}x_{method}.jpg"
    #upscale_image(input_path, output_path, scale_factor, method=method)


if __name__ == "__main__":
    main()