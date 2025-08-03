# 🔍 Image Upscaling Experiments

This project demonstrates how to upscale images using a variety of methods — from classical interpolation (like Lanczos) to the deep learning–based **Real-ESRGAN**. It’s built with Python 3.12 on Ubuntu and includes experiments on both real images and synthetic patterns.

We use **Real-ESRGAN with 4× upscaling** and the official pretrained model:

> `RealESRGAN_x4plus.pth`  
> (Available from the [Real-ESRGAN GitHub repo](https://github.com/xinntao/Real-ESRGAN))

📖 Read the full blog post with images and explanations here:  
👉 [nonzeroviz.github.io/readme/imageupscaling.html](https://nonzeroviz.github.io/readme/imageupscaling.html)

---

## 🚀 Features

- ✅ Traditional upscaling using nearest neighbor, bilinear, bicubic, and Lanczos
- 🧠 Neural network-based enhancement using Real-ESRGAN (4×)
- 🧪 Experimental results on very low-res input (e.g., 5×5 pixel test patterns)
- 🎨 Visualization of Real-ESRGAN's smoothing and detail "hallucination"

---

## 🛠️ Requirements

- Python 3.12+
- PyTorch (1.12+ recommended)
- NumPy
- Pillow
- `ffmpeg` (required for image format conversions)
- Real-ESRGAN and its dependencies