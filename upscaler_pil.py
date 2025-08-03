from PIL import Image

# Path to the original image file
image_path = "img/notmyphoto_or.jpg"


img = Image.open(image_path)

# Resize the image by 4x in width and height using bicubic interpolation
# - This creates a new image with 16 times more pixels (4x width * 4x height)
# - The resize() method doesn't add real detail—it just generates new pixels by estimating colors
# - BICUBIC interpolation considers a 4x4 neighborhood (16 pixels) around each new pixel
#   and fits smooth curves to estimate the new pixel values
# - This gives smoother results than simpler methods, but doesn't make the image sharper or more detailed
upscaled = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)


upscaled.save(image_path[:-7] + "_upscaled_pil.jpg")
