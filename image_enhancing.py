import imageio.v3 as image
import numpy as np
import matplotlib.pyplot as plt

path = "plat.jpg"
my_image = image.imread(path)

def brightness(img, factor):
    img_result = img.astype(np.float32)
    img_result = img_result + factor
    img_result = np.clip(img_result, 0, 255)
    return img_result.astype(np.uint8)

def contrast(img, factor):
    mean = 128
    img_result = img.astype(np.float32)
    img_result = mean + factor * (img_result - mean)
    img_result = np.clip(img_result, 0, 255)
    return img_result.astype(np.uint8)

bright = brightness(my_image, -100)
contrast = contrast(my_image, -2)

plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.imshow(my_image, origin="upper")
plt.title("Gambar Asli")

plt.subplot(1,3,2)
plt.imshow(bright, origin="upper")
plt.title("Gambar Brightness")

plt.subplot(1,3,3)
plt.imshow(contrast, origin="upper")
plt.title("Gambar Contrast")

plt.tight_layout()

plt.show()