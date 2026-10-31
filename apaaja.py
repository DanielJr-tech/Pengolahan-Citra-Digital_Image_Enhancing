import os
import imageio.v3 as iio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# ---------- 1. Baca citra (path mengikuti lokasi file .py) ----------
folder = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(folder, "plat.jpg")
image = iio.imread(path)

# Ubah ke grayscale (ekualisasi histogram bekerja pada 1 kanal)
if image.ndim == 3:
    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
image = np.clip(image, 0, 255).astype(np.uint8)


# ---------- 2. Ekualisasi histogram ----------
def hist_equal(img):
    # Langkah 1: histogram (jumlah piksel tiap intensitas 0-255) via scipy.ndimage
    hist = ndimage.histogram(img, min=0, max=255, bins=256)
    # Langkah 2: CDF (jumlah kumulatif)
    cdf = hist.cumsum()
    # Langkah 3: normalisasi CDF ke rentang 0-255
    cdf_min = cdf[cdf > 0].min()
    cdf_normal = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255
    cdf_normal = np.clip(cdf_normal, 0, 255).astype(np.uint8)
    # Langkah 4: petakan setiap piksel ke nilai barunya
    return cdf_normal[img]


# ---------- 3. Proses citra ----------
gelap = np.clip(image.astype(np.float32) - 60, 0, 255).astype(np.uint8)  # citra kontras rendah
result = hist_equal(gelap)
smooth = ndimage.gaussian_filter(result, sigma=1)  # opsional: haluskan noise

hist_img = ndimage.histogram(gelap, 0, 255, 256)
hist_result = ndimage.histogram(result, 0, 255, 256)

# ---------- 4. Tampilkan ----------
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(gelap, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Awal")

plt.subplot(2, 2, 2)
plt.imshow(result, cmap="gray", vmin=0, vmax=255)
plt.title("Hasil Ekualisasi Histogram")

plt.subplot(2, 2, 3)
plt.plot(hist_img)
plt.title("Histogram Awal")

plt.subplot(2, 2, 4)
plt.plot(hist_result)
plt.title("Histogram Hasil")

plt.tight_layout()
plt.savefig(os.path.join(folder, "hasil.png"))
plt.show()