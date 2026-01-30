"""Generate test images for testing image_sorters.py"""

import numpy as np
from PIL import Image
from pathlib import Path

output_dir = Path("test_data/images")
output_dir.mkdir(parents=True, exist_ok=True)

SIZE = (50, 50)


def save_image(arr, name):
    """Save numpy array as image."""
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(output_dir / f"{name}.png")
    print(f"Created {name}.png")


# === Brightness variations ===
# Dark image
dark = np.full((*SIZE, 3), 30)
save_image(dark, "brightness_dark")

# Medium brightness
medium = np.full((*SIZE, 3), 128)
save_image(medium, "brightness_medium")

# Bright image
bright = np.full((*SIZE, 3), 230)
save_image(bright, "brightness_bright")


# === Contrast variations ===
# Low contrast (all similar gray values)
low_contrast = np.random.normal(128, 10, (*SIZE, 3)).clip(0, 255)
save_image(low_contrast, "contrast_low")

# High contrast (black and white checkerboard)
high_contrast = np.zeros((*SIZE, 3))
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        if (i // 5 + j // 5) % 2 == 0:
            high_contrast[i, j] = 255
save_image(high_contrast, "contrast_high")


# === Color variance variations ===
# Grayscale-like (low color variance)
grayscale_like = np.zeros((*SIZE, 3))
gray_val = np.random.randint(100, 150, SIZE)
grayscale_like[:, :, 0] = gray_val
grayscale_like[:, :, 1] = gray_val
grayscale_like[:, :, 2] = gray_val
save_image(grayscale_like, "color_grayscale")

# Colorful (high color variance) - rainbow gradient
colorful = np.zeros((*SIZE, 3))
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        colorful[i, j, 0] = (i * 5) % 256  # Red varies with row
        colorful[i, j, 1] = (j * 5) % 256  # Green varies with col
        colorful[i, j, 2] = ((i + j) * 3) % 256  # Blue varies diagonally
save_image(colorful, "color_vibrant")


# === Dominant color variations ===
# Red dominant
red_dominant = np.full((*SIZE, 3), [200, 50, 50])
save_image(red_dominant, "dominant_red")

# Green dominant
green_dominant = np.full((*SIZE, 3), [50, 200, 50])
save_image(green_dominant, "dominant_green")

# Blue dominant
blue_dominant = np.full((*SIZE, 3), [50, 50, 200])
save_image(blue_dominant, "dominant_blue")


# === Compression ratio variations ===
# Simple (solid color - compresses well)
simple = np.full((*SIZE, 3), [100, 150, 200])
save_image(simple, "compress_simple")

# Complex (random noise - compresses poorly)
complex_noise = np.random.randint(0, 256, (*SIZE, 3))
save_image(complex_noise, "compress_complex")


# === Frequency content variations ===
# Low frequency (smooth gradient)
low_freq = np.zeros((*SIZE, 3))
for i in range(SIZE[0]):
    val = int(255 * i / SIZE[0])
    low_freq[i, :] = [val, val, val]
save_image(low_freq, "freq_smooth")

# High frequency (fine stripes)
high_freq = np.zeros((*SIZE, 3))
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        if (i + j) % 2 == 0:
            high_freq[i, j] = [200, 200, 200]
        else:
            high_freq[i, j] = [50, 50, 50]
save_image(high_freq, "freq_detailed")

# Medium frequency (larger pattern)
medium_freq = np.zeros((*SIZE, 3))
for i in range(SIZE[0]):
    for j in range(SIZE[1]):
        if (i // 10 + j // 10) % 2 == 0:
            medium_freq[i, j] = [180, 180, 180]
        else:
            medium_freq[i, j] = [80, 80, 80]
save_image(medium_freq, "freq_medium")


print(f"\nGenerated {len(list(output_dir.glob('*.png')))} test images in {output_dir}/")
