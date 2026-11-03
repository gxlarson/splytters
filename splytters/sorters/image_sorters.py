"""
Sorting algorithms for adversarial image dataset partitioning.

These functions rank images by visual properties (brightness, contrast, color,
complexity) to enable train-test splits that maximize dissimilarity.

All functions accept either file paths or PIL Image objects.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import fft
from sklearn.cluster import KMeans

from splytters.sorters._lazy import LazyModule

# Pillow is imported on first use (see _lazy), so importing this module needs no
# [image] extra — only calling a sorter does.
Image = LazyModule("PIL.Image", extra="image")

if TYPE_CHECKING:
    from PIL import Image as _PILImage

    ImageInput = str | Path | _PILImage.Image


def _load_image(img: ImageInput) -> Image.Image:
    """Load image from path or return PIL Image directly."""
    if isinstance(img, (str, Path)):
        # Image.open is lazy and keeps the file handle open until the pixels are
        # accessed; force the load inside a context manager so the fd is closed
        # before we return. Otherwise sorting a large image list leaks one fd
        # per image and can exhaust the descriptor limit (notably on Windows).
        with Image.open(img) as opened:
            opened.load()
            return opened
    if isinstance(img, Image.Image):
        return img
    raise TypeError(
        f"Expected a file path or PIL Image, got {type(img).__name__}"
    )


def _to_array(img: ImageInput, mode: str | None = None) -> np.ndarray:
    """Convert image to numpy array, optionally converting mode first."""
    img = _load_image(img)
    if mode is not None:
        img = img.convert(mode)
    return np.array(img, dtype=np.float32)


def mean_brightness(
    images: list[ImageInput], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by mean brightness (average pixel intensity).

    Converts images to grayscale and computes mean intensity (0-255 scale).

    Useful for adversarial splits: train on mid-tone images,
    test on very dark or very bright images.

    Args:
        images: list of file paths or PIL Image objects
        low_first: if True, darker images first; if False, brighter images first

    Returns:
        List of (index, brightness) tuples sorted by mean brightness.
    """
    scores = []
    for i, img in enumerate(images):
        gray = _to_array(img, mode="L")
        brightness = gray.mean()
        scores.append((i, brightness))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def contrast(
    images: list[ImageInput], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by contrast (standard deviation of pixel intensities).

    Higher contrast means more variation between light and dark regions.
    Lower contrast means flatter, more uniform images.

    Useful for adversarial splits: train on normal contrast images,
    test on very flat or very high contrast images.

    Args:
        images: list of file paths or PIL Image objects
        low_first: if True, low contrast (flat) images first;
                   if False, high contrast images first

    Returns:
        List of (index, contrast) tuples sorted by contrast.
    """
    scores = []
    for i, img in enumerate(images):
        gray = _to_array(img, mode="L")
        std = gray.std()
        scores.append((i, std))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def color_variance(
    images: list[ImageInput], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by color variance (spread across RGB channels).

    Measures how much the RGB channels differ from each other.
    Higher variance indicates more colorful/saturated images.
    Lower variance indicates grayscale-like or desaturated images.

    Useful for adversarial splits: train on neutral/desaturated images,
    test on highly colorful/saturated images.

    Args:
        images: list of file paths or PIL Image objects
        low_first: if True, desaturated images first; if False, colorful images first

    Returns:
        List of (index, color_variance) tuples sorted by color variance.
    """
    scores = []
    for i, img in enumerate(images):
        rgb = _to_array(img, mode="RGB")

        # Compute variance across color channels for each pixel, then average
        # This measures how much R, G, B differ from each other
        channel_variance = rgb.var(axis=2).mean()
        scores.append((i, channel_variance))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def dominant_color(
    images: list[ImageInput],
    n_clusters: int = 5,
    target_color: list[int] | None = None,
    low_first: bool = True,
    random_state: int = 42,
    method: str = "quantize",
) -> list[tuple[int, float]]:
    """
    Sort images by their dominant color.

    Uses deterministic color quantization by default to find dominant colors,
    then sorts by distance from a target color (default: neutral gray). Pass
    ``method="kmeans"`` for the legacy k-means estimator.

    Useful for adversarial splits: train on images with one color palette,
    test on images with different dominant colors.

    Args:
        images: list of file paths or PIL Image objects
        n_clusters: number of color clusters to find (default 5)
        target_color: RGB tuple to measure distance from (default [128, 128, 128] gray)
        low_first: if True, images closest to target color first;
                   if False, images furthest from target first
        random_state: seed for the pixel subsample and k-means (default 42)
        method: dominant-color estimator, either "quantize" (deterministic,
            default) or "kmeans" (legacy k-means behavior)

    Returns:
        List of (index, distance) tuples sorted by distance from target color.

    Raises:
        ValueError: if ``method`` is not "quantize" or "kmeans".
    """
    if method not in {"quantize", "kmeans"}:
        raise ValueError(f"method must be 'quantize' or 'kmeans', got {method!r}")

    if target_color is None:
        target_color = np.array([128, 128, 128])
    else:
        target_color = np.array(target_color)

    scores = []
    for i, img in enumerate(images):
        if method == "quantize":
            pil_img = _load_image(img).convert("RGB")
            palette_img = pil_img.quantize(colors=n_clusters)
            color_counts = palette_img.getcolors(maxcolors=pil_img.width * pil_img.height)
            if not color_counts:  # pragma: no cover - quantized images should fit maxcolors
                dominant_rgb = np.array([0.0, 0.0, 0.0])
            else:
                _, palette_index = max(color_counts, key=lambda p: (p[0], -p[1]))
                palette = palette_img.getpalette()
                offset = palette_index * 3
                dominant_rgb = np.array(palette[offset:offset + 3], dtype=np.float32)
        else:
            rgb = _to_array(img, mode="RGB")

            # Reshape to list of pixels
            pixels = rgb.reshape(-1, 3)

            # Sample pixels if image is large (for speed). Use a seeded generator
            # so the subsample and resulting ranking are reproducible across calls.
            if len(pixels) > 10000:
                rng = np.random.default_rng(random_state)
                indices = rng.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]

            # Find dominant colors via k-means
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            kmeans.fit(pixels)

            # Get the most common cluster (dominant color)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[counts.argmax()]
            dominant_rgb = kmeans.cluster_centers_[dominant_idx]

        # Distance from target color
        distance = np.linalg.norm(dominant_rgb - target_color)
        scores.append((i, float(distance)))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def compression_ratio(
    images: list[ImageInput], quality: int = 85, low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by JPEG compression ratio.

    Simple/uniform images compress well (high ratio).
    Complex/detailed images compress poorly (low ratio).

    Useful for adversarial splits: train on simple/compressible images,
    test on complex/incompressible images.

    Args:
        images: list of file paths or PIL Image objects
        quality: JPEG quality setting for compression test (default 85)
        low_first: if True, complex images first (low ratio);
                   if False, simple images first (high ratio)

    Returns:
        List of (index, ratio) tuples sorted by compression ratio.
        Ratio = uncompressed size / compressed size.
    """
    scores = []
    for i, img in enumerate(images):
        pil_img = _load_image(img).convert("RGB")

        # Uncompressed size (width * height * 3 channels)
        uncompressed_size = pil_img.width * pil_img.height * 3

        # Compress to JPEG in memory
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        compressed_size = buffer.tell()

        ratio = uncompressed_size / compressed_size
        scores.append((i, ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def frequency_content(
    images: list[ImageInput], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by high-frequency content (texture/detail level).

    Uses FFT to analyze frequency distribution. Higher values indicate
    more high-frequency content (edges, textures, fine details).
    Lower values indicate smoother, low-frequency images.

    Useful for adversarial splits: train on smooth/simple images,
    test on textured/detailed images.

    Args:
        images: list of file paths or PIL Image objects
        low_first: if True, smooth/low-frequency images first;
                   if False, textured/high-frequency images first

    Returns:
        List of (index, high_freq_ratio) tuples sorted by frequency content.
    """
    scores = []
    for i, img in enumerate(images):
        gray = _to_array(img, mode="L")

        # Compute 2D FFT and shift zero frequency to center
        f_transform = fft.fft2(gray)
        f_shifted = fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)

        # Create a mask for high frequencies (outer region)
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2

        # Define "low frequency" as inner 25% of frequency space
        low_freq_radius = min(rows, cols) // 8

        y, x = np.ogrid[:rows, :cols]
        distance_from_center = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

        low_freq_mask = distance_from_center <= low_freq_radius
        high_freq_mask = ~low_freq_mask

        # Ratio of high frequency energy to total energy
        total_energy = magnitude.sum()
        if total_energy == 0:
            scores.append((i, 0.0))
            continue

        high_freq_energy = magnitude[high_freq_mask].sum()
        high_freq_ratio = high_freq_energy / total_energy
        scores.append((i, high_freq_ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return [(idx, float(value)) for idx, value in scores]


def sharpness(
    images: list[ImageInput], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort images by sharpness — the variance of the Laplacian (a blur detector).

    A blurry image has little high-frequency detail and a low Laplacian
    variance; a sharp, detailed image has a high one. Distinct from
    ``contrast`` / ``frequency_content`` in isolating edge focus.

    Useful for adversarial / curriculum splits: train on sharp images, test on
    blurry ones (or vice versa).

    Args:
        images: list of file paths or PIL Image objects.
        low_first: if True, the blurriest images first.

    Returns:
        List of (index, laplacian_variance) tuples.
    """
    from scipy.ndimage import laplace

    scores = []
    for i, img in enumerate(images):
        gray = _to_array(img, mode="L")
        scores.append((i, float(laplace(gray).var())))
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores
