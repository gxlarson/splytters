"""Unit tests for image_sorters.py"""

import shutil
from pathlib import Path

import pytest

from splytters.sorters.image_sorters import (
    _load_image,
    color_variance,
    compression_ratio,
    contrast,
    dominant_color,
    frequency_content,
    mean_brightness,
)

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_data" / "images"


def get_image_path(name):
    """Get full path to a test image."""
    return TEST_IMAGES_DIR / f"{name}.png"


def get_sorted_names(results, images):
    """Extract image names from sorted results."""
    return [images[idx].stem for idx, *_ in results]


def find_index(results, images, name):
    """Find the position of an image in sorted results."""
    names = get_sorted_names(results, images)
    return names.index(name)


class TestMeanBrightness:
    """Tests for mean_brightness function."""

    @pytest.fixture
    def brightness_images(self):
        return [
            get_image_path("brightness_dark"),
            get_image_path("brightness_medium"),
            get_image_path("brightness_bright"),
        ]

    def test_orders_dark_before_bright(self, brightness_images):
        """Dark images should come before bright when low_first=True."""
        results = mean_brightness(brightness_images, low_first=True)
        names = get_sorted_names(results, brightness_images)

        assert names.index("brightness_dark") < names.index("brightness_medium")
        assert names.index("brightness_medium") < names.index("brightness_bright")

    def test_orders_bright_before_dark(self, brightness_images):
        """Bright images should come before dark when low_first=False."""
        results = mean_brightness(brightness_images, low_first=False)
        names = get_sorted_names(results, brightness_images)

        assert names.index("brightness_bright") < names.index("brightness_medium")
        assert names.index("brightness_medium") < names.index("brightness_dark")

    def test_returns_correct_structure(self, brightness_images):
        """Results should be list of (index, brightness) tuples."""
        results = mean_brightness(brightness_images)

        assert len(results) == 3
        for idx, brightness in results:
            assert isinstance(idx, int)
            assert isinstance(brightness, float)
            assert 0 <= brightness <= 255


class TestLoadImage:
    """Regression tests for _load_image file-handle management."""

    def test_closes_file_handle(self, tmp_path):
        """_load_image force-loads pixels so the underlying file is closed; the
        source file must be deletable afterward (fails on Windows if a handle
        leaks). Guards against Image.open's lazy fd staying open."""
        src = get_image_path("brightness_medium")
        copied = tmp_path / "copy.png"
        shutil.copy(src, copied)

        img = _load_image(copied)
        assert img.size[0] > 0  # pixels are accessible after the handle closed
        copied.unlink()  # would raise PermissionError on Windows if fd still open
        assert not copied.exists()


class TestContrast:
    """Tests for contrast function."""

    @pytest.fixture
    def contrast_images(self):
        return [
            get_image_path("contrast_low"),
            get_image_path("contrast_high"),
        ]

    def test_orders_low_before_high(self, contrast_images):
        """Low contrast images should come first when low_first=True."""
        results = contrast(contrast_images, low_first=True)
        names = get_sorted_names(results, contrast_images)

        assert names.index("contrast_low") < names.index("contrast_high")

    def test_orders_high_before_low(self, contrast_images):
        """High contrast images should come first when low_first=False."""
        results = contrast(contrast_images, low_first=False)
        names = get_sorted_names(results, contrast_images)

        assert names.index("contrast_high") < names.index("contrast_low")

    def test_high_contrast_has_higher_value(self, contrast_images):
        """High contrast image should have higher std dev value."""
        results = contrast(contrast_images, low_first=True)

        # Get scores by image name
        scores = {contrast_images[idx].stem: score for idx, score in results}

        assert scores["contrast_high"] > scores["contrast_low"]


class TestColorVariance:
    """Tests for color_variance function."""

    @pytest.fixture
    def color_images(self):
        return [
            get_image_path("color_grayscale"),
            get_image_path("color_vibrant"),
        ]

    def test_orders_grayscale_before_vibrant(self, color_images):
        """Grayscale-like images should come first when low_first=True."""
        results = color_variance(color_images, low_first=True)
        names = get_sorted_names(results, color_images)

        assert names.index("color_grayscale") < names.index("color_vibrant")

    def test_orders_vibrant_before_grayscale(self, color_images):
        """Vibrant images should come first when low_first=False."""
        results = color_variance(color_images, low_first=False)
        names = get_sorted_names(results, color_images)

        assert names.index("color_vibrant") < names.index("color_grayscale")


class TestDominantColor:
    """Tests for dominant_color function."""

    @pytest.fixture
    def dominant_images(self):
        return [
            get_image_path("dominant_red"),
            get_image_path("dominant_green"),
            get_image_path("dominant_blue"),
        ]

    def test_red_closest_to_red_target(self, dominant_images):
        """Red image should be closest to red target color."""
        results = dominant_color(
            dominant_images, target_color=[255, 0, 0], low_first=True
        )
        names = get_sorted_names(results, dominant_images)

        assert names[0] == "dominant_red"

    def test_green_closest_to_green_target(self, dominant_images):
        """Green image should be closest to green target color."""
        results = dominant_color(
            dominant_images, target_color=[0, 255, 0], low_first=True
        )
        names = get_sorted_names(results, dominant_images)

        assert names[0] == "dominant_green"

    def test_blue_closest_to_blue_target(self, dominant_images):
        """Blue image should be closest to blue target color."""
        results = dominant_color(
            dominant_images, target_color=[0, 0, 255], low_first=True
        )
        names = get_sorted_names(results, dominant_images)

        assert names[0] == "dominant_blue"

    def test_returns_index_distance_tuples(self, dominant_images):
        """Results should be (index, distance) tuples."""
        results = dominant_color(dominant_images)

        for idx, distance in results:
            assert isinstance(idx, int)
            assert isinstance(distance, float)
            assert distance >= 0

    def test_deterministic_on_large_image(self):
        """Images over 10k pixels subsample their pixels; the ranking must be
        reproducible across calls. Guards the old unseeded np.random.choice
        subsample (KMeans contributes only ~1e-6 float jitter, which does not
        reorder distinct images)."""
        import numpy as np
        from PIL import Image

        rng = np.random.RandomState(0)
        # 130*130 = 16900 pixels (> 10k) triggers the subsampling path.
        imgs = [
            Image.fromarray(rng.randint(0, 256, size=(130, 130, 3), dtype=np.uint8))
            for _ in range(4)
        ]
        r1 = dominant_color(imgs)
        r2 = dominant_color(imgs)
        # Ranking identical, distances stable to within KMeans float jitter.
        assert [i for i, _ in r1] == [i for i, _ in r2]
        np.testing.assert_allclose(
            [d for _, d in r1], [d for _, d in r2], rtol=1e-3
        )


class TestCompressionRatio:
    """Tests for compression_ratio function."""

    @pytest.fixture
    def compress_images(self):
        return [
            get_image_path("compress_simple"),
            get_image_path("compress_complex"),
        ]

    def test_simple_compresses_better(self, compress_images):
        """Simple images should have higher compression ratio."""
        results = compression_ratio(compress_images, low_first=False)

        # Get scores by image name
        scores = {compress_images[idx].stem: ratio for idx, ratio in results}

        assert scores["compress_simple"] > scores["compress_complex"]

    def test_orders_simple_first_when_high_first(self, compress_images):
        """Simple (high ratio) images first when low_first=False."""
        results = compression_ratio(compress_images, low_first=False)
        names = get_sorted_names(results, compress_images)

        assert names[0] == "compress_simple"

    def test_orders_complex_first_when_low_first(self, compress_images):
        """Complex (low ratio) images first when low_first=True."""
        results = compression_ratio(compress_images, low_first=True)
        names = get_sorted_names(results, compress_images)

        assert names[0] == "compress_complex"


class TestFrequencyContent:
    """Tests for frequency_content function."""

    @pytest.fixture
    def freq_images(self):
        return [
            get_image_path("freq_smooth"),
            get_image_path("freq_medium"),
            get_image_path("freq_detailed"),
        ]

    def test_orders_smooth_before_detailed(self, freq_images):
        """Smooth images should come before detailed when low_first=True."""
        results = frequency_content(freq_images, low_first=True)
        names = get_sorted_names(results, freq_images)

        assert names.index("freq_smooth") < names.index("freq_detailed")

    def test_orders_detailed_before_smooth(self, freq_images):
        """Detailed images should come before smooth when low_first=False."""
        results = frequency_content(freq_images, low_first=False)
        names = get_sorted_names(results, freq_images)

        assert names.index("freq_detailed") < names.index("freq_smooth")

    def test_frequency_values_increase(self, freq_images):
        """Frequency content should increase: smooth < medium < detailed."""
        results = frequency_content(freq_images, low_first=True)

        # Get scores by image name
        scores = {freq_images[idx].stem: ratio for idx, ratio in results}

        assert scores["freq_smooth"] < scores["freq_medium"]
        assert scores["freq_medium"] < scores["freq_detailed"]


class TestWithPILImages:
    """Test that functions work with PIL Image objects, not just paths."""

    def test_mean_brightness_with_pil(self):
        """mean_brightness should accept PIL Image objects."""
        from PIL import Image

        images = [
            Image.open(get_image_path("brightness_dark")),
            Image.open(get_image_path("brightness_bright")),
        ]

        results = mean_brightness(images)
        assert len(results) == 2

    def test_contrast_with_pil(self):
        """contrast should accept PIL Image objects."""
        from PIL import Image

        images = [
            Image.open(get_image_path("contrast_low")),
            Image.open(get_image_path("contrast_high")),
        ]

        results = contrast(images)
        assert len(results) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def single_image(self):
        return [get_image_path("brightness_medium")]

    def test_single_image_mean_brightness(self, single_image):
        """mean_brightness should work with a single image."""
        results = mean_brightness(single_image)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_single_image_contrast(self, single_image):
        """contrast should work with a single image."""
        results = contrast(single_image)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_single_image_color_variance(self, single_image):
        """color_variance should work with a single image."""
        results = color_variance(single_image)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_single_image_dominant_color(self, single_image):
        """dominant_color should work with a single image."""
        results = dominant_color(single_image)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] >= 0  # distance should be non-negative

    def test_single_image_compression_ratio(self, single_image):
        """compression_ratio should work with a single image."""
        results = compression_ratio(single_image)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0  # Ratio should be positive

    def test_single_image_frequency_content(self, single_image):
        """frequency_content should work with a single image."""
        results = frequency_content(single_image)
        assert len(results) == 1
        assert results[0][0] == 0
        assert 0 <= results[0][1] <= 1  # Ratio between 0 and 1

    def test_empty_list_mean_brightness(self):
        """mean_brightness should handle empty image list."""
        results = mean_brightness([])
        assert results == []

    def test_empty_list_contrast(self):
        """contrast should handle empty image list."""
        results = contrast([])
        assert results == []

    def test_empty_list_color_variance(self):
        """color_variance should handle empty image list."""
        results = color_variance([])
        assert results == []

    def test_empty_list_dominant_color(self):
        """dominant_color should handle empty image list."""
        results = dominant_color([])
        assert results == []

    def test_empty_list_compression_ratio(self):
        """compression_ratio should handle empty image list."""
        results = compression_ratio([])
        assert results == []

    def test_empty_list_frequency_content(self):
        """frequency_content should handle empty image list."""
        results = frequency_content([])
        assert results == []
