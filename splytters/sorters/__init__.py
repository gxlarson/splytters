"""
Sorting algorithms for adversarial dataset partitioning.

This package provides functions to rank and sort samples by various criteria
to enable train-test splits that maximize dissimilarity.

Modules:
    embedding_sorters: Embedding-based sorting (distance, density, outliers)
    text_sorters: Text-based sorting (length, readability, perplexity)
    image_sorters: Image-based sorting (brightness, contrast, frequency)
    audio_sorters: Audio-based sorting (loudness, spectral, rhythm, timbre)
    tabular_sorters: Tabular data sorting (columns, missing values, outliers)

Imports are **lazy** per modality: ``import splytters.sorters`` pulls in no
optional dependencies, and accessing e.g. ``splytters.sorters.mean_brightness``
only imports Pillow (the ``[image]`` extra). This means each extra is
self-sufficient — you can install ``splytters[image]`` and use the image sorters
without torch, librosa, or pandas.
"""

from __future__ import annotations

import importlib
from typing import Any

# Public name -> (submodule, attribute-in-submodule).
# Aliases (e.g. image vs audio ``compression_ratio``) are resolved here.
_LAZY: dict[str, tuple[str, str]] = {
    # Embedding sorters (deps: numpy, scikit-learn — core only)
    "dist_euclidean": ("embedding_sorters", "dist_euclidean"),
    "distance_to_mean": ("embedding_sorters", "distance_to_mean"),
    "distance_to_nearest_neighbor": ("embedding_sorters", "distance_to_nearest_neighbor"),
    "local_density": ("embedding_sorters", "local_density"),
    "outlier_score": ("embedding_sorters", "outlier_score"),
    # Text sorters (deps: [text])
    "simple_tokenizer": ("text_sorters", "simple_tokenizer"),
    "character_length": ("text_sorters", "character_length"),
    "tokens_length": ("text_sorters", "tokens_length"),
    "sentence_count": ("text_sorters", "sentence_count"),
    "lexical_diversity": ("text_sorters", "lexical_diversity"),
    "vocabulary_rarity": ("text_sorters", "vocabulary_rarity"),
    "perplexity_score": ("text_sorters", "perplexity_score"),
    "readability_score": ("text_sorters", "readability_score"),
    # Image sorters (deps: [image])
    "mean_brightness": ("image_sorters", "mean_brightness"),
    "contrast": ("image_sorters", "contrast"),
    "color_variance": ("image_sorters", "color_variance"),
    "dominant_color": ("image_sorters", "dominant_color"),
    "image_compression_ratio": ("image_sorters", "compression_ratio"),
    "frequency_content": ("image_sorters", "frequency_content"),
    # Audio sorters (deps: [audio])
    "mean_amplitude": ("audio_sorters", "mean_amplitude"),
    "rms_energy": ("audio_sorters", "rms_energy"),
    "dynamic_range": ("audio_sorters", "dynamic_range"),
    "peak_to_average_ratio": ("audio_sorters", "peak_to_average_ratio"),
    "spectral_centroid": ("audio_sorters", "spectral_centroid"),
    "spectral_bandwidth": ("audio_sorters", "spectral_bandwidth"),
    "spectral_rolloff": ("audio_sorters", "spectral_rolloff"),
    "spectral_flatness": ("audio_sorters", "spectral_flatness"),
    "zero_crossing_rate": ("audio_sorters", "zero_crossing_rate"),
    "fundamental_frequency": ("audio_sorters", "fundamental_frequency"),
    "mfcc_mean": ("audio_sorters", "mfcc_mean"),
    "mfcc_variance": ("audio_sorters", "mfcc_variance"),
    "tempo": ("audio_sorters", "tempo"),
    "beat_strength": ("audio_sorters", "beat_strength"),
    "harmonic_ratio": ("audio_sorters", "harmonic_ratio"),
    "audio_compression_ratio": ("audio_sorters", "compression_ratio"),
    # Tabular sorters (deps: [tabular])
    "column_value": ("tabular_sorters", "column_value"),
    "column_rank": ("tabular_sorters", "column_rank"),
    "column_zscore": ("tabular_sorters", "column_zscore"),
    "column_absolute_zscore": ("tabular_sorters", "column_absolute_zscore"),
    "missing_value_ratio": ("tabular_sorters", "missing_value_ratio"),
    "row_sparsity": ("tabular_sorters", "row_sparsity"),
    "tabular_outlier_score": ("tabular_sorters", "outlier_score"),
    "numerical_range_score": ("tabular_sorters", "numerical_range_score"),
    "tabular_distance_to_mean": ("tabular_sorters", "row_distance_to_mean"),
    "categorical_rarity": ("tabular_sorters", "categorical_rarity"),
    "feature_entropy": ("tabular_sorters", "feature_entropy"),
    "multi_column_sort": ("tabular_sorters", "multi_column_sort"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str) -> Any:
    """Lazily import and cache a sorter on first access (PEP 562)."""
    try:
        module_suffix, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"splytters.sorters.{module_suffix}")
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
