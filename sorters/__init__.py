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
"""

from sorters.embedding_sorters import (
    dist_euclidean,
    distance_to_mean,
    distance_to_nearest_neighbor,
    local_density,
    outlier_score,
)

from sorters.text_sorters import (
    simple_tokenizer,
    character_length,
    tokens_length,
    sentence_count,
    lexical_diversity,
    vocabulary_rarity,
    perplexity_score,
    readability_score,
)

from sorters.image_sorters import (
    mean_brightness,
    contrast,
    color_variance,
    dominant_color,
    compression_ratio as image_compression_ratio,
    frequency_content,
)

from sorters.audio_sorters import (
    # Loudness / Energy
    mean_amplitude,
    rms_energy,
    dynamic_range,
    peak_to_average_ratio,
    # Frequency / Spectral
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_flatness,
    zero_crossing_rate,
    fundamental_frequency,
    # Timbre / MFCCs
    mfcc_mean,
    mfcc_variance,
    # Rhythm / Music
    tempo,
    beat_strength,
    harmonic_ratio,
    # Quality
    compression_ratio as audio_compression_ratio,
)

from sorters.tabular_sorters import (
    # Generic column sorting
    column_value,
    column_rank,
    column_zscore,
    column_absolute_zscore,
    # Row-level metrics
    missing_value_ratio,
    row_sparsity,
    outlier_score as tabular_outlier_score,
    numerical_range_score,
    row_distance_to_mean as tabular_distance_to_mean,
    # Categorical
    categorical_rarity,
    feature_entropy,
    # Multi-column
    multi_column_sort,
)

__all__ = [
    # Embedding sorters
    "dist_euclidean",
    "distance_to_mean",
    "distance_to_nearest_neighbor",
    "local_density",
    "outlier_score",
    # Text sorters
    "simple_tokenizer",
    "character_length",
    "tokens_length",
    "sentence_count",
    "lexical_diversity",
    "vocabulary_rarity",
    "perplexity_score",
    "readability_score",
    # Image sorters
    "mean_brightness",
    "contrast",
    "color_variance",
    "dominant_color",
    "image_compression_ratio",
    "frequency_content",
    # Audio sorters - Loudness / Energy
    "mean_amplitude",
    "rms_energy",
    "dynamic_range",
    "peak_to_average_ratio",
    # Audio sorters - Frequency / Spectral
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "zero_crossing_rate",
    "fundamental_frequency",
    # Audio sorters - Timbre / MFCCs
    "mfcc_mean",
    "mfcc_variance",
    # Audio sorters - Rhythm / Music
    "tempo",
    "beat_strength",
    "harmonic_ratio",
    # Audio sorters - Quality
    "audio_compression_ratio",
    # Tabular sorters - Column sorting
    "column_value",
    "column_rank",
    "column_zscore",
    "column_absolute_zscore",
    # Tabular sorters - Row metrics
    "missing_value_ratio",
    "row_sparsity",
    "tabular_outlier_score",
    "numerical_range_score",
    "tabular_distance_to_mean",
    # Tabular sorters - Categorical
    "categorical_rarity",
    "feature_entropy",
    # Tabular sorters - Multi-column
    "multi_column_sort",
]
