"""
Sorting algorithms for adversarial tabular dataset partitioning.

These functions rank rows in tabular data (pandas DataFrames) by various
criteria (missing values, outliers, sparsity, column values) to enable
train-test splits that maximize dissimilarity.

All functions accept pandas DataFrames and return sorted index lists.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def column_value(df, column, low_first=True):
    """
    Sort rows by values in a specified column.

    Generic sorting function that allows users to sort by any column.

    Args:
        df: pandas DataFrame
        column: column name to sort by
        low_first: if True, lowest values first; if False, highest values first

    Returns:
        List of (index, value) tuples sorted by column value.
        NaN values are placed at the end.
    """
    scores = []
    for idx in df.index:
        value = df.loc[idx, column]
        # Handle NaN by placing at end
        if pd.isna(value):
            sort_value = float('inf') if low_first else float('-inf')
        else:
            sort_value = value
        scores.append((idx, value, sort_value))

    scores.sort(key=lambda p: p[2], reverse=not low_first)
    return [(idx, val) for idx, val, _ in scores]


def column_rank(df, column, low_first=True):
    """
    Sort rows by percentile rank within a column.

    Useful when you want to split by relative position rather than
    absolute values.

    Args:
        df: pandas DataFrame
        column: column name to rank by
        low_first: if True, lowest percentile first; if False, highest first

    Returns:
        List of (index, percentile) tuples sorted by percentile rank (0-100).
    """
    series = df[column]
    ranks = series.rank(pct=True, na_option='bottom') * 100

    scores = [(idx, ranks.loc[idx]) for idx in df.index]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def column_zscore(df, column, low_first=True):
    """
    Sort rows by z-score (standard deviations from mean) in a column.

    Useful for finding outliers in a specific column.

    Args:
        df: pandas DataFrame
        column: column name to compute z-scores for
        low_first: if True, lowest z-scores first (below mean);
                   if False, highest z-scores first (above mean)

    Returns:
        List of (index, zscore) tuples sorted by z-score.
        NaN values receive z-score of infinity.
    """
    series = df[column]
    mean = series.mean()
    std = series.std()

    if std == 0:
        # All values are the same
        return [(idx, 0.0) for idx in df.index]

    scores = []
    for idx in df.index:
        value = series.loc[idx]
        if pd.isna(value):
            zscore = float('inf')
        else:
            zscore = (value - mean) / std
        scores.append((idx, zscore))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def column_absolute_zscore(df, column, low_first=True):
    """
    Sort rows by absolute z-score (distance from mean) in a column.

    Useful for adversarial splits: train on typical values (low |z|),
    test on extreme values (high |z|).

    Args:
        df: pandas DataFrame
        column: column name to compute z-scores for
        low_first: if True, typical values first; if False, extreme values first

    Returns:
        List of (index, abs_zscore) tuples sorted by absolute z-score.
    """
    series = df[column]
    mean = series.mean()
    std = series.std()

    if std == 0:
        return [(idx, 0.0) for idx in df.index]

    scores = []
    for idx in df.index:
        value = series.loc[idx]
        if pd.isna(value):
            abs_zscore = float('inf')
        else:
            abs_zscore = abs((value - mean) / std)
        scores.append((idx, abs_zscore))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def missing_value_ratio(df, low_first=True):
    """
    Sort rows by proportion of missing (NaN/None) values.

    Useful for adversarial splits: train on complete rows,
    test on rows with missing data.

    Args:
        df: pandas DataFrame
        low_first: if True, complete rows first; if False, sparse rows first

    Returns:
        List of (index, missing_ratio) tuples sorted by missing ratio (0-1).
    """
    scores = []
    n_cols = len(df.columns)

    for idx in df.index:
        row = df.loc[idx]
        n_missing = row.isna().sum()
        ratio = n_missing / n_cols if n_cols > 0 else 0
        scores.append((idx, ratio))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def row_sparsity(df, zero_threshold=1e-10, low_first=True):
    """
    Sort rows by proportion of zero (or near-zero) values.

    Useful for adversarial splits: train on dense rows,
    test on sparse rows.

    Args:
        df: pandas DataFrame (numeric columns only)
        zero_threshold: values with abs < threshold are considered zero
        low_first: if True, dense rows first; if False, sparse rows first

    Returns:
        List of (index, sparsity) tuples sorted by sparsity ratio (0-1).
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    n_cols = len(numeric_df.columns)

    if n_cols == 0:
        return [(idx, 0.0) for idx in df.index]

    scores = []
    for idx in df.index:
        row = numeric_df.loc[idx]
        n_zeros = (row.abs() < zero_threshold).sum()
        sparsity = n_zeros / n_cols
        scores.append((idx, sparsity))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def outlier_score(df, method="isolation_forest", low_first=True, **kwargs):
    """
    Sort rows by anomaly/outlier score.

    Uses outlier detection algorithms to score how unusual each row is.

    Useful for adversarial splits: train on normal/typical rows,
    test on outliers/anomalies.

    Args:
        df: pandas DataFrame (numeric columns only, NaN will be filled with median)
        method: outlier detection algorithm, one of:
            - 'isolation_forest': Isolation Forest (fast, good for high dimensions)
            - 'lof': Local Outlier Factor (density-based)
            - 'zscore': Mean absolute z-score across columns
        low_first: if True, normal rows first; if False, outliers first
        **kwargs: additional arguments passed to the outlier detector

    Returns:
        List of (index, outlier_score) tuples sorted by outlier score.
    """
    # Select only numeric columns and fill NaN
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    if len(numeric_df.columns) == 0:
        return [(idx, 0.0) for idx in df.index]

    X = numeric_df.values

    if method == "isolation_forest":
        detector = IsolationForest(random_state=42, **kwargs)
        detector.fit(X)
        # Negate so higher = more outlier
        raw_scores = -detector.score_samples(X)

    elif method == "lof":
        detector = LocalOutlierFactor(novelty=False, **kwargs)
        detector.fit_predict(X)
        # Negate so higher = more outlier
        raw_scores = -detector.negative_outlier_factor_

    elif method == "zscore":
        # Compute mean absolute z-score across all columns
        zscores = np.abs(stats.zscore(X, nan_policy='omit'))
        raw_scores = np.nanmean(zscores, axis=1)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    scores = [(df.index[i], raw_scores[i]) for i in range(len(df))]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def numerical_range_score(df, low_first=True):
    """
    Sort rows by how extreme their numerical values are (distance from median).

    Computes the mean percentile distance from 50th percentile across all
    numeric columns.

    Useful for adversarial splits: train on typical values,
    test on extreme values.

    Args:
        df: pandas DataFrame
        low_first: if True, typical rows first; if False, extreme rows first

    Returns:
        List of (index, extremity) tuples sorted by extremity score (0-50).
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        return [(idx, 0.0) for idx in df.index]

    # Compute percentile ranks for each column
    ranks = numeric_df.rank(pct=True) * 100

    # Distance from 50th percentile (0 = median, 50 = extreme)
    extremity = (ranks - 50).abs().mean(axis=1)

    scores = [(idx, extremity.loc[idx]) for idx in df.index]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def categorical_rarity(df, column, low_first=True):
    """
    Sort rows by rarity of categorical value in a column.

    Useful for adversarial splits: train on common categories,
    test on rare categories.

    Args:
        df: pandas DataFrame
        column: categorical column name
        low_first: if True, common categories first; if False, rare categories first

    Returns:
        List of (index, frequency) tuples sorted by category frequency.
        Higher frequency = more common.
    """
    # Compute value counts as frequencies
    value_counts = df[column].value_counts(normalize=True)

    scores = []
    for idx in df.index:
        value = df.loc[idx, column]
        if pd.isna(value):
            freq = 0.0  # Treat NaN as rarest
        else:
            freq = value_counts.get(value, 0.0)
        scores.append((idx, freq))

    # Note: low_first=True means common first (high frequency first)
    # So we reverse the logic
    scores.sort(key=lambda p: p[1], reverse=low_first)
    return scores


def feature_entropy(df, low_first=True):
    """
    Sort rows by entropy of categorical features.

    Rows with rare combinations of categorical values have higher "entropy"
    in the sense that they're less predictable.

    Approximated by summing -log(freq) for each categorical value.

    Useful for adversarial splits: train on common patterns,
    test on rare patterns.

    Args:
        df: pandas DataFrame
        low_first: if True, common patterns first; if False, rare patterns first

    Returns:
        List of (index, neg_log_freq) tuples sorted by rarity score.
    """
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(cat_cols) == 0:
        return [(idx, 0.0) for idx in df.index]

    # Compute frequencies for each categorical column
    freq_maps = {}
    for col in cat_cols:
        freq_maps[col] = df[col].value_counts(normalize=True).to_dict()

    scores = []
    for idx in df.index:
        neg_log_freq = 0.0
        for col in cat_cols:
            value = df.loc[idx, col]
            if pd.isna(value):
                freq = 1e-10  # Very rare
            else:
                freq = freq_maps[col].get(value, 1e-10)
            neg_log_freq += -np.log(freq + 1e-10)
        scores.append((idx, neg_log_freq))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def row_distance_to_mean(df, low_first=True):
    """
    Sort rows by Euclidean distance from the column-wise mean.

    Similar to embedding distance_to_mean but operates directly on
    tabular features.

    Useful for adversarial splits: train on typical rows,
    test on atypical rows.

    Args:
        df: pandas DataFrame (numeric columns only)
        low_first: if True, typical rows first; if False, atypical rows first

    Returns:
        List of (index, distance) tuples sorted by distance from mean.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    if len(numeric_df.columns) == 0:
        return [(idx, 0.0) for idx in df.index]

    # Compute column-wise mean
    mean_vector = numeric_df.mean().values

    scores = []
    for idx in df.index:
        row_vector = numeric_df.loc[idx].values
        distance = np.linalg.norm(row_vector - mean_vector)
        scores.append((idx, distance))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def multi_column_sort(df, columns, weights=None, low_first=True):
    """
    Sort rows by weighted combination of multiple columns.

    Useful when you want to combine multiple criteria into a single sort.

    Args:
        df: pandas DataFrame
        columns: list of column names to sort by
        weights: list of weights for each column (default: equal weights)
                 Positive weight = higher value increases score
                 Negative weight = higher value decreases score
        low_first: if True, lowest combined score first

    Returns:
        List of (index, combined_score) tuples sorted by combined score.
    """
    if weights is None:
        weights = [1.0] * len(columns)

    if len(columns) != len(weights):
        raise ValueError("Number of columns must match number of weights")

    # Normalize each column to 0-1 range
    normalized = pd.DataFrame(index=df.index)
    for col in columns:
        series = df[col]
        min_val = series.min()
        max_val = series.max()
        if max_val > min_val:
            normalized[col] = (series - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5

    # Compute weighted sum
    scores = []
    for idx in df.index:
        combined = sum(
            weights[i] * normalized.loc[idx, col]
            for i, col in enumerate(columns)
        )
        scores.append((idx, combined))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores
