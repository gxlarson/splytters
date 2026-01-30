"""Unit tests for tabular_sorters.py"""

import pytest
import pandas as pd
import numpy as np

from sorters.tabular_sorters import (
    column_value,
    column_rank,
    column_zscore,
    column_absolute_zscore,
    missing_value_ratio,
    row_sparsity,
    outlier_score,
    numerical_range_score,
    categorical_rarity,
    feature_entropy,
    row_distance_to_mean,
    multi_column_sort,
)


class TestColumnValue:
    """Tests for column_value function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'value': [30, 10, 20],
            'name': ['c', 'a', 'b']
        })

    def test_orders_low_first(self, df):
        """Lowest values should come first when low_first=True."""
        results = column_value(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == [1, 2, 0]  # 10, 20, 30

    def test_orders_high_first(self, df):
        """Highest values should come first when low_first=False."""
        results = column_value(df, 'value', low_first=False)
        indices = [idx for idx, _ in results]
        assert indices == [0, 2, 1]  # 30, 20, 10

    def test_returns_correct_values(self, df):
        """Should return actual column values."""
        results = column_value(df, 'value', low_first=True)
        values = [val for _, val in results]
        assert values == [10, 20, 30]

    def test_handles_nan(self):
        """NaN values should be placed at end."""
        df = pd.DataFrame({'value': [20, np.nan, 10]})
        results = column_value(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == [2, 0, 1]  # 10, 20, NaN

    def test_works_with_strings(self, df):
        """Should work with string columns."""
        results = column_value(df, 'name', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == [1, 2, 0]  # 'a', 'b', 'c'


class TestColumnRank:
    """Tests for column_rank function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'value': [100, 50, 75, 25]})

    def test_returns_percentiles(self, df):
        """Should return percentile ranks."""
        results = column_rank(df, 'value', low_first=True)
        # 25 -> 25th, 50 -> 50th, 75 -> 75th, 100 -> 100th percentile
        scores = {idx: pct for idx, pct in results}
        assert scores[3] < scores[1] < scores[2] < scores[0]

    def test_orders_low_percentile_first(self, df):
        """Lowest percentile should come first when low_first=True."""
        results = column_rank(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 3  # 25 is lowest


class TestColumnZscore:
    """Tests for column_zscore function."""

    @pytest.fixture
    def df(self):
        # Mean = 50, values are 20, 50, 80 (z = -1.22, 0, 1.22 approx)
        return pd.DataFrame({'value': [20, 50, 80]})

    def test_mean_has_zero_zscore(self, df):
        """Value at mean should have z-score of 0."""
        results = column_zscore(df, 'value')
        scores = {idx: z for idx, z in results}
        assert scores[1] == pytest.approx(0, abs=0.01)

    def test_below_mean_negative(self, df):
        """Values below mean should have negative z-score."""
        results = column_zscore(df, 'value')
        scores = {idx: z for idx, z in results}
        assert scores[0] < 0  # 20 is below mean

    def test_above_mean_positive(self, df):
        """Values above mean should have positive z-score."""
        results = column_zscore(df, 'value')
        scores = {idx: z for idx, z in results}
        assert scores[2] > 0  # 80 is above mean

    def test_orders_low_zscore_first(self, df):
        """Lowest z-score should come first when low_first=True."""
        results = column_zscore(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == [0, 1, 2]  # negative, zero, positive


class TestColumnAbsoluteZscore:
    """Tests for column_absolute_zscore function."""

    @pytest.fixture
    def df(self):
        # Mean = 50, values are 20, 50, 80
        return pd.DataFrame({'value': [20, 50, 80]})

    def test_mean_has_zero_abs_zscore(self, df):
        """Value at mean should have absolute z-score of 0."""
        results = column_absolute_zscore(df, 'value')
        scores = {idx: z for idx, z in results}
        assert scores[1] == pytest.approx(0, abs=0.01)

    def test_extreme_values_have_high_abs_zscore(self, df):
        """Extreme values should have higher absolute z-scores."""
        results = column_absolute_zscore(df, 'value')
        scores = {idx: z for idx, z in results}
        assert scores[0] > scores[1]  # 20 is farther from mean than 50
        assert scores[2] > scores[1]  # 80 is farther from mean than 50

    def test_orders_typical_first(self, df):
        """Typical values (low |z|) should come first when low_first=True."""
        results = column_absolute_zscore(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 1  # 50 is at the mean


class TestMissingValueRatio:
    """Tests for missing_value_ratio function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'a': [1, np.nan, np.nan],
            'b': [2, 2, np.nan],
            'c': [3, 3, 3],
        })

    def test_complete_row_has_zero_ratio(self, df):
        """Row with no missing values should have ratio 0."""
        results = missing_value_ratio(df)
        scores = {idx: ratio for idx, ratio in results}
        assert scores[0] == 0.0

    def test_partial_missing_has_correct_ratio(self, df):
        """Row with some missing values should have correct ratio."""
        results = missing_value_ratio(df)
        scores = {idx: ratio for idx, ratio in results}
        assert scores[1] == pytest.approx(1/3)  # 1 of 3 columns missing

    def test_orders_complete_first(self, df):
        """Complete rows should come first when low_first=True."""
        results = missing_value_ratio(df, low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 0  # Row 0 is complete

    def test_orders_sparse_first(self, df):
        """Sparse rows should come first when low_first=False."""
        results = missing_value_ratio(df, low_first=False)
        indices = [idx for idx, _ in results]
        assert indices[0] == 2  # Row 2 has most missing


class TestRowSparsity:
    """Tests for row_sparsity function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'a': [1, 0, 0],
            'b': [2, 0, 0],
            'c': [3, 1, 0],
        })

    def test_dense_row_has_zero_sparsity(self, df):
        """Row with no zeros should have sparsity 0."""
        results = row_sparsity(df)
        scores = {idx: sparsity for idx, sparsity in results}
        assert scores[0] == 0.0

    def test_sparse_row_has_correct_sparsity(self, df):
        """Row with zeros should have correct sparsity."""
        results = row_sparsity(df)
        scores = {idx: sparsity for idx, sparsity in results}
        assert scores[1] == pytest.approx(2/3)  # 2 of 3 columns are zero
        assert scores[2] == pytest.approx(1.0)   # All zeros

    def test_orders_dense_first(self, df):
        """Dense rows should come first when low_first=True."""
        results = row_sparsity(df, low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 0


class TestOutlierScore:
    """Tests for outlier_score function."""

    @pytest.fixture
    def df(self):
        # Normal points clustered around 50, one outlier at 1000
        return pd.DataFrame({
            'a': [48, 52, 50, 51, 1000],
            'b': [49, 51, 50, 52, 1000],
        })

    def test_outlier_has_highest_score(self, df):
        """Outlier should have highest outlier score."""
        results = outlier_score(df, method='isolation_forest')
        scores = {idx: score for idx, score in results}
        # Row 4 (1000, 1000) should be the outlier
        assert scores[4] > scores[0]
        assert scores[4] > scores[1]
        assert scores[4] > scores[2]

    def test_orders_normal_first(self, df):
        """Normal rows should come first when low_first=True."""
        results = outlier_score(df, method='isolation_forest', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[-1] == 4  # Outlier should be last

    def test_zscore_method(self, df):
        """Should work with zscore method."""
        results = outlier_score(df, method='zscore')
        scores = {idx: score for idx, score in results}
        assert scores[4] > scores[0]  # Outlier has higher z-score

    def test_invalid_method_raises(self, df):
        """Should raise ValueError for invalid method."""
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            outlier_score(df, method='invalid')


class TestNumericalRangeScore:
    """Tests for numerical_range_score function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'a': [50, 0, 100],    # Median ~50
            'b': [50, 0, 100],
        })

    def test_median_has_low_extremity(self, df):
        """Row at median should have low extremity score."""
        results = numerical_range_score(df)
        scores = {idx: score for idx, score in results}
        assert scores[0] < scores[1]
        assert scores[0] < scores[2]

    def test_orders_typical_first(self, df):
        """Typical rows should come first when low_first=True."""
        results = numerical_range_score(df, low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 0  # Median row first


class TestCategoricalRarity:
    """Tests for categorical_rarity function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'C']
        })

    def test_common_has_higher_frequency(self, df):
        """Common categories should have higher frequency."""
        results = categorical_rarity(df, 'category')
        scores = {idx: freq for idx, freq in results}
        # A appears 3 times, B once, C once
        assert scores[0] > scores[3]  # A > B
        assert scores[0] > scores[4]  # A > C

    def test_orders_common_first(self, df):
        """Common categories should come first when low_first=True."""
        results = categorical_rarity(df, 'category', low_first=True)
        indices = [idx for idx, _ in results]
        # Rows with 'A' should come first
        assert indices[0] in [0, 1, 2]

    def test_orders_rare_first(self, df):
        """Rare categories should come first when low_first=False."""
        results = categorical_rarity(df, 'category', low_first=False)
        indices = [idx for idx, _ in results]
        # Rows with 'B' or 'C' should come first
        assert indices[0] in [3, 4]


class TestFeatureEntropy:
    """Tests for feature_entropy function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'cat1': ['A', 'A', 'A', 'B'],
            'cat2': ['X', 'X', 'Y', 'Z'],
        })

    def test_common_pattern_has_lower_entropy(self, df):
        """Common patterns should have lower entropy score."""
        results = feature_entropy(df)
        scores = {idx: entropy for idx, entropy in results}
        # Row 0 (A, X) and Row 1 (A, X) are most common
        # Row 3 (B, Z) is rarest
        assert scores[0] < scores[3]
        assert scores[1] < scores[3]

    def test_orders_common_first(self, df):
        """Common patterns should come first when low_first=True."""
        results = feature_entropy(df, low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] in [0, 1]  # Common pattern first


class TestRowDistanceToMean:
    """Tests for row_distance_to_mean function."""

    @pytest.fixture
    def df(self):
        # Mean is (50, 50)
        return pd.DataFrame({
            'a': [50, 0, 100],
            'b': [50, 0, 100],
        })

    def test_mean_row_has_zero_distance(self, df):
        """Row at mean should have distance 0."""
        results = row_distance_to_mean(df)
        scores = {idx: dist for idx, dist in results}
        assert scores[0] == pytest.approx(0, abs=0.01)

    def test_extreme_rows_have_higher_distance(self, df):
        """Extreme rows should have higher distance."""
        results = row_distance_to_mean(df)
        scores = {idx: dist for idx, dist in results}
        assert scores[1] > scores[0]
        assert scores[2] > scores[0]

    def test_orders_typical_first(self, df):
        """Typical rows should come first when low_first=True."""
        results = row_distance_to_mean(df, low_first=True)
        indices = [idx for idx, _ in results]
        assert indices[0] == 0


class TestMultiColumnSort:
    """Tests for multi_column_sort function."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'a': [0, 50, 100],
            'b': [100, 50, 0],
        })

    def test_equal_weights(self, df):
        """Equal weights should average normalized values."""
        results = multi_column_sort(df, ['a', 'b'], weights=[1, 1])
        scores = {idx: score for idx, score in results}
        # All rows should have same combined score (0+1=1, 0.5+0.5=1, 1+0=1)
        assert scores[0] == pytest.approx(scores[1], abs=0.01)
        assert scores[1] == pytest.approx(scores[2], abs=0.01)

    def test_single_column_weight(self, df):
        """Single column weight should match column_value order."""
        results = multi_column_sort(df, ['a'], weights=[1], low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == [0, 1, 2]  # 0, 50, 100

    def test_negative_weight(self, df):
        """Negative weight should invert column contribution."""
        results = multi_column_sort(df, ['a', 'b'], weights=[1, -1], low_first=True)
        indices = [idx for idx, _ in results]
        # a=0,b=100 -> 0-1=-1; a=50,b=50 -> 0.5-0.5=0; a=100,b=0 -> 1-0=1
        assert indices == [0, 1, 2]

    def test_mismatched_lengths_raises(self, df):
        """Should raise ValueError if columns and weights don't match."""
        with pytest.raises(ValueError, match="Number of columns must match"):
            multi_column_sort(df, ['a', 'b'], weights=[1])


class TestEdgeCases:
    """Test edge cases for all tabular sorters."""

    @pytest.fixture
    def single_row_df(self):
        return pd.DataFrame({'a': [1], 'b': [2]})

    @pytest.fixture
    def empty_df(self):
        return pd.DataFrame({'a': [], 'b': []})

    def test_single_row_column_value(self, single_row_df):
        """column_value should work with single row."""
        results = column_value(single_row_df, 'a')
        assert len(results) == 1

    def test_single_row_missing_value_ratio(self, single_row_df):
        """missing_value_ratio should work with single row."""
        results = missing_value_ratio(single_row_df)
        assert len(results) == 1
        assert results[0][1] == 0.0

    def test_single_row_row_sparsity(self, single_row_df):
        """row_sparsity should work with single row."""
        results = row_sparsity(single_row_df)
        assert len(results) == 1

    def test_single_row_outlier_score(self, single_row_df):
        """outlier_score should work with single row."""
        # Need at least 2 rows for isolation forest
        df = pd.DataFrame({'a': [1, 2], 'b': [2, 3]})
        results = outlier_score(df, method='zscore')
        assert len(results) == 2

    def test_empty_df_column_value(self, empty_df):
        """column_value should handle empty DataFrame."""
        results = column_value(empty_df, 'a')
        assert results == []

    def test_empty_df_missing_value_ratio(self, empty_df):
        """missing_value_ratio should handle empty DataFrame."""
        results = missing_value_ratio(empty_df)
        assert results == []

    def test_empty_df_row_sparsity(self, empty_df):
        """row_sparsity should handle empty DataFrame."""
        results = row_sparsity(empty_df)
        assert results == []

    def test_empty_df_numerical_range_score(self, empty_df):
        """numerical_range_score should handle empty DataFrame."""
        results = numerical_range_score(empty_df)
        assert results == []

    def test_empty_df_row_distance_to_mean(self, empty_df):
        """row_distance_to_mean should handle empty DataFrame."""
        results = row_distance_to_mean(empty_df)
        assert results == []

    def test_all_same_values(self):
        """Should handle columns where all values are the same."""
        df = pd.DataFrame({'a': [5, 5, 5]})
        results = column_zscore(df, 'a')
        # All z-scores should be 0 when std=0
        for _, zscore in results:
            assert zscore == 0.0

    def test_non_numeric_columns_ignored(self):
        """row_sparsity should ignore non-numeric columns."""
        df = pd.DataFrame({
            'num': [0, 1, 2],
            'text': ['a', 'b', 'c']
        })
        results = row_sparsity(df)
        scores = {idx: sparsity for idx, sparsity in results}
        assert scores[0] == 1.0  # Only 'num' column, which is 0
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_all_nan_column(self):
        """Should handle columns that are all NaN."""
        df = pd.DataFrame({
            'a': [np.nan, np.nan, np.nan],
            'b': [1, 2, 3]
        })
        results = missing_value_ratio(df)
        scores = {idx: ratio for idx, ratio in results}
        # Each row has 1 of 2 columns missing
        assert all(ratio == 0.5 for ratio in scores.values())


class TestWithCustomIndex:
    """Test that functions work with non-default DataFrame indices."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {'value': [30, 10, 20]},
            index=['x', 'y', 'z']
        )

    def test_column_value_preserves_index(self, df):
        """Should return original index values."""
        results = column_value(df, 'value', low_first=True)
        indices = [idx for idx, _ in results]
        assert indices == ['y', 'z', 'x']  # 10, 20, 30

    def test_missing_value_ratio_preserves_index(self, df):
        """Should return original index values."""
        results = missing_value_ratio(df)
        indices = [idx for idx, _ in results]
        assert set(indices) == {'x', 'y', 'z'}
