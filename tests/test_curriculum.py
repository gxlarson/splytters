"""Tests for the ordering-driven curriculum split."""

import numpy as np
import pytest

from splytters import sorted_stratified_split


def test_per_class_first_fraction_goes_to_train():
    # Two classes, interleaved; order is just 0..9 ascending.
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    order = list(range(10))
    train, test = sorted_stratified_split(order, y, train_size=0.6)

    # 60% of each class (3 of 5) -> train, taken in order.
    assert sorted(train.tolist()) == [0, 1, 2, 5, 6, 7]
    assert sorted(test.tolist()) == [3, 4, 8, 9]


def test_partition_is_complete_and_disjoint():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=60)
    order = rng.permutation(60)
    train, test = sorted_stratified_split(order, y, train_size=0.7)

    assert set(train.tolist()) | set(test.tolist()) == set(range(60))
    assert not (set(train.tolist()) & set(test.tolist()))


def test_respects_the_given_order_within_class():
    y = np.array([0, 0, 0, 0])
    # Reverse order: highest index first.
    order = [3, 2, 1, 0]
    train, test = sorted_stratified_split(order, y, train_size=0.5)
    # First half of the order for class 0 is [3, 2].
    assert sorted(train.tolist()) == [2, 3]
    assert sorted(test.tolist()) == [0, 1]


def test_largest_first_reverses_the_order():
    y = np.array([0, 0, 0, 0])
    order = [0, 1, 2, 3]
    train, _ = sorted_stratified_split(order, y, train_size=0.5, largest_first=True)
    assert sorted(train.tolist()) == [2, 3]


def test_accepts_sorter_style_index_score_pairs():
    y = np.array([0, 0, 1, 1])
    # As returned by a sorter: (index, score) ascending by score.
    ranking = [(2, 0.1), (0, 0.2), (3, 0.5), (1, 0.9)]
    train, test = sorted_stratified_split(ranking, y, train_size=0.5)
    # Class 0 order: [0, 1] -> train [0]; class 1 order: [2, 3] -> train [2].
    assert sorted(train.tolist()) == [0, 2]
    assert sorted(test.tolist()) == [1, 3]


def test_absolute_count_is_per_class():
    y = np.array([0, 0, 0, 1, 1, 1])
    order = list(range(6))
    train, _ = sorted_stratified_split(order, y, train_size=2)
    # 2 per class.
    assert sorted(train.tolist()) == [0, 1, 3, 4]


@pytest.mark.parametrize(
    "order, y, train_size",
    [
        ([0, 1, 2], [0, 1], 0.5),          # length mismatch
        ([0, 1, 1], [0, 0, 1], 0.5),       # not a permutation (dup)
        ([0, 1, 2], [0, 0, 1], 1.5),       # fraction out of range
    ],
)
def test_invalid_inputs_raise(order, y, train_size):
    with pytest.raises(ValueError):
        sorted_stratified_split(order, y, train_size=train_size)


def test_empty_input_returns_two_empty_arrays():
    train, test = sorted_stratified_split([], [], train_size=0.7)
    assert train.tolist() == []
    assert test.tolist() == []
    assert train.dtype == np.intp and test.dtype == np.intp


def test_class_too_small_for_fraction_goes_entirely_to_test():
    # A singleton class at train_size=0.7 resolves to 0 train (floor) -> all test;
    # a larger class still splits normally. Pins this intentional behaviour.
    y = np.array([0, 1, 1, 1, 1])  # class 0 has a single sample
    order = list(range(5))
    train, test = sorted_stratified_split(order, y, train_size=0.7)

    assert 0 not in train.tolist()          # singleton class -> no train sample
    assert 0 in test.tolist()
    assert set(train.tolist()) == {1, 2}     # class 1: first int(4*0.7)=2 of 4
    assert set(train.tolist()) | set(test.tolist()) == set(range(5))
