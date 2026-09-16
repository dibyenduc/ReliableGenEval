"""Unit tests for the paired sign-flip permutation test."""

import numpy as np
import pandas as pd
import pytest

from evalcore.stats.permutation import paired_permutation_test


def test_no_true_difference_gives_high_p_value():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(3.0, 1.0, size=200))
    b = pd.Series(rng.normal(3.0, 1.0, size=200))
    result = paired_permutation_test(a, b, n_permutations=5000, random_state=1)
    assert result.p_value > 0.05


def test_large_true_difference_gives_low_p_value():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(4.5, 0.3, size=200))
    b = pd.Series(rng.normal(2.5, 0.3, size=200))
    result = paired_permutation_test(a, b, n_permutations=5000, random_state=1)
    assert result.p_value < 0.01


def test_reproducible_with_fixed_seed():
    a = pd.Series(np.arange(50, dtype=float))
    b = pd.Series(np.arange(50, dtype=float) * 0.95)
    r1 = paired_permutation_test(a, b, n_permutations=1000, random_state=7)
    r2 = paired_permutation_test(a, b, n_permutations=1000, random_state=7)
    assert r1.p_value == r2.p_value


def test_mismatched_lengths_raise():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="equal length"):
        paired_permutation_test(a, b)


def test_one_sided_greater_alternative():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(4.0, 0.3, size=200))
    b = pd.Series(rng.normal(3.0, 0.3, size=200))
    result_greater = paired_permutation_test(a, b, alternative="greater", n_permutations=3000, random_state=1)
    result_less = paired_permutation_test(a, b, alternative="less", n_permutations=3000, random_state=1)
    assert result_greater.p_value < 0.01
    assert result_less.p_value > 0.5


def test_identical_series_gives_p_value_near_one():
    a = pd.Series([3.0, 3.0, 3.0, 3.0])
    b = pd.Series([3.0, 3.0, 3.0, 3.0])
    result = paired_permutation_test(a, b, n_permutations=1000, random_state=1)
    assert result.p_value == pytest.approx(1.0)


def test_median_diff_statistic_runs():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(4.0, 0.5, size=100))
    b = pd.Series(rng.normal(3.0, 0.5, size=100))
    result = paired_permutation_test(a, b, statistic="median_diff", n_permutations=2000, random_state=1)
    assert result.p_value < 0.05

