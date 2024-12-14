import pytest

import numpy as np
from metaheuristic_designer.operators.operator_functions.crossover import *

# n_indiv = 100
# n_components = 10
n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) + 10 * np.arange(n_indiv).reshape((n_indiv, 1))
sample_pop_bin1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) < np.arange(n_indiv).reshape((n_indiv, 1)) % (n_components + 1)
sample_pop_bin1 = sample_pop_bin1.astype(int)


def test_1p_cross():
    result_arr = cross_1p(sample_pop1)
    assert result_arr.shape == sample_pop1.shape


def test_2p_cross():
    result_arr = cross_2p(sample_pop1)
    assert result_arr.shape == sample_pop1.shape


def test_mp_cross():
    result_arr = cross_mp(sample_pop1)
    assert result_arr.shape == sample_pop1.shape


def test_weighted_average_cross():
    result_arr = weighted_average_cross(sample_pop1, 0.25)
    assert result_arr.shape == sample_pop1.shape


def test_blxalpha():
    result_arr = blxalpha(sample_pop1, 0.5)
    assert result_arr.shape == sample_pop1.shape


def test_xor_cross_bin():
    result_arr = xor_cross(sample_pop_bin1)
    assert result_arr.shape == sample_pop1.shape


def test_xor_cross_byte():
    result_arr = xor_cross(sample_pop1)
    assert result_arr.shape == sample_pop1.shape


def test_multi_cross():
    result_arr = multi_cross(sample_pop1, 3)
    assert result_arr.shape == sample_pop1.shape


def test_cross_inter_avg():
    result_arr = cross_inter_avg(sample_pop1, 3)
    assert result_arr.shape == sample_pop1.shape
