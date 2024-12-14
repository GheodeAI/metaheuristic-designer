import pytest

import numpy as np
from metaheuristic_designer.operators.operator_functions.permutation import *
import metaheuristic_designer as mhd

rng = mhd.reset_seed(0)

n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components))
sample_pop1 = rng.permuted(sample_pop1, axis=1)


def test_swap():
    result_arr = permute_mutation(sample_pop1, 2)
    assert result_arr.shape == sample_pop1.shape

def test_roll():
    result_arr = roll_mutation(sample_pop1, 1)
    assert result_arr.shape == sample_pop1.shape

def test_invert():
    result_arr = invert_mutation(sample_pop1)
    assert result_arr.shape == sample_pop1.shape

def test_pmx():
    result_arr = pmx(sample_pop1)
    assert result_arr.shape == sample_pop1.shape

def test_order_cross():
    result_arr = order_cross(sample_pop1)
    assert result_arr.shape == sample_pop1.shape

