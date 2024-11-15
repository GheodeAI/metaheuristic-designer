import pytest

import numpy as np
import scipy as sp
from metaheuristic_designer import Individual
from metaheuristic_designer.operators.operator_functions.permutation import *
import metaheuristic_designer as mhd

rng = mhd.reset_seed(1)

# n_indiv = 100
# n_components = 10
n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components))
sample_pop1 = rng.permuted(sample_pop1, axis=1)

print()
print(sample_pop1)

def test_swap():
    print()
    result_arr = permute_mutation(sample_pop1, 2)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

def test_roll():
    print()
    result_arr = roll_mutation(sample_pop1, 1)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

def test_invert():
    print()
    result_arr = invert_mutation(sample_pop1)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

def test_pmx():
    print()
    result_arr = pmx(sample_pop1)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

def test_order_cross():
    print()
    result_arr = order_cross(sample_pop1)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

