import pytest

import numpy as np
import scipy as sp
from metaheuristic_designer import Individual
from metaheuristic_designer.operators.mutation import *
import metaheuristic_designer as mhd

# n_indiv = 100
# n_components = 10
n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) + 10 * np.arange(n_indiv).reshape((n_indiv, 1))
sample_pop_bin1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) < np.arange(n_indiv).reshape((n_indiv, 1)) % (n_components + 1)
sample_pop_bin1 = sample_pop_bin1.astype(int)

print()
print(sample_pop1)

def test_gaussian_mutation():
    print()
    result_arr = gaussian_mutation(sample_pop1, 0.1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_cauchy_mutation():
    print()
    result_arr = cauchy_mutation(sample_pop1, 0.1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_laplace_mutation():
    print()
    result_arr = laplace_mutation(sample_pop1, 0.1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_uniform_mutation():
    print()
    result_arr = uniform_mutation(sample_pop1, 0.1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_poisson_mutation():
    print()
    result_arr = poisson_mutation(sample_pop1, 1, 1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_bernoulli_mutation():
    print()
    result_arr = bernoulli_mutation(sample_pop1, 0.5)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_mutate_sample():
    print()
    result_arr = mutate_sample(sample_pop1.astype(float), distrib=ProbDist.GAUSS, scale=1, N=1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_mutate_noise():
    print()
    result_arr = mutate_noise(sample_pop1.astype(float), distrib=ProbDist.GAUSS, loc=0, scale=1, N=1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_random_sample():
    print()
    result_arr = rand_sample(sample_pop1.astype(float), distrib=ProbDist.GAUSS, scale=1, N=1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_random_noise():
    print()
    result_arr = rand_noise(sample_pop1.astype(float), distrib=ProbDist.GAUSS, loc=0, scale=1, N=1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_xor_mask_byte():
    print()
    result_arr = xor_mask(sample_pop1, 2)
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape

def test_xor_mask_bin():
    print()
    result_arr = xor_mask(sample_pop_bin1, 2, "bin")
    print(result_arr)
    assert result_arr.shape == sample_pop1.shape