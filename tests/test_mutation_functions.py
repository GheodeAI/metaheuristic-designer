import pytest

import numpy as np
import scipy as sp
from metaheuristic_designer.operators.operator_functions.mutation import *
import metaheuristic_designer as mhd

rng = mhd.reset_seed(0)

n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) + 10 * np.arange(n_indiv).reshape((n_indiv, 1))
sample_pop_bin1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) < np.arange(n_indiv).reshape((n_indiv, 1)) % (n_components + 1)
sample_pop_bin1 = sample_pop_bin1.astype(int)


def test_gaussian_mutation():
    result_arr = gaussian_mutation(sample_pop1, 0.1)
    assert result_arr.shape == sample_pop1.shape

def test_cauchy_mutation():
    result_arr = cauchy_mutation(sample_pop1, 0.1)
    assert result_arr.shape == sample_pop1.shape

def test_laplace_mutation():
    result_arr = laplace_mutation(sample_pop1, 0.1)
    assert result_arr.shape == sample_pop1.shape

def test_uniform_mutation():
    result_arr = uniform_mutation(sample_pop1, 0.1)
    assert result_arr.shape == sample_pop1.shape

def test_poisson_mutation():
    result_arr = poisson_mutation(sample_pop1, 1, 1)
    assert result_arr.shape == sample_pop1.shape

def test_bernoulli_mutation():
    result_arr = bernoulli_mutation(sample_pop1, 0.5)
    assert result_arr.shape == sample_pop1.shape

def test_mutate_sample():
    result_arr = mutate_sample(sample_pop1.astype(float), distrib=ProbDist.GAUSS, scale=1, N=1)
    assert result_arr.shape == sample_pop1.shape

def test_mutate_noise():
    result_arr = mutate_noise(sample_pop1.astype(float), distrib=ProbDist.GAUSS, loc=0, scale=1, N=1)
    assert result_arr.shape == sample_pop1.shape

def test_random_sample():
    result_arr = rand_sample(sample_pop1.astype(float), distrib=ProbDist.GAUSS, scale=1, N=1)
    assert result_arr.shape == sample_pop1.shape

def test_random_noise():
    result_arr = rand_noise(sample_pop1.astype(float), distrib=ProbDist.GAUSS, loc=0, scale=1, N=1)
    assert result_arr.shape == sample_pop1.shape

def test_xor_mask_byte():
    result_arr = xor_mask(sample_pop1, 2)
    assert result_arr.shape == sample_pop1.shape

def test_xor_mask_bin():
    result_arr = xor_mask(sample_pop_bin1, 2, "bin")
    assert result_arr.shape == sample_pop1.shape