import pytest

import numpy as np
import scipy as sp
from metaheuristic_designer import Individual
from metaheuristic_designer.operators.operator_functions.differential_evolution import *
import metaheuristic_designer as mhd

# n_indiv = 100
# n_components = 10
n_indiv = 10
n_components = 5
sample_pop1 = np.tile(np.arange(n_components), n_indiv).reshape((n_indiv, n_components)) + 10 * np.arange(n_indiv).reshape((n_indiv, 1))

print()
print(sample_pop1)

def test_de_rand1():
    print()
    result_arr = DE_rand1(sample_pop1.astype(float), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_best1():
    print()
    result_arr = DE_best1(sample_pop1.astype(float), -np.arange(n_indiv), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_rand2():
    print()
    result_arr = DE_rand2(sample_pop1.astype(float), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_best2():
    print()
    result_arr = DE_best2(sample_pop1.astype(float), -np.arange(n_indiv), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_current_to_rand1():
    print()
    result_arr = DE_current_to_rand1(sample_pop1.astype(float), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_current_to_best1():
    print()
    result_arr = DE_current_to_best1(sample_pop1.astype(float), -np.arange(n_indiv), 0.9, 0.8)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape

def test_de_current_to_pbest1():
    print()
    result_arr = DE_current_to_pbest1(sample_pop1.astype(float), -np.arange(n_indiv), 0.9, 0.8, 0.1)
    print(result_arr.round(3))
    assert result_arr.shape == sample_pop1.shape