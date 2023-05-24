import pytest

import numpy as np
from pyevolcomp import Individual
from pyevolcomp.Initializers import *


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize(
    "min_val, max_val", [
        (0,1),
        (-1,1),
        (-100,2),
        (2, 24)
    ]
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_vec_init(vec_size, min_val, max_val, pop_size):
    pop_init = UniformVectorInitializer(vec_size, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.max() <= max_val
        assert rand_inidv.genotype.min() >= min_val
        assert rand_inidv.genotype.size == vec_size
    
        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.max() <= max_val
        assert rand_inidv.genotype.min() >= min_val
        assert rand_inidv.genotype.size == vec_size
    
    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.max() <= max_val
        assert indiv.genotype.min() >= min_val
        assert indiv.genotype.size == vec_size


@pytest.mark.parametrize("list_size", [1, 2, 10, 100])
@pytest.mark.parametrize(
    "min_val, max_val", [
        (0,1),
        (-1,1),
        (-100,2),
        (2, 24)
    ]
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_list_init(list_size, min_val, max_val, pop_size):
    pop_init = UniformListInitializer(list_size, min_val, max_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert max(rand_inidv.genotype) <= max_val
        assert min(rand_inidv.genotype) >= min_val
        assert len(rand_inidv.genotype) == list_size
    
        rand_inidv = pop_init.generate_individual(None)
        assert max(rand_inidv.genotype) <= max_val
        assert min(rand_inidv.genotype) >= min_val
        assert len(rand_inidv.genotype) == list_size
    
    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert max(indiv.genotype) <= max_val
        assert min(indiv.genotype) >= min_val
        assert len(indiv.genotype) == list_size


@pytest.mark.parametrize("vec_size", [1, 2, 10, 100])
@pytest.mark.parametrize(
    "mean_val, std_val", [
        (0,1),
        (-1,1),
        (-100,2),
        (2, 24)
    ]
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_vec_init(vec_size, mean_val, std_val, pop_size):
    pop_init = GaussianVectorInitializer(vec_size, mean_val, std_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert rand_inidv.genotype.size == vec_size
    
        rand_inidv = pop_init.generate_individual(None)
        assert rand_inidv.genotype.size == vec_size
    
    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert indiv.genotype.size == vec_size


@pytest.mark.parametrize("list_size", [1, 2, 10, 100])
@pytest.mark.parametrize(
    "mean_val, std_val", [
        (0,1),
        (-1,1),
        (-100,2),
        (2, 24)
    ]
)
@pytest.mark.parametrize("pop_size", [1, 10, 100])
def test_uniform_list_init(list_size, mean_val, std_val, pop_size):
    pop_init = GaussianListInitializer(list_size, mean_val, std_val, pop_size)

    for _ in range(30):
        rand_inidv = pop_init.generate_random(None)
        assert len(rand_inidv.genotype) == list_size
    
        rand_inidv = pop_init.generate_individual(None)
        assert len(rand_inidv.genotype) == list_size
    
    rand_pop = pop_init.generate_population(None)
    assert len(rand_pop) == pop_size

    for indiv in rand_pop:
        assert len(indiv.genotype) == list_size

