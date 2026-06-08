import pytest
import numpy as np

from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.benchmarks import *
import metaheuristic_designer as mhd

real_benchmarks = [
    MaxOnes,
    Sphere,
    HighCondElliptic,
    BentCigar,
    Discus,
    Rosenbrock,
    Ackley,
    Weierstrass,
    Griewank,
    Rastrigin,
    ModSchwefel,
    Katsuura,
    HappyCat,
    HGBat,
    ExpandedGriewankPlusRosenbrock,
    ExpandedShafferF6,
    SumPowell,
    N4XinSheYang,
]


@pytest.mark.parametrize("dimension", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_objective_real(dimension, bench_class):
    objfunc = bench_class(dimension)
    pop_init = UniformInitializer(dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, rng=42)

    if objfunc.vectorized:
        # Vectorized objectives expect a 2‑D batch (population matrix).
        # Create a small population of 2 individuals.
        pop_init_small = UniformInitializer(dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=2, rng=42)
        population = pop_init_small.generate_population(objfunc)
        geno_matrix = population.genotype_matrix  # shape (2, dimension)
        result = objfunc.objective(geno_matrix)  # should return (2,)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.issubdtype(result.dtype, np.floating) or np.issubdtype(result.dtype, np.integer)
    else:
        # Non‑vectorized objectives expect a single 1‑D solution vector.
        rand_vec = pop_init.generate_random()  # shape (dimension,)
        objfunc.objective(rand_vec)


@pytest.mark.parametrize("dimension", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_repair_solution(dimension, bench_class):
    objfunc = bench_class(dimension)
    # Create a small population to get a 2‑D genotype matrix for repair.
    pop_init = UniformInitializer(dimension, -1_000_000, 1_000_000, population_size=3, rng=42)
    population = pop_init.generate_population(objfunc)
    geno_matrix = population.genotype_matrix  # shape (3, dimension)

    # repair_solution now works on genotypes (2‑D matrices).
    repaired = objfunc.repair_solution(geno_matrix)
    assert isinstance(repaired, np.ndarray) and repaired.ndim == 2
    assert repaired.shape == geno_matrix.shape
    assert repaired.min() >= objfunc.lower_bound
    assert repaired.max() <= objfunc.upper_bound


@pytest.mark.parametrize("dimension", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_fitness(dimension, bench_class):
    objfunc = bench_class(dimension)
    pop_init = UniformInitializer(dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, rng=42)
    population = pop_init.generate_population(objfunc)

    # The fitness method internally calls objective; it works for both vectorized and non‑vectorized.
    objfunc.fitness(population)
