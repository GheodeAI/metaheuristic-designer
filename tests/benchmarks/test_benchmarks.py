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


@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_objective_real(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformInitializer(vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=42)

    if objfunc.vectorized:
        # Vectorized objectives expect a 2‑D batch (population matrix).
        # Create a small population of 2 individuals.
        pop_init_small = UniformInitializer(vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=2, random_state=42)
        population = pop_init_small.generate_population(objfunc)
        geno_matrix = population.genotype_matrix  # shape (2, vecsize)
        result = objfunc.objective(geno_matrix)  # should return (2,)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.issubdtype(result.dtype, np.floating) or np.issubdtype(result.dtype, np.integer)
    else:
        # Non‑vectorized objectives expect a single 1‑D solution vector.
        rand_vec = pop_init.generate_random()  # shape (vecsize,)
        objfunc.objective(rand_vec)


@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_repair_solution(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    # Create a small population to get a 2‑D genotype matrix for repair.
    pop_init = UniformInitializer(vecsize, -1_000_000, 1_000_000, pop_size=3, random_state=42)
    population = pop_init.generate_population(objfunc)
    geno_matrix = population.genotype_matrix  # shape (3, vecsize)

    # repair_solution now works on genotypes (2‑D matrices).
    repaired = objfunc.repair_solution(geno_matrix)
    assert isinstance(repaired, np.ndarray) and repaired.ndim == 2
    assert repaired.shape == geno_matrix.shape
    assert repaired.min() >= objfunc.low_lim
    assert repaired.max() <= objfunc.up_lim


@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_fitness(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformInitializer(vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=42)
    population = pop_init.generate_population(objfunc)

    # The fitness method internally calls objective; it works for both vectorized and non‑vectorized.
    objfunc.fitness(population)
