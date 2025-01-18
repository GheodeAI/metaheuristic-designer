import pytest

from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.benchmarks import *
import metaheuristic_designer as mhd

mhd.reset_seed(0)


real_benchmarks = [
    MaxOnes,
    MaxOnesReal,
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
    pop_init = UniformVectorInitializer(vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)

    rand_vec = pop_init.generate_random()
    objfunc.objective(rand_vec)

@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_repair_solution(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformVectorInitializer(vecsize, -1000000, 1000000, pop_size=100)

    rand_vec = pop_init.generate_random()
    repaired = objfunc.repair_solution(rand_vec)
    assert repaired.min() >= objfunc.low_lim
    assert repaired.max() <= objfunc.up_lim

@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_repair_solution(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformVectorInitializer(vecsize, -1000000, 1000000, pop_size=100)

    rand_vec = pop_init.generate_random()
    objfunc.penalize(rand_vec)

@pytest.mark.parametrize("vecsize", [2, 5, 10, 20, 30])
@pytest.mark.parametrize("bench_class", real_benchmarks)
def test_fitness(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformVectorInitializer(vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)

    population = pop_init.generate_population(objfunc)
    objfunc.fitness(population, adjusted=False)
    objfunc.fitness(population, adjusted=True)
