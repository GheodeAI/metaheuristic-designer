import pytest

from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.benchmarks import *
import metaheuristic_designer as mhd

mhd.reset_seed(0)


benchmark_functions = [
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
@pytest.mark.parametrize("bench_class", benchmark_functions)
def test_real_benchmarks(vecsize, bench_class):
    objfunc = bench_class(vecsize)
    pop_init = UniformVectorInitializer(
        vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100
    )

    population = pop_init.generate_population(objfunc)
    for indiv in population:
        objfunc.fitness(indiv, adjusted=False)
        objfunc.fitness(indiv, adjusted=True)


@pytest.mark.parametrize("bench_class", benchmark_functions)
def test_real_benchmarks(bench_class):
    objfunc = bench_class(20)
    pop_init = UniformVectorInitializer(20, -10000, 10000, pop_size=100)

    population = pop_init.generate_population(objfunc)
    for indiv in population:
        objfunc.repair_solution(indiv.genotype)
