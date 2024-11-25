import pytest
from copy import copy 
from metaheuristic_designer import Population
from metaheuristic_designer.operators import OperatorVector, vector_ops_map
from metaheuristic_designer.benchmarks.benchmark_funcs import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

vector_ops = [i for i in vector_ops_map.keys() if i not in ["mutate1sigma", "mutatensigmas", "samplesigma"]]

pop_size = 100
example_population1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_population2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_population3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))

def test_errors():
    with pytest.raises(ValueError):
        OperatorVector("not_a_method")


@pytest.mark.parametrize("population", [example_population1, example_population2, example_population3])
@pytest.mark.parametrize("op_method", vector_ops)
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    operator = OperatorVector(op_method, "default")

    if op_method in ["xor", "flip", "xorcross", "flipcross"]:
        population = copy(population)
        population.genotype_set = population.genotype_set.astype(int)

    new_population = operator.evolve(population, pop_init)
    assert isinstance(new_population, Population)