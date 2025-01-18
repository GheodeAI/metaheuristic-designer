import pytest
import numpy as np
from metaheuristic_designer import Population
from metaheuristic_designer.selectionMethods import SurvivorSelection, surv_method_map
from metaheuristic_designer.benchmarks import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

surv_methods = [i for i in surv_method_map]

pop_size = 100
example_population1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_population1.fitness = np.arange(pop_size)
example_offspring1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_offspring1.fitness = np.arange(pop_size) + pop_size

example_population2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_population2.fitness = np.arange(pop_size)
example_offspring2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_offspring2.fitness = np.arange(pop_size) + pop_size

example_population3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))
example_population3.fitness = np.arange(pop_size)
example_offspring3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))
example_offspring3.fitness = np.arange(pop_size) + pop_size


@pytest.mark.parametrize("population, offspring", [
        (example_population1, example_offspring1),
        (example_population2, example_offspring2),
        (example_population3, example_offspring3)
])
@pytest.mark.parametrize("op_method", surv_methods)
def test_basic_working(population, offspring, op_method):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    surv_sel = SurvivorSelection(op_method, {"Fd": 0.1, "Pd": 0.1, "attempts": 3, "maxPopSize": pop_size, "amount": 10, "p": 0.5})

    selected_population = surv_sel.select(population, offspring)

    assert isinstance(selected_population, Population)
    assert id(selected_population) != id(population)
