import pytest
import numpy as np
from metaheuristic_designer import Population
from metaheuristic_designer.selectionMethods import ParentSelection, parent_sel_map
from metaheuristic_designer.benchmarks import Sphere
from metaheuristic_designer.initializers import UniformVectorInitializer
import metaheuristic_designer as mhd

mhd.reset_seed(0)

parent_sel_methods = [i for i in parent_sel_map]

pop_size = 100
example_population1 = Population(Sphere(3), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 3)))
example_population1.fitness = np.arange(pop_size)

example_population2 = Population(Sphere(20), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 20)))
example_population2.fitness = np.arange(pop_size)

example_population3 = Population(Sphere(100), mhd.RAND_GEN.uniform(-100, 100, (pop_size, 100)))
example_population3.fitness = np.arange(pop_size)


@pytest.mark.parametrize("population", [example_population1,example_population2,example_population3])
@pytest.mark.parametrize("op_method", parent_sel_methods)
def test_basic_working(population, op_method):
    pop_init = UniformVectorInitializer(population.vec_size, 0, 1, pop_size)
    surv_sel = ParentSelection(op_method, {"amount": 20, "p": 0.5})

    selected_population = surv_sel.select(population)

    assert isinstance(selected_population, Population)
    assert id(selected_population) != id(population) or op_method == "nothing"