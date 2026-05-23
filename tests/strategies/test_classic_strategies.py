import numpy as np
import pytest

from conftest import (
    rng,
    dummy_initializer,
    dummy_operator,
    dummy_parent_selection,
    dummy_survivor_selection,
)

from metaheuristic_designer.strategies.classic.DE import DE
from metaheuristic_designer.strategies.classic.ES import ES
from metaheuristic_designer.strategies.classic.GA import GA
from metaheuristic_designer.strategies.classic.SA import SA
from metaheuristic_designer.strategies.classic.random_search import RandomSearch
from metaheuristic_designer.strategies.classic.CMA_ES import CMA_ES

from metaheuristic_designer.operator import Operator
from metaheuristic_designer.survivor_selection_base import SurvivorSelection


# ===================================================================
#  CMA_ES – not yet implemented, test written as reminder
# ===================================================================
def test_cma_es_instantiation(rng, dummy_initializer):
    # Will fail until CMA_ES is fully implemented.
    algo = CMA_ES(
        initializer=dummy_initializer,
        name="CMA-ES-test",
    )
    assert algo.name == "CMA-ES-test"
    assert hasattr(algo, "operator")


# ===================================================================
#  DE
# ===================================================================
def test_de_default_creation(rng, dummy_initializer):
    algo = DE(initializer=dummy_initializer, random_state=rng)
    assert algo.name == "DE"
    assert algo.operator is not None
    assert algo.survivor_sel is not None


def test_de_custom_name_and_params(rng, dummy_initializer):
    algo = DE(initializer=dummy_initializer, de_operator_name="DE/rand/1", name="MyDE", F=0.5, Cr=0.7, random_state=rng)
    assert algo.name == "MyDE"


# ===================================================================
#  ES
# ===================================================================
def test_es_mutation_only(rng, dummy_initializer, dummy_operator):
    algo = ES(initializer=dummy_initializer, mutation_op=dummy_operator, random_state=rng)
    assert algo.operator is dummy_operator
    assert algo.name == "ES"


def test_es_with_crossover(rng, dummy_initializer, dummy_operator):
    algo = ES(initializer=dummy_initializer, mutation_op=dummy_operator, crossover_op=dummy_operator, random_state=rng)
    from metaheuristic_designer.operators.composite_operator import CompositeOperator

    assert isinstance(algo.operator, CompositeOperator)


# ===================================================================
#  GA
# ===================================================================
def test_ga_creation(rng, dummy_initializer, dummy_operator, dummy_parent_selection, dummy_survivor_selection):
    algo = GA(
        initializer=dummy_initializer,
        mutation_op=dummy_operator,
        crossover_op=dummy_operator,
        parent_sel=dummy_parent_selection,
        survivor_sel=dummy_survivor_selection,
        mutation_prob=0.1,
        crossover_prob=0.9,
        random_state=rng,
    )
    assert algo.name == "GA"
    assert isinstance(algo.operator, Operator)


def test_ga_default_names(rng, dummy_initializer, dummy_operator, dummy_parent_selection, dummy_survivor_selection):
    algo = GA(
        initializer=dummy_initializer,
        mutation_op=dummy_operator,
        crossover_op=dummy_operator,
        parent_sel=dummy_parent_selection,
        survivor_sel=dummy_survivor_selection,
        random_state=rng,
    )
    assert algo.name == "GA"


# ===================================================================
#  RandomSearch
# ===================================================================
def test_random_search_creation(rng, dummy_initializer):
    algo = RandomSearch(initializer=dummy_initializer, name="RS", random_state=rng)
    assert algo.name == "RS"
    assert algo.operator is not None


# ===================================================================
#  Simulated Annealing
# ===================================================================
def test_sa_initial_temperature(rng, dummy_initializer, dummy_operator):
    algo = SA(initializer=dummy_initializer, operator=dummy_operator, temperature_init=200, random_state=rng)
    assert algo.temperature == 200


def test_sa_temperature_decreases_after_many_steps(rng, dummy_initializer, dummy_operator):
    algo = SA(initializer=dummy_initializer, operator=dummy_operator, iterations=2, temperature_init=100, alpha=0.5, random_state=rng)
    # Initial temperature
    assert algo.temperature == 100
    # Call step many times to guarantee temperature drops
    for _ in range(10):
        algo.update(0)
    # Temperature must be less than initial
    assert algo.temperature < 100
