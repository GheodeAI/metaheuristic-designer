import numpy as np
from conftest import (
    dummy_objfunc,
    dummy_strategy,
    dummy_parent_selection,
    rng,
)

from metaheuristic_designer.algorithms.memetic_algorithm import MemeticAlgorithm
from metaheuristic_designer.survivor_selection import SurvivorSelectionFromLambda


def test_memetic_algorithm_step(dummy_objfunc, dummy_strategy, dummy_parent_selection):
    # A trivial local search that just copies the offspring
    local_search = dummy_strategy  # reuse dummy strategy as local search
    improve_choice = dummy_parent_selection  # selects all parents

    algo = MemeticAlgorithm(
        objfunc=dummy_objfunc,
        search_strategy=dummy_strategy,
        local_search=local_search,
        improve_choice=improve_choice,
        ngen=1,
        verbose=False,
    )

    pop = algo.initialize()
    new_pop = algo.step(population=pop)
    assert len(new_pop) == len(pop)
    assert len(algo.fit_history) == 1


def test_memetic_algorithm_name(dummy_objfunc, dummy_strategy, dummy_survivor_selection):
    algo = MemeticAlgorithm(
        objfunc=dummy_objfunc,
        search_strategy=dummy_strategy,
        local_search=dummy_strategy,
        improve_choice=dummy_survivor_selection,
        name="CustomMemetic",
        ngen=1, verbose=False,
    )
    assert algo.name == "CustomMemetic"
    # When no custom name, uses "Memetic {search_strategy.name}"
    algo2 = MemeticAlgorithm(
        objfunc=dummy_objfunc,
        search_strategy=dummy_strategy,
        local_search=dummy_strategy,
        improve_choice=dummy_survivor_selection,
        ngen=1, verbose=False,
    )
    assert algo2.name == f"Memetic {dummy_strategy.name}"