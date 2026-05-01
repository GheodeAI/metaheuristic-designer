import numpy as np
import json
from conftest import (
    sphere_objfunc,
    simple_strategy,
    rng,
)

from metaheuristic_designer.algorithms.standard_algorithm import StandardAlgorithm
from metaheuristic_designer.algorithms.memetic_algorithm import MemeticAlgorithm
from metaheuristic_designer.algorithms.algorithm_selection import AlgorithmSelection
from metaheuristic_designer.parent_selection import NullParentSelection, create_parent_selection
from metaheuristic_designer.strategies import GA, LocalSearch


def test_full_pipeline_one_generation(sphere_objfunc, simple_strategy):
    algo = StandardAlgorithm(
        sphere_objfunc, simple_strategy,
        ngen=1, neval=100, verbose=False,
        stop_cond="ngen",
    )
    algo.initialize()
    pop = algo.step()
    assert len(pop) == simple_strategy.pop_size
    _, best_fit = algo.best_solution()
    assert np.isfinite(best_fit)


def test_full_pipeline_five_generations(sphere_objfunc, simple_strategy):
    algo = StandardAlgorithm(
        sphere_objfunc, simple_strategy,
        ngen=5, neval=500, verbose=False,
        stop_cond="ngen",
    )
    algo.optimize()
    _, best_fit = algo.best_solution()
    assert np.isfinite(best_fit)
    assert len(algo.fit_history) == 5


def test_memetic_pipeline(sphere_objfunc, dummy_strategy, dummy_initializer, rng):
    # Create a separate local searcher – using dummy components plus an order‑preserving survivor selection
    from metaheuristic_designer.search_strategy import SearchStrategy
    from metaheuristic_designer.survivor_selection import create_survivor_selection
    from metaheuristic_designer.operators import create_operator

    # Operator that preserves order (NullOperator does nothing, but it's order‑preserving)
    operator = create_operator("nothing")  # NullOperator is order‑preserving
    survivor_sel = create_survivor_selection("one_to_one")  # order‑preserving

    local_searcher = SearchStrategy(
        initializer=dummy_initializer,
        operator=operator,
        survivor_sel=survivor_sel,
        name="local_searcher",
    )

    # Improvement selector (any parent selection works; we don't care about order here)
    improvement_selector = NullParentSelection()

    algo = MemeticAlgorithm(
        sphere_objfunc,
        search_strategy=dummy_strategy,       # global search (simple Null strategy)
        local_search=local_searcher,          # completely separate instance
        improvement_selection=improvement_selector,
        ngen=2, neval=200, verbose=False, stop_cond="ngen",
    )
    algo.optimize()
    _, best_fit = algo.best_solution()
    assert np.isfinite(best_fit)


def test_serialization_after_run(sphere_objfunc, simple_strategy, tmp_path):
    algo = StandardAlgorithm(
        sphere_objfunc, simple_strategy,
        ngen=2, neval=100, verbose=False,
        stop_cond="ngen",
    )
    algo.optimize()
    state = algo.get_state(show_fit_history=True, show_gen_history=False, show_population=False)
    assert "fit_history" in state
    assert len(state["fit_history"]) == 2

    fpath = tmp_path / "state.json"
    algo.store_state(str(fpath), readable=True, show_fit_history=True, show_gen_history=False, show_population=False)
    with open(fpath, "r") as f:
        data = json.load(f)
    assert data["name"] == "integration_strat"
    assert data["fit_history"] == state["fit_history"]


def test_algorithm_selection_run(sphere_objfunc, simple_strategy):
    algo1 = StandardAlgorithm(sphere_objfunc, simple_strategy, ngen=1, neval=50, verbose=False, stop_cond="ngen", name="algo1")
    algo2 = StandardAlgorithm(sphere_objfunc, simple_strategy, ngen=1, neval=50, verbose=False, stop_cond="ngen", name="algo2")
    sel = AlgorithmSelection([algo1, algo2], params={"repetitions": 2, "verbose": False})
    best_sol, best_fit, report = sel.optimize()
    assert best_fit is not None
    # AlgorithmSelection groups by name; we should have one row per algorithm
    assert len(report) == 2
    assert set(report["name"]) == {"algo1", "algo2"}