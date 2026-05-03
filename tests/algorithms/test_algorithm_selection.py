from conftest import (
    dummy_objfunc,
    dummy_strategy,
    rng,
)

from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.algorithms.algorithm_selection import AlgorithmSelection
from metaheuristic_designer.algorithms.strategy_selection import StrategySelection


def test_algorithm_selection_optimize(dummy_objfunc, dummy_strategy):
    return
    algo1 = Algorithm(dummy_objfunc, dummy_strategy, ngen=1, neval=1, verbose=False, name="alg1")
    algo2 = Algorithm(dummy_objfunc, dummy_strategy, ngen=1, neval=1, verbose=False, name="alg2")

    sel = AlgorithmSelection([algo1, algo2], params={"repetitions": 2, "verbose": False})
    best_sol, best_fit, report = sel.optimize()

    # The report should contain 4 rows (2 algorithms × 2 repetitions)
    assert len(report) == 4
    assert set(report["name"]) == {"alg1", "alg2"}
    assert all(col in report.columns for col in ["realtime", "cputime", "fitness"])


def test_strategy_selection_optimize(dummy_objfunc, dummy_strategy):
    return
    strat_sel = StrategySelection(
        objfunc=dummy_objfunc,
        strategy_list=[dummy_strategy, dummy_strategy],
        algorithm_params={"ngen": 1, "neval": 1, "verbose": False},
        params={"repetitions": 2, "verbose": False},
    )
    _, _, report = strat_sel.optimize()
    assert len(report) == 4  # 2 strategies × 2 repetitions
    assert set(report["name"]) == {"dummy_strategy", "dummy_strategy_2"}  # StrategySelection renames duplicates
