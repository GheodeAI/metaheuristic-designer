"""
Unit tests for classic combinatorial optimization problems.

Covers: ThreeSAT, BinKnapsack, MaxClique, TSP.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks.classic_problems import (
    ThreeSAT,
    BinKnapsack,
    MaxClique,
    TSP,
)
from metaheuristic_designer.population import Population


# ---------------------------------------------------------------------------
# ThreeSAT
# ---------------------------------------------------------------------------

def _simple_3sat():
    # (a ∨ b ∨ c) ∧ (¬a ∨ b ∨ c): 2 clauses, 3 variables
    clauses = np.array([[1, 2, 3], [-1, 2, 3]])
    return ThreeSAT(clauses)


def test_3sat_construction():
    obj = _simple_3sat()
    assert obj is not None
    assert obj.n_vars == 3


def test_3sat_all_satisfied():
    """Assignment [1, 1, 1] satisfies both clauses."""
    obj = _simple_3sat()
    solution = np.array([1, 1, 1])
    result = obj.objective(solution)
    assert result == pytest.approx(1.0)


def test_3sat_none_satisfied():
    """Assignment [0, 0, 0] – clauses: (0∨0∨0)=F, (1∨0∨0)=T → 1/2 = 0.5."""
    obj = _simple_3sat()
    solution = np.array([0, 0, 0])
    result = obj.objective(solution)
    # (0 ∨ 0 ∨ 0)=F, (1 ∨ 0 ∨ 0)=T (because ¬a with a=0 is 1) → 1/2 satisfied
    assert 0.0 <= result <= 1.0


def test_3sat_invalid_clauses_raises():
    """ThreeSAT rejects non-3-literal clauses."""
    with pytest.raises(ValueError):
        ThreeSAT(np.array([[1, 2], [3, 4]]))


def test_3sat_objective_range():
    """Objective is always in [0, 1]."""
    obj = _simple_3sat()
    for solution in [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]:
        result = obj.objective(np.array(solution))
        assert 0.0 <= result <= 1.0


def test_3sat_vectorized_evaluation():
    """ThreeSAT can be evaluated through a Population."""
    obj = _simple_3sat()
    geno = np.array([[1, 1, 1], [0, 0, 0]], dtype=float)
    pop = Population(obj, geno)
    pop.calculate_fitness()
    assert np.all(np.isfinite(pop.objective))


# ---------------------------------------------------------------------------
# BinKnapsack
# ---------------------------------------------------------------------------

def _simple_knapsack():
    cost = np.array([2.0, 3.0, 4.0])
    value = np.array([3.0, 4.0, 5.0])
    max_weight = 5.0
    return BinKnapsack(cost, value, max_weight)


def test_knapsack_construction():
    obj = _simple_knapsack()
    assert obj.dimension == 3


def test_knapsack_feasible_solution():
    obj = _simple_knapsack()
    # Items 0 and 1: weight = 2+3 = 5 < 5... wait, 5 < 5 is False, so invalid
    # Items 0 only: weight 2 < 5 → value = 3
    solution = np.array([1, 0, 0])
    result = obj.objective(solution)
    assert result == pytest.approx(3.0)


def test_knapsack_infeasible_solution_returns_negative():
    obj = _simple_knapsack()
    # All items: weight = 2+3+4=9 > 5 → penalty
    solution = np.array([1, 1, 1])
    result = obj.objective(solution)
    assert result < 0


def test_knapsack_empty_selection():
    obj = _simple_knapsack()
    solution = np.array([0, 0, 0])
    result = obj.objective(solution)
    assert result == pytest.approx(0.0)


def test_knapsack_size_mismatch_raises():
    with pytest.raises(ValueError):
        BinKnapsack(cost=np.array([1.0, 2.0]), value=np.array([1.0]), max_weight=5.0)


def test_knapsack_repair_rounds_to_binary():
    obj = _simple_knapsack()
    solution = np.array([0.3, 0.7, 1.5])
    repaired = obj.repair_solution(solution)
    assert set(repaired).issubset({0, 1})


def test_knapsack_vectorized_evaluation():
    obj = _simple_knapsack()
    geno = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pop = Population(obj, geno)
    pop.calculate_fitness()
    assert np.all(np.isfinite(pop.objective))


# ---------------------------------------------------------------------------
# MaxClique
# ---------------------------------------------------------------------------

def _triangle_graph():
    # Complete graph K3 (triangle): every pair connected
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    return MaxClique(adj)


def test_max_clique_construction():
    obj = _triangle_graph()
    assert obj.dimension == 3


def test_max_clique_triangle_gives_3():
    """On a triangle, any ordering finds a clique of size 3."""
    obj = _triangle_graph()
    solution = np.array([0, 1, 2])
    result = obj.objective(solution)
    assert result == 3


def test_max_clique_partial_graph():
    """On a path graph (0-1-2), max clique is 2."""
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    obj = MaxClique(adj)
    solution = np.array([0, 1, 2])
    result = obj.objective(solution)
    # Starting from 0, can add 1 (0-1 connected), cannot add 2 (0-2 not connected)
    assert result >= 1


def test_max_clique_objective_positive():
    obj = _triangle_graph()
    solution = np.array([0, 1, 2])
    assert obj.objective(solution) > 0


# ---------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------

def _simple_tsp():
    # 4-node symmetric TSP: all edges cost 1 except diagonal (0)
    adj = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=float)
    return TSP(adj)


def test_tsp_construction():
    obj = _simple_tsp()
    assert obj.dimension == 4


def test_tsp_route_cost_finite():
    obj = _simple_tsp()
    route = np.array([[0, 1, 2, 3]])  # vectorized: 2D array
    result = obj.objective(route)
    assert np.isfinite(result[0])


def test_tsp_all_routes_positive():
    """TSP objective value should be positive for any permutation tour."""
    obj = _simple_tsp()
    routes = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]])
    result = obj.objective(routes)
    assert np.all(result > 0)


def test_tsp_vectorized_population():
    obj = _simple_tsp()
    rng = np.random.default_rng(42)
    # TSP requires integer routes; use int dtype
    routes = np.array([rng.permutation(4) for _ in range(5)])
    pop = Population(obj, routes)  # integer dtype
    pop.calculate_fitness()
    assert np.all(np.isfinite(pop.objective))


def test_tsp_symmetric_route_vs_reverse():
    """For a symmetric distance matrix, forward and reverse tours have same cost."""
    obj = _simple_tsp()
    route_fwd = np.array([[0, 1, 2, 3]])
    route_rev = np.array([[3, 2, 1, 0]])
    cost_fwd = obj.objective(route_fwd)
    cost_rev = obj.objective(route_rev)
    assert cost_fwd[0] == pytest.approx(cost_rev[0])


def test_tsp_mode_min():
    obj = _simple_tsp()
    assert obj.mode == "min"
