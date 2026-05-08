"""
Integration tests for Estimation of Distribution Algorithms (EDA).

Contract:
- BernoulliPBIL and BernoulliUMDA must run without error on binary problems.
- GaussianPBIL and GaussianUMDA must run without error on continuous problems.
- BinomialPBIL and BinomialUMDA must run without error on categorical problems.
- CrossEntropyMethod must run without error on continuous problems.
- Best solution has the correct dimension.
- Fitness must be finite.
- Reproducible with fixed seed.
"""

import numpy as np
import pytest

from metaheuristic_designer.benchmarks import MaxOnes, Sphere
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.strategies.EDA import BernoulliPBIL, BernoulliUMDA
from metaheuristic_designer.strategies.EDA.PBIL import GaussianPBIL, BinomialPBIL
from metaheuristic_designer.strategies.EDA.UMDA import GaussianUMDA, BinomialUMDA
from metaheuristic_designer.strategies.EDA.cross_entropy_method import CrossEntropyMethod
from metaheuristic_designer.algorithm import Algorithm


COMMON = {
    "stop_cond": "max_iterations",
    "max_iterations": 30,
    "reporter": "silent",
}


def _build_pbil(dim, seed):
    objfunc = MaxOnes(dimension=dim)
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=0, upper_bound=1,
        pop_size=pop_size, dtype=int, random_state=seed
    )
    strategy = BernoulliPBIL(initializer, offspring_size=pop_size, random_state=seed)
    return Algorithm(objfunc, strategy, **COMMON)


def _build_umda(dim, seed):
    objfunc = MaxOnes(dimension=dim)
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=0, upper_bound=1,
        pop_size=pop_size, dtype=int, random_state=seed
    )
    strategy = BernoulliUMDA(initializer, offspring_size=pop_size, random_state=seed)
    return Algorithm(objfunc, strategy, **COMMON)


# ---------------------------------------------------------------------------
# BernoulliPBIL on MaxOnes
# ---------------------------------------------------------------------------

def test_pbil_binary_runs():
    algo = _build_pbil(dim=8, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_pbil_binary_solution_dimension():
    algo = _build_pbil(dim=8, seed=1)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (8,)


@pytest.mark.xfail(
    reason="BUG: EDA strategies are not fully reproducible. Even with the same seed, consecutive "
           "runs with identical configuration can produce different fitness values. This indicates "
           "a global random state leak — likely from an initializer or distribution that calls "
           "np.random.* instead of the injected random_state. See ERRORES.md."
)
def test_pbil_binary_reproducible():
    f1 = _build_pbil(dim=8, seed=42).optimize().best_solution()[1]
    f2 = _build_pbil(dim=8, seed=42).optimize().best_solution()[1]
    assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# BernoulliUMDA on MaxOnes
# ---------------------------------------------------------------------------

def test_umda_binary_runs():
    algo = _build_umda(dim=8, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_umda_binary_solution_dimension():
    algo = _build_umda(dim=8, seed=2)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (8,)


@pytest.mark.xfail(reason="EDA does not appear to be fully reproducible with seed control (see ERRORES.md)")
def test_umda_binary_reproducible():
    f1 = _build_umda(dim=8, seed=7).optimize().best_solution()[1]
    f2 = _build_umda(dim=8, seed=7).optimize().best_solution()[1]
    assert f1 == pytest.approx(f2)


# ---------------------------------------------------------------------------
# GaussianPBIL on Sphere
# ---------------------------------------------------------------------------

def _build_gaussian_pbil(dim, seed):
    objfunc = Sphere(dimension=dim, mode="min")
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=-5, upper_bound=5,
        pop_size=pop_size, dtype=float, random_state=seed
    )
    strategy = GaussianPBIL(initializer, offspring_size=pop_size, random_state=seed, lr=0.1)
    return Algorithm(objfunc, strategy,
                     stop_cond="max_iterations", max_iterations=30, reporter="silent")


def test_gaussian_pbil_runs():
    algo = _build_gaussian_pbil(dim=4, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_gaussian_pbil_solution_dimension():
    algo = _build_gaussian_pbil(dim=4, seed=1)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (4,)


# ---------------------------------------------------------------------------
# GaussianUMDA on Sphere
# ---------------------------------------------------------------------------

def _build_gaussian_umda(dim, seed):
    objfunc = Sphere(dimension=dim, mode="min")
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=-5, upper_bound=5,
        pop_size=pop_size, dtype=float, random_state=seed
    )
    strategy = GaussianUMDA(initializer, offspring_size=pop_size, random_state=seed)
    return Algorithm(objfunc, strategy,
                     stop_cond="max_iterations", max_iterations=30, reporter="silent")


def test_gaussian_umda_runs():
    algo = _build_gaussian_umda(dim=4, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_gaussian_umda_solution_dimension():
    algo = _build_gaussian_umda(dim=4, seed=3)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (4,)


# ---------------------------------------------------------------------------
# BinomialPBIL on MaxOnes
# ---------------------------------------------------------------------------

def _build_binomial_pbil(dim, seed, n_categories=2):
    objfunc = MaxOnes(dimension=dim)
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=0, upper_bound=n_categories - 1,
        pop_size=pop_size, dtype=int, random_state=seed
    )
    strategy = BinomialPBIL(initializer, offspring_size=pop_size, random_state=seed,
                             n=n_categories, lr=0.1)
    return Algorithm(objfunc, strategy,
                     stop_cond="max_iterations", max_iterations=30, reporter="silent")


def test_binomial_pbil_runs():
    algo = _build_binomial_pbil(dim=8, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_binomial_pbil_requires_n_parameter():
    """BinomialPBIL raises ValueError when n is not specified."""
    objfunc = MaxOnes(dimension=4)
    initializer = UniformInitializer(dimension=4, lower_bound=0, upper_bound=1,
                                     pop_size=10, dtype=int, random_state=0)
    with pytest.raises(ValueError, match="n"):
        BinomialPBIL(initializer, offspring_size=10)


# ---------------------------------------------------------------------------
# BinomialUMDA on MaxOnes
# ---------------------------------------------------------------------------

def _build_binomial_umda(dim, seed, n_categories=2):
    objfunc = MaxOnes(dimension=dim)
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=0, upper_bound=n_categories - 1,
        pop_size=pop_size, dtype=int, random_state=seed
    )
    strategy = BinomialUMDA(initializer, offspring_size=pop_size, random_state=seed,
                             n=n_categories)
    return Algorithm(objfunc, strategy,
                     stop_cond="max_iterations", max_iterations=30, reporter="silent")


def test_binomial_umda_runs():
    algo = _build_binomial_umda(dim=8, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_binomial_umda_requires_n_parameter():
    """BinomialUMDA raises ValueError when n is not specified."""
    objfunc = MaxOnes(dimension=4)
    initializer = UniformInitializer(dimension=4, lower_bound=0, upper_bound=1,
                                     pop_size=10, dtype=int, random_state=0)
    with pytest.raises(ValueError, match="n"):
        BinomialUMDA(initializer, offspring_size=10)


# ---------------------------------------------------------------------------
# CrossEntropyMethod on Sphere
# ---------------------------------------------------------------------------

def _build_cem(dim, seed):
    objfunc = Sphere(dimension=dim, mode="min")
    pop_size = 20
    initializer = UniformInitializer(
        dimension=dim, lower_bound=-5, upper_bound=5,
        pop_size=pop_size, dtype=float, random_state=seed
    )
    strategy = CrossEntropyMethod(initializer, random_state=seed, elite_amount=5)
    return Algorithm(objfunc, strategy,
                     stop_cond="max_iterations", max_iterations=30, reporter="silent")


def test_cem_runs():
    algo = _build_cem(dim=4, seed=0)
    population = algo.optimize()
    solution, fitness = population.best_solution()
    assert solution is not None
    assert np.isfinite(fitness)


def test_cem_solution_dimension():
    algo = _build_cem(dim=4, seed=5)
    population = algo.optimize()
    solution, _ = population.best_solution()
    assert solution.shape == (4,)
