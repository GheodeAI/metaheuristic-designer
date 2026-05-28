import numpy as np
import pytest

# Import the wrappers – adjust the paths to your actual package structure
from metaheuristic_designer.analysis import CMAWrapper
from metaheuristic_designer.analysis import DEAPWrapper
from metaheuristic_designer.analysis import NevergradWrapper
from metaheuristic_designer.analysis import PyGMOWrapper
from metaheuristic_designer.analysis import ScipyWrapper

from metaheuristic_designer.benchmarks import Sphere


@pytest.fixture(scope="module")
def sphere():
    """3-D Sphere for fast testing."""
    return Sphere(dimension=3, mode="min")


@pytest.fixture(scope="module")
def random_initial_obj(sphere):
    """Worst-case baseline: a random point in the bounds."""
    rng = np.random.default_rng(42)
    x0 = rng.uniform(sphere.lower_bound, sphere.upper_bound, size=sphere.dimension)
    return sphere.objective(x0)


# ----- CMA-ES ----------------------------------------------------------
def test_cma_improves(sphere, random_initial_obj):
    solver = CMAWrapper(sphere, sigma0=0.3, max_iterations=20, seed=42)
    solver.optimize()
    _, best = solver.best_solution()
    assert best < random_initial_obj, f"CMA did not improve: best {best} >= random {random_initial_obj}"


# ----- DEAP ------------------------------------------------------------
def test_deap_improves(sphere, random_initial_obj):
    # Minimal builder for DEAP – must be defined here because we dropped
    # the recorder parameter. Import DEAP inside the test to keep optional.
    deap = pytest.importorskip("deap")
    from deap import base, creator, tools, algorithms
    import random

    def build_ga(objfunc, seed):
        random.seed(seed)
        np.random.seed(seed)
        DIM = objfunc.dimension
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -5.12, 5.12)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (objfunc.objective(np.array(ind)),))
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        pop = toolbox.population(n=20)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)
        return toolbox, pop, stats, hof

    solver = DEAPWrapper(sphere, build_fn=build_ga, ngen=20, seed=42, cxpb=0.7, mutpb=0.3, algorithm=algorithms.eaSimple)
    solver.optimize()
    _, best = solver.best_solution()
    assert best < random_initial_obj, f"DEAP did not improve: best {best} >= random {random_initial_obj}"


# ----- Nevergrad -------------------------------------------------------
def test_nevergrad_improves(sphere, random_initial_obj):
    pytest.importorskip("nevergrad")
    solver = NevergradWrapper(sphere, optimizer_name="DE", budget=400, seed=42)
    solver.optimize()
    _, best = solver.best_solution()
    assert best < random_initial_obj, f"Nevergrad did not improve: best {best} >= random {random_initial_obj}"


# ----- PyGMO -----------------------------------------------------------
def test_pygmo_improves(sphere, random_initial_obj):
    pytest.importorskip("pygmo")
    solver = PyGMOWrapper(sphere, algorithm="de", pop_size=10, generations=20, seed=42)
    solver.optimize()
    _, best = solver.best_solution()
    assert best < random_initial_obj, f"PyGMO did not improve: best {best} >= random {random_initial_obj}"


# ----- SciPy -----------------------------------------------------------
def test_scipy_improves(sphere, random_initial_obj):
    solver = ScipyWrapper(sphere, method="differential_evolution", maxiter=10, seed=42)
    solver.optimize()
    _, best = solver.best_solution()
    # SciPy's differential_evolution needs a few more iterations; we use 10
    assert best < random_initial_obj, f"SciPy did not improve: best {best} >= random {random_initial_obj}"
