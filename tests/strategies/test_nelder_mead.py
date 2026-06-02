import numpy as np
import pytest

from metaheuristic_designer.initializer import Initializer
from metaheuristic_designer.objective_function import ObjectiveFromLambda
from metaheuristic_designer.population import Population
from metaheuristic_designer.strategies.classic.nelder_mead import NelderMead


class FixedInitializer(Initializer):
    """Deterministic initializer that always returns the same simplex."""

    def __init__(self, points, **kwargs):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be a 2-D array")
        super().__init__(
            dimension=points.shape[1],
            population_size=points.shape[0],
            **kwargs,
        )
        self.points = points

    def generate_random(self):
        return self.points[0].copy()

    def generate_population(self, objfunc, n_individuals=None):
        return Population(objfunc, self.points.copy(), encoding=self.encoding)


def make_peak_objective():
    """A simple 2D maximization problem with a clear optimum near [1, 1]."""
    return ObjectiveFromLambda(
        lambda x: -np.sum((np.asarray(x, dtype=float) - np.array([1.0, 1.0])) ** 2),
        dimension=2,
        lower_bound=-5,
        upper_bound=5,
        mode="max",
        name="Peak2D",
    )


def test_nelder_mead_creation_and_simplex_size(rng):
    init = FixedInitializer(
        points=[
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        random_state=rng,
    )

    algo = NelderMead(initializer=init, random_state=rng)

    assert algo.name == "NelderMead"
    assert algo.population_size == 3
    assert algo.initializer.dimension == 2


def test_nelder_mead_one_step_keeps_valid_simplex_and_does_not_worsen_best(rng):
    init = FixedInitializer(
        points=[
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        random_state=rng,
    )
    objfunc = make_peak_objective()
    algo = NelderMead(initializer=init, random_state=rng)

    pop = algo.initialize(objfunc)
    pop.calculate_fitness()

    old_best = pop.best_fitness
    next_pop = algo.perturb(pop)

    assert next_pop.population_size == 3
    assert next_pop.dimension == 2
    assert next_pop.genotype_matrix.shape == (3, 2)
    assert np.isfinite(next_pop.genotype_matrix).all()
    assert np.isfinite(next_pop.fitness).all()
    assert next_pop.best_fitness >= old_best