"""Nelder–Mead simplex search strategy."""
from __future__ import annotations

from copy import copy
from typing import Optional

import numpy as np

from ...initializer import Initializer
from ...operator import NullOperator
from ...population import Population
from ...utils import RNGLike, check_random_state
from ..static_population import StaticPopulation


class NelderMead(StaticPopulation):
    """
    Nelder–Mead simplex search strategy.

    The framework uses maximization internally through fitness values.
    This implementation works directly with that convention.
    """

    def __init__(
        self,
        initializer: Initializer,
        name: str = "NelderMead",
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
        random_state: Optional[RNGLike] = None,
        **kwargs,
    ):
        # Nelder–Mead needs a simplex with n + 1 vertices.
        if initializer.population_size != initializer.dimension + 1:
            initializer.population_size = initializer.dimension + 1

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.sigma = float(sigma)

        random_state = check_random_state(random_state)

        super().__init__(
            initializer=initializer,
            operator=NullOperator(),
            name=name,
            random_state=random_state,
            alpha=self.alpha,
            gamma=self.gamma,
            rho=self.rho,
            sigma=self.sigma,
            **kwargs,
        )

    @staticmethod
    def _clone_population(pop: Population) -> Population:
        """Create a safe copy of a population."""
        cloned = Population(
            pop.objfunc,
            np.array(pop.genotype_matrix, dtype=float, copy=True),
            encoding=pop.encoding,
        )
        cloned.fitness = np.array(pop.fitness, copy=True)
        cloned.objective = np.array(pop.objective, copy=True)
        cloned.fitness_calculated = np.array(pop.fitness_calculated, copy=True)
        cloned.historical_best_matrix = np.array(pop.historical_best_matrix, copy=True)
        cloned.historical_best_fitness = np.array(pop.historical_best_fitness, copy=True)
        cloned.best = copy(pop.best)
        cloned.best_fitness = copy(pop.best_fitness)
        cloned.best_objective = copy(pop.best_objective)
        return cloned

    def _ordered_simplex(self, pop: Population) -> Population:
        """Return the simplex sorted from best to worst fitness."""
        order = np.argsort(-np.asarray(pop.fitness, dtype=float))
        return pop.take_selection(order)

    def _evaluate_point(
        self,
        parents: Population,
        point: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Repair and evaluate a single candidate point."""
        repaired = parents.objfunc.repair_solution(np.atleast_2d(point))
        repaired = np.asarray(repaired, dtype=float).reshape(1, -1)

        candidate = Population(parents.objfunc, repaired, encoding=parents.encoding)
        candidate.calculate_fitness()

        return (
            np.array(candidate.genotype_matrix[0], copy=True),
            float(candidate.fitness[0]),
            float(candidate.objective[0]),
        )

    def _shrink_simplex(self, simplex: Population) -> Population:
        """Shrink all vertices toward the best point."""
        ordered = self._ordered_simplex(simplex)
        genotypes = np.asarray(ordered.genotype_matrix, dtype=float)
        best = genotypes[0]

        new_genotypes = np.array(genotypes, copy=True)
        for idx in range(1, ordered.population_size):
            new_genotypes[idx] = best + self.sigma * (genotypes[idx] - best)

        new_genotypes = np.asarray(
            ordered.objfunc.repair_solution(new_genotypes),
            dtype=float,
        )

        new_pop = Population(ordered.objfunc, new_genotypes, encoding=ordered.encoding)
        new_pop.calculate_fitness()
        new_pop.update_best_from_parents(ordered)
        return new_pop

    def perturb(self, parents: Population, **kwargs) -> Population:
        """
        Build the next simplex from the current one.

        The returned population is already evaluated.
        """
        simplex = self._ordered_simplex(parents)
        genotypes = np.asarray(simplex.genotype_matrix, dtype=float)
        fitness = np.asarray(simplex.fitness, dtype=float)

        best_fit = float(fitness[0])
        second_worst_fit = float(fitness[-2])
        worst_fit = float(fitness[-1])

        centroid = np.mean(genotypes[:-1], axis=0)
        worst = genotypes[-1]

        reflected, reflected_fit, reflected_obj = self._evaluate_point(
            simplex,
            centroid + self.alpha * (centroid - worst),
        )

        # Reflection is the new best -> try expansion.
        if reflected_fit > best_fit:
            expanded, expanded_fit, expanded_obj = self._evaluate_point(
                simplex,
                centroid + self.gamma * (reflected - centroid),
            )
            if expanded_fit > reflected_fit:
                chosen_point, chosen_fit, chosen_obj = expanded, expanded_fit, expanded_obj
            else:
                chosen_point, chosen_fit, chosen_obj = reflected, reflected_fit, reflected_obj

        # Reflection improves the simplex, but is not the best point.
        elif reflected_fit > second_worst_fit:
            chosen_point, chosen_fit, chosen_obj = reflected, reflected_fit, reflected_obj

        # Outside contraction.
        elif reflected_fit > worst_fit:
            outside, outside_fit, outside_obj = self._evaluate_point(
                simplex,
                centroid + self.rho * (reflected - centroid),
            )
            if outside_fit >= reflected_fit:
                chosen_point, chosen_fit, chosen_obj = outside, outside_fit, outside_obj
            else:
                return self._shrink_simplex(simplex)

        # Inside contraction.
        else:
            inside, inside_fit, inside_obj = self._evaluate_point(
                simplex,
                centroid - self.rho * (centroid - worst),
            )
            if inside_fit > worst_fit:
                chosen_point, chosen_fit, chosen_obj = inside, inside_fit, inside_obj
            else:
                return self._shrink_simplex(simplex)

        new_pop = self._clone_population(simplex)
        new_pop.genotype_matrix[-1] = chosen_point
        new_pop.fitness[-1] = chosen_fit
        new_pop.objective[-1] = chosen_obj
        new_pop.fitness_calculated[-1] = True

        if new_pop.historical_best_fitness[-1] < chosen_fit:
            new_pop.historical_best_fitness[-1] = chosen_fit
            new_pop.historical_best_matrix[-1] = chosen_point

        new_pop.step()
        new_pop.update_best_from_parents(simplex)
        return new_pop