
"""
Base class for the Population module.

This module implements a data structure to hold the collection of solutions we are considering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Iterable, Tuple, Any, Optional, Iterator
from copy import copy
import numpy as np
from .objective_function import ObjectiveFunc
from .encoding import Encoding, DefaultEncoding
from .encodings import ParameterExtendingEncoding
from .utils import VectorLike, MatrixLike, MaskLike

logger = logging.getLogger(__name__)


@dataclass(eq=False, slots=True)
class Population:
    """Container for a set of candidate solutions and their fitness.

    A ``Population`` holds the genotype matrix, fitness and objective
    values, historical bests, and the current best individual.  It
    is the central data structure passed between components of the
    optimization loop.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        The objective function that will evaluate the population.
    genotype_matrix : ndarray
        2-D array of shape ``(N, M)`` containing the genotypes.
    encoding : Encoding, optional
        The encoding used to translate between genotype and phenotype.
        Defaults to :class:`DefaultEncoding`.
    """

    # --- User defined fields ---
    objfunc: ObjectiveFunc
    genotype_matrix: MatrixLike = field(repr=False)
    encoding: Encoding = field(default=DefaultEncoding)

    # --- Derived parameters ---
    population_size: int = field(init=False)
    dimension: int = field(init=False)

    # Fitness of each individual in the population
    fitness: VectorLike = field(init=False)
    objective: VectorLike = field(init=False)
    fitness_calculated: MaskLike = field(init=False)

    # Best solution found so far
    best: Any = field(init=False, default=None)
    best_fitness: float = field(init=False, default=None)
    best_objective: float = field(init=False, default=None)

    # Best individual in each spot of the population
    historical_best_matrix: MatrixLike = field(init=False, repr=None)
    historical_best_fitness: VectorLike = field(init=False, repr=None)

    def __post_init__(self):
        self.population_size = self.genotype_matrix.shape[0]
        self.dimension = self.genotype_matrix.shape[1]

        self.fitness = np.full(self.population_size, -np.inf)
        self.objective = np.full(self.population_size, -np.inf)
        self.fitness_calculated = np.zeros(self.population_size)

        self.best = None
        self.best_fitness = None
        self.best_objective = None

        self.historical_best_matrix = self.genotype_matrix
        self.historical_best_fitness = np.full(self.population_size, -np.inf)

    def __len__(self) -> int:
        return self.genotype_matrix.shape[0]

    def __iter__(self) -> Iterator[VectorLike]:
        for row in self.genotype_matrix:
            yield row

    def __repr__(self) -> str:
        return (
            "Population{"
            f"\n\tobjfunc = {self.objfunc.name}"
            f"\n\tgenotype_matrix = {self.genotype_matrix}"
            f"\n\tpop_size = {self.population_size}"
            f"\n\tvec_size = {self.dimension}"
            f"\n\tfitness = {self.fitness}"
            f"\n\tobjective = {self.objective}"
            f"\n\tfitness_calculated = {self.fitness_calculated}"
            f"\n\thistorical_best_matrix = {self.historical_best_matrix}"
            f"\n\thistorical_best_fitness = {self.historical_best_fitness}"
            f"\n\tbest = {self.best}"
            f"\n\tbest_fitness = {self.best_fitness}"
            f"\n\tbest_objective = {self.best_objective}"
            "\n}"
        )

    def __copy__(self) -> Population:
        copied_pop = Population(self.objfunc, copy(self.genotype_matrix), encoding=self.encoding)
        copied_pop.fitness = copy(self.fitness)
        copied_pop.objective = copy(self.objective)
        copied_pop.fitness_calculated = copy(self.fitness_calculated)
        copied_pop.historical_best_matrix = copy(self.historical_best_matrix)
        copied_pop.historical_best_fitness = copy(self.historical_best_fitness)
        copied_pop.best = copy(self.best)
        copied_pop.best_fitness = copy(self.best_fitness)
        copied_pop.best_objective = copy(self.best_objective)

        return copied_pop

    def update(self, progress: float = 0) -> Population:
        """
        Updates the best solution in the population.

        Returns
        -------
        self: Population
        """

        if self.best is None or np.any(self.best_fitness < self.fitness):
            best_idx = np.argmax(self.fitness)
            self.best = self.genotype_matrix[best_idx, :]
            self.best_fitness = self.fitness[best_idx]
            self.best_objective = self.objective[best_idx]

        return self

    def best_individual(self) -> Tuple[MatrixLike, float]:
        """Return the best genotype and its maximized fitness value.

        Returns
        -------
        best_genotype : MatrixLike
            The genotype vector of the best individual.
        best_fitness : float
            The internal fitness (always maximized).
        """

        return self.best, self.best_fitness

    def best_solution(self) -> Tuple[Any, float]:
        """Return the best decoded solution and its raw objective value.

        Returns
        -------
        solution : Any
            The decoded phenotype of the best individual.
        objective : float
            The raw objective value.
        """

        # Decode needs a matrix, so we ad a virtual dimension
        best_solution_vec = self.best[None, :]

        # The encoding returns an iterable of solutions, so we extract the first (and only) one.
        best_solution_vec = self.encoding.decode(best_solution_vec)
        if isinstance(best_solution_vec, np.ndarray) and best_solution_vec.ndim > 1:
            best_solution_vec = best_solution_vec.squeeze()
        else:
            best_solution_vec = best_solution_vec[0]

        return best_solution_vec, self.best_objective

    def get_state(self) -> dict:
        """Return a dictionary with the current population state.

        Returns
        -------
        dict
            Keys include ``genotype_matrix``, ``fitness``, ``objective``,
            historical bests, and the best individual.
        """

        data = {
            "genotype_matrix": self.genotype_matrix,
            "fitness": self.fitness,
            "objective": self.objective,
            "historical_best_matrix": self.genotype_matrix,
            "historical_best_fitness": self.historical_best_fitness,
            "best": self.best,
            "best_fitness": self.best_fitness,
            "best_objective": self.best_objective,
            "encoding": type(self.encoding).__name__,
        }

        return data

    def debug_repr(self, max_solutions: int = 5, max_vars: int = 5) -> str:
        """Return a compact string representation for debugging.

        Parameters
        ----------
        max_solutions : int, optional
            Maximum number of rows to include in the preview.
        max_vars : int, optional
            Maximum number of columns to include in the preview.

        Returns
        -------
        str
        """

        genotype_matrix = self.genotype_matrix
        shape = genotype_matrix.shape

        if genotype_matrix.size == 0:
            matrix_preview = "[]"
        elif genotype_matrix.size > max_solutions * max_vars:
            n_sol = min(max_solutions, shape[0])
            n_vars = min(max_vars, shape[1])
            preview = genotype_matrix[:n_sol, :n_vars]
            matrix_preview = f"array({preview}, shape={shape}) ... " f"(showing first {n_sol} rows, {n_vars} cols)"
        else:
            matrix_preview = f"array({genotype_matrix})"

        return (
            f"Population(\n"
            f"  objfunc={self.objfunc.name},\n"
            f"  size={self.population_size}, dims={self.dimension},\n"
            f"  fitness=[{self.fitness.min():.3e}, {self.fitness.max():.3e}],\n"
            f"  objective=[{self.objective.min():.3e}, {self.objective.max():.3e}],\n"
            f"  best_fitness={self.best_fitness:.3e},\n"
            f"  best_objective={self.best_objective:.3e},\n"
            f"  genotype_matrix={matrix_preview}\n"
            f")"
        )