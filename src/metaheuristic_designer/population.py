"""
Base class for the Population module.

This module implements a data structure to hold the collection of solutions we are considering.
"""

from __future__ import annotations
import logging
from typing import Iterable, Tuple, Any, Optional, Iterator, TYPE_CHECKING
from copy import copy
import numpy as np
from .encoding import Encoding, DefaultEncoding
from .encodings import ParameterExtendingEncoding
from .utils import VectorLike, MatrixLike, MaskLike


logger = logging.getLogger(__name__)


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

    def __init__(self, genotype_matrix: MatrixLike, encoding: Optional[Encoding] = None):
        # Population of solutions
        self.genotype_matrix = genotype_matrix

        # Size of the population
        self.population_size = genotype_matrix.shape[0]
        self.dimension = genotype_matrix.shape[1]

        # Fitness of each individual in the population
        self.fitness = np.full(self.population_size, -np.inf)
        self.objective = np.full(self.population_size, -np.inf)
        self.fitness_calculated = np.zeros(self.population_size)

        # Best solution found so far
        self.best = None
        self.best_fitness = None
        self.best_objective = None

        # Best individual in each spot of the population
        self.historical_best_matrix = genotype_matrix
        self.historical_best_fitness = np.full(self.population_size, -np.inf)

        # Encoding to use
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        self.index = 0

    def __len__(self) -> int:
        return self.population_size

    def __iter__(self) -> Iterator[VectorLike]:
        for row in self.genotype_matrix:
            yield row

    def __repr__(self) -> str:
        return (
            "Population{"
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
        copied_pop = Population(copy(self.genotype_matrix), encoding=self.encoding)
        copied_pop.fitness = copy(self.fitness)
        copied_pop.objective = copy(self.objective)
        copied_pop.fitness_calculated = copy(self.fitness_calculated)
        copied_pop.historical_best_matrix = copy(self.historical_best_matrix)
        copied_pop.historical_best_fitness = copy(self.historical_best_fitness)
        copied_pop.best = copy(self.best)
        copied_pop.best_fitness = copy(self.best_fitness)
        copied_pop.best_objective = copy(self.best_objective)

        return copied_pop

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

    def update_genotype(self, genotype_source: MatrixLike | Population, update_fitness_mask: bool = True) -> Population:
        """Replace the genotype matrix.

        Parameters
        ----------
        genotype_source : ndarray or Population
            New genotypes.  If a ``Population`` is given, its genotype
            matrix is used.

        Returns
        -------
        Population
            ``self``, with updated genotypes and, if the size changed,
            re-initialized fitness and historical bests.
        """

        if isinstance(genotype_source, Population):
            genotype_matrix = genotype_source.genotype_matrix
        else:
            genotype_matrix = genotype_source

        if genotype_matrix.shape[1] != self.dimension:
            raise ValueError("Individual vector size should not change when updating the population.")

        if len(genotype_matrix) != len(self.genotype_matrix):
            self.fitness = np.full(len(genotype_matrix), -np.inf)
            self.objective = np.full(len(genotype_matrix), -np.inf)
            self.fitness_calculated = np.zeros(len(genotype_matrix))
            self.historical_best_fitness = np.full(len(genotype_matrix), -np.inf)
            self.historical_best_matrix = copy(genotype_matrix)
            logger.debug("Genotype matrix will change size.")
        elif update_fitness_mask:
            self.fitness_calculated = np.all(self.genotype_matrix == genotype_matrix, axis=1)
        self.genotype_matrix = genotype_matrix
        self.population_size = genotype_matrix.shape[0]

        logger.debug("Updated genotype matrix.")

        return self

    def take_selection(self, selection_idx: MaskLike) -> Population:
        """
        Takes a subset of the population given a mask.

        Parameters
        ----------
        selection_idx: ndarray
            An array of indices or a mask that indicate which individuals to take from the population.

        Returns
        -------
        selected_population: Population
            A copy of the population containing only the chosen individuals.
        """

        selected_genotype_matrix = copy(self.genotype_matrix[selection_idx, :])

        selected_pop = Population(selected_genotype_matrix, encoding=self.encoding)
        selected_pop.fitness = copy(self.fitness[selection_idx])
        selected_pop.objective = copy(self.objective[selection_idx])
        selected_pop.fitness_calculated = copy(self.fitness_calculated[selection_idx])
        selected_pop.historical_best_matrix = copy(self.historical_best_matrix[selection_idx, :])
        selected_pop.historical_best_fitness = copy(self.historical_best_fitness[selection_idx])
        selected_pop.best = copy(self.best)
        selected_pop.best_fitness = copy(self.best_fitness)
        selected_pop.best_objective = copy(self.best_objective)

        logger.debug("Taken selection from population.")

        return selected_pop

    def apply_selection(self, selected_pop: Population, selection_idx: MaskLike) -> Population:
        """
        Replaces the chosen individuals from the input population to the current population.

        Parameters
        ----------
        selected_pop: Population
            Population where to take the individuals that will replace the ones in the population.
        selection_idx: ndarray
            An array of indices or a mask that indicate which individuals to take from the population.

        Returns
        -------
        self: Population
        """

        self.genotype_matrix[selection_idx, :] = selected_pop.genotype_matrix
        self.fitness_calculated[selection_idx] = False
        self.fitness[selection_idx] = selected_pop.fitness
        self.objective[selection_idx] = selected_pop.objective

        self.historical_best_matrix[selection_idx, :] = selected_pop.historical_best_matrix
        self.historical_best_fitness[selection_idx] = selected_pop.historical_best_fitness

        if self.best is None or (selected_pop.best is not None and self.best_fitness < selected_pop.best_fitness):
            self.best = selected_pop.best
            self.best_fitness = selected_pop.best_fitness
            self.best_objective = selected_pop.best_objective

        logger.debug("Applied precomputed selection from population.")

        return self

    def take_slice(self, mask: MaskLike) -> Population:
        """
        Takes a subset of the components in the population vectors.

        Parameters
        ----------
        mask: ndarray
            An array of indices or a mask that indicate which components to take from each vector in the population.

        Returns
        -------
        sliced_population: Population
            A copy of the population containing the masked individuals.
        """

        sliced_genotype_matrix = copy(self.genotype_matrix[:, mask])

        sliced_pop = Population(sliced_genotype_matrix, encoding=self.encoding)
        sliced_pop.dimension = sliced_genotype_matrix.shape[1]
        sliced_pop.historical_best_matrix = copy(self.historical_best_matrix[:, mask])
        sliced_pop.historical_best_fitness = copy(self.historical_best_fitness)
        sliced_pop.fitness_calculated = copy(self.fitness_calculated)
        sliced_pop.fitness = copy(self.fitness)
        sliced_pop.objective = copy(self.objective)
        sliced_pop.best = copy(self.best)
        sliced_pop.best_fitness = copy(self.best_fitness)
        sliced_pop.best_objective = copy(self.best_objective)

        logger.debug("Taken slice from population.")

        return sliced_pop

    def apply_slice(self, sliced_pop: Population, mask: MaskLike) -> Population:
        """
        Apply the values of the population to a subset of the components of the population vectors.

        Parameters
        ----------
        sliced_pop: Population
            Population where to take the individuals from which we will take the components that will replace the ones in the
            current population.
        mask: ndarray
            An array of indices or a mask that indicate which components to take from each vector in the population.

        Returns
        -------
        self: Population
        """

        self.genotype_matrix[:, mask] = sliced_pop.genotype_matrix
        self.fitness_calculated[:] = False

        if self.best is None or (sliced_pop.best is not None and self.best_fitness < sliced_pop.best_fitness):
            self.best = sliced_pop.best
            self.best_fitness = sliced_pop.best_fitness
            self.best_objective = sliced_pop.best_objective

        logger.debug("Applied precomputed slice from population.")

        return self

    @staticmethod
    def join_populations(population1: Population, population2: Population) -> Population:
        """Concatenate two populations into a new one.

        Parameters
        ----------
        population1 : Population
            First population.
        population2 : Population
            Second population.

        Returns
        -------
        Population
            A new population containing all individuals from both inputs.
        """

        joined_genotype_matrix = np.concatenate((population1.genotype_matrix, population2.genotype_matrix), axis=0)

        joined_pop = Population(joined_genotype_matrix, encoding=population1.encoding)
        joined_pop.historical_best_matrix = np.concatenate((population1.historical_best_matrix, population2.historical_best_matrix), axis=0)
        joined_pop.historical_best_fitness = np.concatenate((population1.historical_best_fitness, population2.historical_best_fitness))
        joined_pop.fitness_calculated = np.concatenate((population1.fitness_calculated, population2.fitness_calculated))
        joined_pop.fitness = np.concatenate((population1.fitness, population2.fitness))
        joined_pop.objective = np.concatenate((population1.objective, population2.objective))

        if population1.best is None or (population2.best is not None and population1.best_fitness > population2.best_fitness):
            joined_pop.best = population1.best
            joined_pop.best_fitness = population1.best_fitness
            joined_pop.best_objective = population1.best_objective
        else:
            joined_pop.best = population2.best
            joined_pop.best_fitness = population2.best_fitness
            joined_pop.best_objective = population2.best_objective

        logger.debug("Merged two populations into one.")

        return joined_pop

    def join(self, other_population: Population) -> Population:
        """
        Adds to the current population the individuals of the input population.

        Parameters
        ----------
        other_population: Population
            Population that will be concatenated with the current one.


        Returns
        -------
        joined_populations: Population
            A population containing both the individuals from the current population and the ones from the input population.
        """

        self.genotype_matrix = np.concatenate((self.genotype_matrix, other_population.genotype_matrix), axis=0)
        self.population_size += other_population.genotype_matrix.shape[0]
        self.historical_best_matrix = np.concatenate((self.historical_best_matrix, other_population.historical_best_matrix), axis=0)
        self.historical_best_fitness = np.concatenate((self.historical_best_fitness, other_population.historical_best_fitness))
        self.fitness_calculated = np.concatenate((self.fitness_calculated, other_population.fitness_calculated), axis=0)
        self.fitness = np.concatenate((self.fitness, other_population.fitness))
        self.objective = np.concatenate((self.objective, other_population.objective))

        if self.best is None or (other_population.best is not None and self.best_fitness < other_population.best_fitness):
            self.best = other_population.best
            self.best_fitness = other_population.best_fitness
            self.best_objective = other_population.best_objective

        logger.debug("Merged one population into the current one.")

        return self

    def sort_population(self) -> Population:
        """
        Sorts the individuals by fitness.

        Returns
        -------
        self: Population
        """

        fitness_order = np.argsort(self.fitness)

        self.genotype_matrix = self.genotype_matrix[fitness_order, :]
        self.historical_best_matrix = self.historical_best_matrix[fitness_order, :]
        self.historical_best_fitness = self.historical_best_fitness[fitness_order]
        self.fitness_calculated = self.fitness_calculated[fitness_order]
        self.fitness = self.fitness[fitness_order]
        self.objective = self.objective[fitness_order]

        logger.debug("Sorted population.")

        return self

    def update_best_from_parents(self, parents: Population) -> Population:
        """Update the best solution if a better one exists in *parents*.

        Parameters
        ----------
        parents : Population
            Population whose best individual may improve the current one.

        Returns
        -------
        Population
            ``self``, with possibly updated ``best``, ``best_fitness``,
            and ``best_objective``.
        """

        if self.best is None or (parents.best is not None and self.best_fitness < parents.best_fitness):
            self.best = parents.best
            self.best_fitness = parents.best_fitness
            self.best_objective = parents.best_objective

        return self

    def step(self, _progress: float = 0) -> Population:
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

    def repeat(self, amount: int = 2) -> Population:
        """
        Duplicates the individuals of the population.

        Parameters
        ----------
        amount: int, optional
            The amount of times to repeat the individuals in the population.

        Returns
        -------
        repeated_population: Population
        """

        genotype_matrix = np.tile(self.genotype_matrix, (amount, 1))
        fitness_calculated = np.tile(self.fitness_calculated, (amount))
        fitness = np.tile(self.fitness, (amount))
        objective = np.tile(self.objective, (amount))
        best = self.best
        best_fitness = self.best_fitness
        best_objective = self.best_objective

        new_population = Population(genotype_matrix, encoding=self.encoding)
        new_population.fitness_calculated = fitness_calculated
        new_population.fitness = fitness
        new_population.objective = objective
        new_population.best = best
        new_population.best_fitness = best_fitness
        new_population.best_objective = best_objective

        logger.debug("Added %d copies of each individual to the population", amount)

        return new_population

    def decode(self, encoding: Optional[Encoding] = None) -> Iterable:
        """
        Return the population passed through the decoding function defined in the encoding.

        Returns
        -------
        decoded_population: Any
        """

        if encoding is None:
            encoding = self.encoding

        return self.encoding.decode(self.genotype_matrix)

    def decode_params(self, encoding: Optional[Encoding] = None) -> Iterable:
        """Decode the auxiliary parameters stored in the genotype.

        Only works with :class:`ParameterExtendingEncoding`.

        Parameters
        ----------
        encoding : Encoding, optional
            Encoding to use; defaults to ``self.encoding``.

        Returns
        -------
        dict or None
            Dictionary of parameter arrays, or ``None`` if the encoding
            does not support extended parameters.
        """

        if encoding is None:
            encoding = self.encoding

        if isinstance(self.encoding, ParameterExtendingEncoding):
            return self.encoding.decode_params(self.genotype_matrix)
        else:
            return None

    def encode(self, encoding: Optional[Encoding] = None) -> MatrixLike:
        """Encode the current population using the given encoding.

        Parameters
        ----------
        encoding : Encoding, optional
            Encoding to use; defaults to ``self.encoding``.

        Returns
        -------
        MatrixLike
            The encoded genotype matrix.
        """

        if encoding is None:
            encoding = self.encoding

        return encoding.encode(self.genotype_matrix)

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
            f"  size={self.population_size}, dims={self.dimension},\n"
            f"  fitness=[{self.fitness.min():.3e}, {self.fitness.max():.3e}],\n"
            f"  objective=[{self.objective.min():.3e}, {self.objective.max():.3e}],\n"
            f"  best_fitness={self.best_fitness:.3e},\n"
            f"  best_objective={self.best_objective:.3e},\n"
            f"  genotype_matrix={matrix_preview}\n"
            f")"
        )
