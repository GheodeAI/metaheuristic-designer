
"""
Base class for the Population module.

This module implements a data structure to hold the collection of solutions we are considering.
"""

from __future__ import annotations
import logging
from typing import Iterable, Optional
from copy import copy
import numpy as np

from metaheuristic_designer.objective_function import ObjectiveFunc

from .population import Population
from .encoding import Encoding
from .encodings import ParameterExtendingEncoding
from .utils import VectorLike, MatrixLike, MaskLike

logger = logging.getLogger(__name__)

def update_genotype(population: Population, genotype_source: MatrixLike | Population) -> Population:
    """Replace the genotype matrix.

    Parameters
    ----------
    population: Population
        Input population
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

    if genotype_matrix.shape[1] != population.dimension:
        raise ValueError("Individual vector size should not change when updating the population.")

    population.genotype_matrix = genotype_matrix
    population.population_size = genotype_matrix.shape[0]
    if len(genotype_matrix) != len(population.genotype_matrix):
        population.fitness = np.full(len(genotype_matrix), -np.inf)
        population.objective = np.full(len(genotype_matrix), -np.inf)
        population.fitness_calculated = np.zeros(len(genotype_matrix))

        population.historical_best_fitness = np.full(len(genotype_matrix), -np.inf)
        population.historical_best_matrix = copy(genotype_matrix)
        logger.debug("Genotype matrix will change size.")
    else:
        population.fitness_calculated = np.all(population.genotype_matrix == genotype_matrix, axis=1)

    logger.debug("Updated genotype matrix.")

    return population

def take_selection(population: Population, selection_idx: MaskLike) -> Population:
    """
    Takes a subset of the population given a mask.

    Parameters
    ----------
    population: Population
        Input population
    selection_idx: ndarray
        An array of indices or a mask that indicate which individuals to take from the population.

    Returns
    -------
    Population
        A copy of the population containing only the chosen individuals.
    """

    selected_pop = Population.__new__(Population) # empty copy

    selected_pop.objfunc = population.objfunc
    selected_pop.genotype_matrix = population.genotype_matrix[selection_idx, :]
    selected_pop.encoding = population.encoding

    selected_pop.population_size = selected_pop.genotype_matrix.shape[0]
    selected_pop.dimension = selected_pop.genotype_matrix.shape[1]

    selected_pop.fitness = population.fitness[selection_idx]
    selected_pop.objective = population.objective[selection_idx]
    selected_pop.fitness_calculated = population.fitness_calculated[selection_idx]

    selected_pop.best = population.best
    selected_pop.best_fitness = population.best_fitness
    selected_pop.best_objective = population.best_objective

    selected_pop.historical_best_matrix = population.historical_best_matrix[selection_idx, :]
    selected_pop.historical_best_fitness = population.historical_best_fitness[selection_idx]

    logger.debug("Taken selection from population.")

    return population

def apply_selection(population: Population, selected_pop: Population, selection_idx: MaskLike) -> Population:
    """
    Replaces the chosen individuals from the input population to the current population.

    Parameters
    ----------
    population: Population
        Input population
    selected_pop: Population
        Population where to take the individuals that will replace the ones in the population.
    selection_idx: ndarray
        An array of indices or a mask that indicate which individuals to take from the population.

    Returns
    -------
    Population
        The passed instance of the population with the indicated information replaced.
    """

    population.genotype_matrix[selection_idx, :] = selected_pop.genotype_matrix

    population.fitness[selection_idx] = selected_pop.fitness
    population.objective[selection_idx] = selected_pop.objective
    population.fitness_calculated[selection_idx] = False

    population.historical_best_matrix[selection_idx, :] = selected_pop.historical_best_matrix
    population.historical_best_fitness[selection_idx] = selected_pop.historical_best_fitness

    if population.best is None or (selected_pop.best is not None and population.best_fitness < selected_pop.best_fitness):
        population.best = selected_pop.best
        population.best_fitness = selected_pop.best_fitness
        population.best_objective = selected_pop.best_objective

    logger.debug("Applied precomputed selection from population.")

    return population

def take_slice(population: Population, mask: MaskLike) -> Population:
    """
    Takes a subset of the components in the population vectors.

    Parameters
    ----------
    population: Population
        Input population
    mask: ndarray
        An array of indices or a mask that indicate which components to take from each vector in the population.

    Returns
    -------
    Population
        A copy of the population containing the masked individuals.
    """


    sliced_pop = Population().__new__(Population)

    sliced_pop.objfunc = population.objfunc
    sliced_pop.genotype_matrix = population.genotype_matrix[:, mask]
    sliced_pop.encoding = population.encoding

    sliced_pop.population_size = sliced_pop.genotype_matrix.shape[0]
    sliced_pop.dimension = sliced_pop.genotype_matrix.shape[1]

    sliced_pop.fitness = copy(population.fitness)
    sliced_pop.objective = copy(population.objective)
    sliced_pop.fitness_calculated = copy(population.fitness_calculated)

    sliced_pop.best = copy(population.best)
    sliced_pop.best_fitness = copy(population.best_fitness)
    sliced_pop.best_objective = copy(population.best_objective)

    sliced_pop.historical_best_matrix = population.historical_best_matrix[:, mask]
    sliced_pop.historical_best_fitness = copy(population.historical_best_fitness)

    logger.debug("Taken slice from population.")

    return sliced_pop

def apply_slice(population: Population, sliced_pop: Population, mask: MaskLike) -> Population:
    """
    Apply the values of the population to a subset of the components of the population vectors.

    Parameters
    ----------
    population: Population
        Input population
    sliced_pop: Population
        Population where to take the individuals from which we will take the components that will replace the ones in the
        current population.
    mask: ndarray
        An array of indices or a mask that indicate which components to take from each vector in the population.

    Returns
    -------
    Population
        The passed instance of the population with the indicated information replaced.
    """

    population.genotype_matrix[:, mask] = sliced_pop.genotype_matrix
    population.historical_best_matrix[:, mask] = sliced_pop.historical_best_matrix
    population.fitness_calculated[:] = False

    if population.best is None or (sliced_pop.best is not None and population.best_fitness < sliced_pop.best_fitness):
        population.best = sliced_pop.best
        population.best_fitness = sliced_pop.best_fitness
        population.best_objective = sliced_pop.best_objective

    logger.debug("Applied precomputed slice from population.")

    return population

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


    joined_pop = Population().__new__(Population)

    joined_pop.objfunc = population1.objfunc
    joined_pop.genotype_matrix = np.concatenate((population1.genotype_matrix, population2.genotype_matrix), axis=0)
    joined_pop.encoding = population1.encoding

    joined_pop.population_size = joined_pop.genotype_matrix.shape[0]
    joined_pop.dimension = joined_pop.genotype_matrix.shape[1]

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

    joined_pop.historical_best_matrix = np.concatenate((population1.historical_best_matrix, population2.historical_best_matrix), axis=0)
    joined_pop.historical_best_fitness = np.concatenate((population1.historical_best_fitness, population2.historical_best_fitness))

    logger.debug("Merged two populations into one.")

    return joined_pop


def sort_population(population: Population) -> Population:
    """
    Sorts the individuals by fitness.

    Parameters
    ----------
    population: Population
        Input population

    Returns
    -------
    Population
        Population sorted by fitness
    """

    fitness_order = np.argsort(population.fitness)

    population.genotype_matrix = population.genotype_matrix[fitness_order, :]
    population.historical_best_matrix = population.historical_best_matrix[fitness_order, :]
    population.historical_best_fitness = population.historical_best_fitness[fitness_order]
    population.fitness_calculated = population.fitness_calculated[fitness_order]
    population.fitness = population.fitness[fitness_order]
    population.objective = population.objective[fitness_order]

    logger.debug("Sorted population.")

    return population

def update_best(population: Population, donor_population: Population) -> Population:
    """Update the best solution if a better one exists in the donor population.

    Parameters
    ----------
    population: Population
        Input population
    donor_population : Population
        Population whose best individual may improve the current one.

    Returns
    -------
    Population
        input population with possibly updated ``best``, ``best_fitness``,
        and ``best_objective``.
    """

    if population.best is None or (donor_population.best is not None and population.best_fitness < donor_population.best_fitness):
        population.best = donor_population.best
        population.best_fitness = donor_population.best_fitness
        population.best_objective = donor_population.best_objective

    return population

def repeat(population: Population, amount: int = 2) -> Population:
    """
    Duplicates the individuals of the population.

    Parameters
    ----------
    population: Population
        Input population
    amount: int, optional
        The amount of times to repeat the individuals in the population.

    Returns
    -------
    Population
    """


    repeated_pop = Population().__new__(Population)

    repeated_pop.objfunc = population.objfunc
    repeated_pop.genotype_matrix = np.tile(population.genotype_matrix, (amount, 1))
    repeated_pop.encoding = population.encoding

    repeated_pop.population_size = repeated_pop.genotype_matrix.shape[0]
    repeated_pop.dimension = repeated_pop.genotype_matrix.shape[1]
    
    repeated_pop.fitness = np.tile(population.fitness, amount)
    repeated_pop.objective = np.tile(population.objective, amount)
    repeated_pop.fitness_calculated = np.tile(population.fitness_calculated, amount)

    repeated_pop.best = population.best
    repeated_pop.best_fitness = population.best_fitness
    repeated_pop.best_objective = population.best_objective

    repeated_pop.historical_best_fitness = population.historical_best_fitness
    repeated_pop.historical_best_matrix = population.historical_best_matrix

    logger.debug("Added %d copies of each individual to the population", amount)

    return repeated_pop

def update_fitness(population: Population, objfunc: ObjectiveFunc) -> Population:
    """
    Calculates the fitness of the individual if it has not been calculated before

    Parameters
    ----------
    parallel: bool, optional
        Whether to evaluate the individuals in the population in parallel.
    threads: int, optional
        Number of processes to use at once if calculating the fitness in parallel.

    Returns
    -------
    Population
    """

    prev_fitness = copy(population.fitness)

    # Objective values and fitness values are modified in place after the call
    new_fitness = objfunc.fitness(population)

    improved_mask = prev_fitness < population.fitness

    population.fitness = new_fitness
    population.historical_best_fitness[improved_mask] = population.fitness[improved_mask]
    population.historical_best_matrix[improved_mask, :] = population.genotype_matrix[improved_mask, :]

    if population.best is None or np.any(population.fitness > population.best_fitness):
        best_idx = np.argmax(population.fitness)
        population.best = population.genotype_matrix[best_idx]
        population.best_fitness = population.fitness[best_idx]
        population.best_objective = population.objective[best_idx]

    logger.debug("Updated the fitness of the individuals.")

    return population

def repair_solutions(self) -> Population:
    """
    Repairs the solutions in the population.

    Returns
    -------
    self: Population
    """

    self.genotype_matrix = self.objfunc.repair_solution(self.genotype_matrix)
    return self

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