""" """

import numpy as np
from ...initializer import Initializer
from ...utils import check_random_state


def compute_statistic(population_matrix, stat_name="mean", weights=None):
    """
    Parameters
    ----------
    population_matrix: numpy.array
        Matrix containing the set of tentative solutions.
    initializer: Initializer
        Initializer instance that handles random initializtion of the population.
    stat_name: str, optional
        Name of the statistic to use, options are "mean", "average", "median" and "std", by default "mean".
    weights: numpy.array, optional
        Vector indicating the weights to apply if "average" is selected, by default None.

    Returns
    -------
        Component-wise statistic vector.
    """

    new_population = None
    match stat_name:
        case "mean":
            new_population = np.mean(population_matrix, axis=0)
        case "average":
            if weights is None:
                weights = np.ones(population_matrix.shape[1])
            new_population = np.average(population_matrix, weights=weights, axis=0)
        case "median":
            new_population = np.median(population_matrix, axis=0)
        case "std":
            new_population = np.std(population_matrix, axis=0)

    return new_population


def random_initialize(population_matrix, initializer: Initializer, random_state=None):
    """


    Parameters
    ----------
    population_matrix: numpy.array
        Matrix containing the set of tentative solutions.
    initializer: Initializer
        Initializer instance that handles random initializtion of the population.

    Returns
    -------
        Randomly initialized population
    """

    random_population_marix = np.empty_like(population_matrix)
    for i, _ in enumerate(population_matrix):
        random_population_marix[i, :] = initializer.generate_random()

    return random_population_marix


def random_reset(population_matrix, initializer: Initializer, random_state=None, n: int = 1):
    """
    Randomly resets n components of each solution.

    Parameters
    ----------
    population_matrix: numpy.array
        Matrix containing the set of tentative solutions.
    initializer: Initializer
        Initializer instance that handles random initializtion of the population.
    n: int, optional
        Number of components to reset, by default 1

    Returns
    -------
        Population matrix with randomly changed components.
    """

    random_state = check_random_state(random_state)

    random_population_marix = np.empty_like(population_matrix)
    for i, _ in enumerate(population_matrix):
        random_population_marix[i, :] = initializer.generate_random()

    mask_pos = np.tile(
        np.arange(population_matrix.shape[1]) < n,
        population_matrix.shape[0],
    ).reshape(population_matrix.shape)

    mask_pos = random_state.permuted(mask_pos, axis=1)

    population_matrix[mask_pos] = random_population_marix

    return population_matrix
