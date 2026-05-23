"""
Differential evolution operator implementations.
"""

import logging
from typing import Optional
from ...utils import MatrixLike, RNGLike, VectorLike, check_random_state
import numpy as np

logger = logging.getLogger(__name__)


def differential_evolution_rand1(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/rand/1 mutation and binomial crossover.

    For each target vector, three distinct random individuals are
    chosen.  A donor vector is formed as
    ``x_r1 + F * (x_r2 - x_r3)``.  Components are then taken from
    the donor with probability *Cr* and from the target otherwise.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population, shape ``(N, M)``.
    fitness_array : VectorLike
        Fitness values (used only by the ``/best/`` variants).
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 4:
        raise ValueError("Cannot apply DE/rand/1 with a population size smaller than 4.")

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal unreachable random number
    r = np.argpartition(rand, 2, axis=1)[:, :3]  # The index with lowest random number wins
    r1, r2, r3 = r.T  # is of size (popsize, 3), so r.T will be (3, popsize)

    v = population_matrix[r1] + F * (population_matrix[r2] - population_matrix[r3])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best1(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/best/1 mutation and binomial crossover.

    The donor is formed using the best individual as the base:
    ``x_best + F * (x_r1 - x_r2)`` where *r1* and *r2* are
    distinct and different from *best*.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values; the index of the maximum is used as *best*.
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 3:
        raise ValueError("Cannot apply DE/best/1 with a population size smaller than 3.")

    r_best = np.argmax(fitness_array)

    rand = random_state.random((popsize, popsize))

    np.fill_diagonal(rand, np.inf)  # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf  # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2]  # The index with lowest random number wins
    r1, r2 = r.T  # is of size (popsize, 2), so r.T will be (2, popsize)

    v = population_matrix[r_best] + F * (population_matrix[r1] - population_matrix[r2])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_rand2(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/rand/2 mutation and binomial crossover.

    Two difference vectors are used:
    ``x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values (unused in this variant).
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 6:
        raise ValueError("Cannot apply DE/rand/2 with a population size smaller than 6.")

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal unreachable random number
    r = np.argpartition(rand, 4, axis=1)[:, :5]  # The index with lowest random number wins
    r1, r2, r3, r4, r5 = r.T  # is of size (popsize, 5), so r.T will be (5, popsize)

    v = population_matrix[r1] + F * (population_matrix[r2] - population_matrix[r3]) + F * (population_matrix[r4] - population_matrix[r5])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best2(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/best/2 mutation and binomial crossover.

    The best individual is the base, and two difference vectors
    are added:
    ``x_best + F*(x_r1 - x_r2) + F*(x_r3 - x_r4)``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values; the best is the one with highest fitness.
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 5:
        raise ValueError("Cannot apply DE/best/2 with a population size smaller than 5.")

    r_best = np.argmax(fitness_array)
    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf  # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 3, axis=1)[:, :4]  # The index with lowest random number wins
    r1, r2, r3, r4 = r.T  # is of size (popsize, 4), so r.T will be (4, popsize)

    v = population_matrix[r_best] + F * (population_matrix[r1] - population_matrix[r2]) + F * (population_matrix[r3] - population_matrix[r4])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_rand1(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/current-to-rand/1 mutation and binomial crossover.

    Each target vector *x_i* is combined with a random individual
    and a difference vector:
    ``x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)``,
    where *K* is drawn uniformly in [0,1] per individual.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values (unused).
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 4:
        raise ValueError("Cannot apply DE/current-to-rand/1 with a population size smaller than 4.")

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal unreachable random number
    r = np.argpartition(rand, 2, axis=1)[:, :3]  # The index with lowest random number wins
    r1, r2, r3 = r.T  # is of size (popsize, 3), so r.T will be (3, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r1] - population_matrix) + F * (population_matrix[r2] - population_matrix[r3])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_best1(
    population_matrix: MatrixLike, fitness_array: VectorLike, random_state: Optional[RNGLike] = None, F: float = 0.8, Cr: float = 0.9, **kwargs
) -> MatrixLike:
    """
    DE/current-to-best/1 mutation and binomial crossover.

    ``x_i + K*(x_best - x_i) + F*(x_r1 - x_r2)``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values; the best is the one with highest fitness.
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 3:
        raise ValueError("Cannot apply DE/current-to-best/1 with a population size smaller than 3.")

    r_best = np.argmax(fitness_array)
    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf  # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2]  # The index with lowest random number wins
    r1, r2 = r.T  # is of size (popsize, 2), so r.T will be (2, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r_best] - population_matrix) + F * (population_matrix[r1] - population_matrix[r2])

    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_pbest1(
    population_matrix: MatrixLike,
    fitness_array: VectorLike,
    random_state: Optional[RNGLike] = None,
    F: float = 0.8,
    Cr: float = 0.9,
    p: float = 0.1,
    **kwargs,
) -> MatrixLike:
    """
    DE/current-to-pbest/1 mutation and binomial crossover.

    Instead of the single best, one of the top ``p*N`` individuals
    is randomly chosen as *pbest*:
    ``x_i + K*(x_pbest - x_i) + F*(x_r1 - x_r2)``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population.
    fitness_array : VectorLike
        Fitness values; the top *p* fraction is selected.
    random_state : RNGLike, optional
        Random number generator.
    F : float, optional
        Scale factor (default 0.8).
    Cr : float, optional
        Crossover probability (default 0.9).
    p : float, optional
        Fraction of the population considered as elite (default 0.1).

    Returns
    -------
    MatrixLike
        Trial population of the same shape.
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    if popsize < 3:
        raise ValueError("Cannot apply DE/current-to-pbest/1 with a population size smaller than 3.")

    n_best_max_idx = np.ceil(population_matrix.shape[0] * p).astype(int)
    n_best_idx = np.argsort(fitness_array)[::-1][:n_best_max_idx]
    r_pbest = random_state.choice(n_best_idx, replace=True, size=population_matrix.shape[0])

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf)  # Set diagonal to an unreachable random number
    rand[:, r_pbest] = np.inf  # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2]  # The index with lowest random number wins
    r1, r2 = r.T  # is of size (popsize, 2), so r.T will be (2, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r_pbest] - population_matrix) + F * (population_matrix[r1] - population_matrix[r2])

    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix
