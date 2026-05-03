"""
Differential evolution operator implementations.
"""

import numpy as np
from ...utils import check_random_state


def differential_evolution_rand1(population_matrix, _fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/rand/1

    Parameters
    ----------
    population_matrix
        _description_
    _fitness_array
        _description_

    Returns
    -------
        _description_
    """
    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal unreachable random number
    r = np.argpartition(rand, 2, axis=1)[:, :3] # The index with lowest random number wins
    r1, r2, r3 = r.T # is of size (popsize, 3), so r.T will be (3, popsize)

    v = population_matrix[r1] + F * (population_matrix[r2] - population_matrix[r3])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best1(population_matrix, fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/best/1

    Parameters
    ----------
    population_matrix
        _description_
    fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    r_best = np.argmax(fitness_array)

    rand = random_state.random((popsize, popsize))

    np.fill_diagonal(rand, np.inf) # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2] # The index with lowest random number wins
    r1, r2 = r.T # is of size (popsize, 2), so r.T will be (2, popsize)

    v = population_matrix[r_best] + F * (population_matrix[r1] - population_matrix[r2])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_rand2(population_matrix, _fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/rand/2

    Parameters
    ----------
    population_matrix
        _description_
    _fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal unreachable random number
    r = np.argpartition(rand, 4, axis=1)[:, :5] # The index with lowest random number wins
    r1, r2, r3, r4, r5 = r.T # is of size (popsize, 5), so r.T will be (5, popsize)

    v = population_matrix[r1] + F * (population_matrix[r2] - population_matrix[r3]) + F * (population_matrix[r4] - population_matrix[r5])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best2(population_matrix, fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/best/2

    Parameters
    ----------
    population_matrix
        _description_
    fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    r_best = np.argmax(fitness_array)
    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 3, axis=1)[:, :4] # The index with lowest random number wins
    r1, r2, r3, r4 = r.T # is of size (popsize, 4), so r.T will be (4, popsize)

    v = population_matrix[r_best] + F * (population_matrix[r1] - population_matrix[r2]) + F * (population_matrix[r3] - population_matrix[r4])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_rand1(population_matrix, _fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/current-to-rand/1

    Parameters
    ----------
    population_matrix
        _description_
    _fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal unreachable random number
    r = np.argpartition(rand, 2, axis=1)[:, :3] # The index with lowest random number wins
    r1, r2, r3 = r.T # is of size (popsize, 3), so r.T will be (3, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r1] - population_matrix) + F * (population_matrix[r2] - population_matrix[r3])
    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_best1(population_matrix, fitness_array, random_state=None, F=0.8, Cr=0.9, **kwargs):
    """
    Performs the differential evolution operator DE/current-to-best/1

    Parameters
    ----------
    population_matrix
        _description_
    fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    r_best = np.argmax(fitness_array)
    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal to an unreachable random number
    rand[:, r_best] = np.inf # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2] # The index with lowest random number wins
    r1, r2 = r.T # is of size (popsize, 2), so r.T will be (2, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r_best] - population_matrix) + F * (population_matrix[r1] - population_matrix[r2])

    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_pbest1(population_matrix, fitness_array, random_state=None, F=0.8, Cr=0.9, p=0.1, **kwargs):
    """
    Performs the differential evolution operator DE/current-to-pbest/1

    Parameters
    ----------
    population_matrix
        _description_
    fitness_array
        _description_

    Returns
    -------
        _description_
    """

    random_state = check_random_state(random_state)
    popsize = population_matrix.shape[0]

    n_best_max_idx = np.ceil(population_matrix.shape[0] * p).astype(int)
    n_best_idx = np.argsort(fitness_array)[::-1][:n_best_max_idx]
    r_pbest = random_state.choice(n_best_idx, replace=True, size=population_matrix.shape[0])

    rand = random_state.random((popsize, popsize))
    np.fill_diagonal(rand, np.inf) # Set diagonal to an unreachable random number
    rand[:, r_pbest] = np.inf # Set the r_best column to an unreachable random number
    r = np.argpartition(rand, 1, axis=1)[:, :2] # The index with lowest random number wins
    r1, r2 = r.T # is of size (popsize, 2), so r.T will be (2, popsize)

    K = random_state.random((popsize, 1))
    v = population_matrix + K * (population_matrix[r_pbest] - population_matrix) + F * (population_matrix[r1] - population_matrix[r2])

    mask_pos = random_state.random(population_matrix.shape) <= Cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix
