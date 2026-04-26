"""
Differential evolution operator implementations.
"""

import numpy as np
from ...utils import check_random_state


def differential_evolution_rand1(population_matrix, _fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r3 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = r1 + f * (r2 - r3)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best1(population_matrix, fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r_best = population_matrix[np.argmax(fitness_array)][None, :]
    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = r_best + f * (r1 - r2)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_rand2(population_matrix, _fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r3 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r4 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r5 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = r1 + f * (r2 - r3) + f * (r4 - r5)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_best2(population_matrix, fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r_best = population_matrix[np.argmax(fitness_array)][None, :]
    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r3 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r4 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = r_best + f * (r1 - r2) + f * (r3 - r4)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_rand1(population_matrix, _fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r3 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = population_matrix + random_state.random() * (r1 - population_matrix) + f * (r2 - r3)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_best1(population_matrix, fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]

    r_best = population_matrix[np.argmax(fitness_array)][None, :]
    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = population_matrix + random_state.random() * (r_best - population_matrix) + f * (r1 - r2)
    mask_pos = random_state.random(population_matrix.shape) <= cr
    population_matrix[mask_pos] = v[mask_pos]
    return population_matrix


def differential_evolution_current_to_pbest1(population_matrix, fitness_array, random_state=None, **kwargs):
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

    f = kwargs["F"]
    cr = kwargs["Cr"]
    p = kwargs["p"]

    n_best_max_idx = np.ceil(population_matrix.shape[0] * p).astype(int)
    n_best_idx = np.argsort(fitness_array)[::-1][:n_best_max_idx]
    chosen_idx = random_state.choice(n_best_idx, replace=True, size=population_matrix.shape[0])

    r_best = population_matrix[chosen_idx]
    r1 = population_matrix[random_state.permutation(population_matrix.shape[0])]
    r2 = population_matrix[random_state.permutation(population_matrix.shape[0])]

    v = population_matrix + random_state.random() * (r_best - population_matrix) + f * (r1 - r2)
    mask_pos = random_state.random(population_matrix.shape) <= cr

    population_matrix[mask_pos] = v[mask_pos]

    return population_matrix
