"""

"""

import math
import numpy as np
from ...utils import check_random_state


def one_point_crossover(population_array, _fitness_array, random_state=None):
    """
    Performs a 1-point crossover between one half of the population and the rest.
    """

    random_state = check_random_state(random_state)

    if population_array.shape[1] == 1:
        return population_array

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_points = random_state.integers(1, population_array.shape[1], (parents1.shape[0], 1))
    idx_matrix = np.tile(np.arange(population_array.shape[1]), (parents1.shape[0], 1))
    cross_mask = idx_matrix < cross_points

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def two_point_crossover(population_array, _fitness_array, random_state=None):
    """
    Performs a 2-point crossover between one half of the population and the rest.
    """

    random_state = check_random_state(random_state)

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_points1 = random_state.integers(1, population_array.shape[1] - 1, (parents1.shape[0], 1))
    cross_points2 = random_state.integers(cross_points1 + 1, parents1.shape[1], (parents1.shape[0], 1))

    idx_matrix = np.tile(np.arange(population_array.shape[1]), (parents1.shape[0], 1))
    slice_mask1 = idx_matrix < cross_points1
    slice_mask2 = idx_matrix >= cross_points2
    cross_mask = slice_mask1 | slice_mask2

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def uniform_crossover(population_array, _fitness_array, random_state=None):
    """
    Performs an uniform crossover between one half of the population_array and the rest.
    """

    random_state = check_random_state(random_state)

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_mask = random_state.random(parents1.shape) < 0.5

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def multiparent_discrete_crossover(population_array, _fitness_array, N=3, random_state=None):
    """
    Performs a multipoint crossover between 'n_indiv' randomly chosen individuals for each member of the population_array.
    """

    random_state = check_random_state(random_state)

    n_indiv = np.minimum(N, population_array.shape[0])

    indiv_chosen = np.tile(np.arange(population_array.shape[0]), (n_indiv, 1))
    indiv_chosen = random_state.permuted(indiv_chosen, axis=1).T

    selection_mask = random_state.integers(0, n_indiv, population_array.shape)

    components_chosen = indiv_chosen[np.arange(indiv_chosen.shape[0])[:, None], selection_mask]

    return population_array[components_chosen, np.arange(population_array.shape[1])]



def averaged_crossover(population_array, _fitness_array, alpha=0.5, random_state=None):
    """
    Performs a weighted average between each individual and a random member of the population_array.
    """

    random_state = check_random_state(random_state)

    population_shuffled = population_array[random_state.permutation(population_array.shape[0]), :]

    return (1 - alpha) * population_array + alpha * population_shuffled


def blx_alpha_crossover(population_array, _fitness_array, alpha=0.5, low=None, high=None, random_state=None):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """
    random_state = check_random_state(random_state)

    if low is None:
        low = np.min(population_array, axis=0)
    if high is None:
        high = np.max(population_array, axis=0)

    diff = alpha * (high - low)
    low = population_array - diff
    high = population_array + diff

    return random_state.random(population_array.shape) * (high - low) + low


def sbx_crossover(population_array, _fitness_array, F=1, random_state=None):
    """
    Performs the SBX crossing operator between two vectors.
    """

    random_state = check_random_state(random_state)

    population_shuffled = population_array[random_state.permutation(population_array.shape[0])]

    beta = np.empty_like(population_array)
    u = random_state.random(population_array.shape)

    _val1 = (2 * u) ** (1 / (F + 1))
    _val2 = (0.5 * (1 - u)) ** (1 / (F + 1))

    beta[u <= 0.5] = _val1[u <= 0.5]
    beta[u > 0.5] = _val2[u > 0.5]
    sign = random_state.choice([-1, 1], size=(population_array.shape[0], 1))

    return 0.5 * (population_array + population_shuffled) + sign * 0.5 * beta * (population_array - population_shuffled)


def bitwise_xor_crossover(population_array, _fitness_array, random_state=None):
    """
    Applies the XOR operation between each component of individuals in the population_array. The crossover is performed
    between the first and second half of the population_array
    """

    random_state = check_random_state(random_state)

    population_shuffled = population_array[random_state.permutation(population_array.shape[0])]

    return population_array ^ population_shuffled



def cross_inter_avg(population_array, _fitness_array, N=3, random_state=None):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population_array.
    """

    random_state = check_random_state(random_state)

    n_indiv = np.minimum(N, population_array.shape[0])

    # TODO: individuals should be chosen with replacement
    for i in range(n_indiv):
        population_shuffled = population_array[random_state.permutation(population_array.shape[0]), :]
        population_array += population_shuffled

    return population_array / n_indiv
