"""

"""

import math
import numpy as np
from ...utils import RAND_GEN


def cross_1p(population_array, _fitness_array):
    """
    Performs a 1-point crossover between one half of the population and the rest.
    """

    if population_array.shape[1] == 1:
        return population_array

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_points = RAND_GEN.integers(1, population_array.shape[1], (parents1.shape[0], 1))
    idx_matrix = np.tile(np.arange(population_array.shape[1]), (parents1.shape[0], 1))
    cross_mask = idx_matrix < cross_points

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def cross_2p(population_array, _fitness_array):
    """
    Performs a 2-point crossover between one half of the population and the rest.
    """

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_points1 = RAND_GEN.integers(1, population_array.shape[1] - 1, (parents1.shape[0], 1))
    cross_points2 = RAND_GEN.integers(cross_points1 + 1, parents1.shape[1], (parents1.shape[0], 1))

    idx_matrix = np.tile(np.arange(population_array.shape[1]), (parents1.shape[0], 1))
    slice_mask1 = idx_matrix < cross_points1
    slice_mask2 = idx_matrix >= cross_points2
    cross_mask = slice_mask1 | slice_mask2

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def cross_mp(population_array, _fitness_array):
    """
    Performs a multipoint crossover between one half of the population_array and the rest.
    """

    half_size = population_array.shape[0] / 2
    parents1 = population_array[: math.ceil(half_size)]
    parents2 = population_array[math.floor(half_size) :]

    cross_mask = RAND_GEN.random(parents1.shape) < 0.5

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population_array.shape[0]]


def weighted_average_cross(population_array, _fitness_array, **kwargs):
    """
    Performs a weighted average between each individual and a random member of the population_array.
    """

    alpha = kwargs.get("alpha", 0.5)

    population_shuffled = population_array[RAND_GEN.permutation(population_array.shape[0]), :]

    return (1 - alpha) * population_array + alpha * population_shuffled


def blxalpha(population_array, _fitness_array, **kwargs):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """
    alpha = kwargs.get("alpha", 0.5)
    parent_min = kwargs.get("low", np.min(population_array, axis=0))
    parent_max = kwargs.get("up", np.max(population_array, axis=0))
    diff = alpha * (parent_max - parent_min)
    low = population_array - diff
    high = population_array + diff

    return RAND_GEN.random(population_array.shape) * (high - low) + low


def sbx(population_array, _fitness_array, **kwargs):
    """
    Performs the SBX crossing operator between two vectors.
    """

    strength = kwargs.get("F", 1)

    population_shuffled = population_array[RAND_GEN.permutation(population_array.shape[0])]

    beta = np.empty_like(population_array)
    u = RAND_GEN.random(population_array.shape)

    _val1 = (2 * u) ** (1 / (strength + 1))
    _val2 = (0.5 * (1 - u)) ** (1 / (strength + 1))

    beta[u <= 0.5] = _val1[u <= 0.5]
    beta[u > 0.5] = _val2[u > 0.5]
    sign = RAND_GEN.choice([-1, 1], size=(population_array.shape[0], 1))

    return 0.5 * (population_array + population_shuffled) + sign * 0.5 * beta * (population_array - population_shuffled)


def xor_cross(population_array, _fitness_array):
    """
    Applies the XOR operation between each component of individuals in the population_array. The crossover is performed
    between the first and second half of the population_array
    """

    population_shuffled = population_array[RAND_GEN.permutation(population_array.shape[0])]

    return population_array ^ population_shuffled


def multi_cross(population_array, _fitness_array, **kwargs):
    """
    Performs a multipoint crossover between 'n_indiv' randomly chosen individuals for each member of the population_array.
    """

    n_indiv = kwargs.get("N", 3)
    n_indiv = np.minimum(n_indiv, population_array.shape[0])

    indiv_chosen = np.tile(np.arange(population_array.shape[0]), (n_indiv, 1))
    indiv_chosen = RAND_GEN.permuted(indiv_chosen, axis=1).T

    selection_mask = RAND_GEN.integers(0, n_indiv, population_array.shape)

    components_chosen = indiv_chosen[np.arange(indiv_chosen.shape[0])[:, None], selection_mask]

    return population_array[components_chosen, np.arange(population_array.shape[1])]


def cross_inter_avg(population_array, _fitness_array, **kwargs):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population_array.
    """

    n_indiv = kwargs.get("N", 3)
    n_indiv = np.minimum(n_indiv, population_array.shape[0])

    # TODO: individuals should be chosen with replacement
    for i in range(n_indiv):
        population_shuffled = population_array[RAND_GEN.permutation(population_array.shape[0]), :]
        population_array += population_shuffled

    return population_array / n_indiv
