import math
import random
import numpy as np
import scipy as sp
import scipy.stats
import enum
from enum import Enum
from ..utils import RAND_GEN


def cross_1p(population):
    """
    Performs a 1-point crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_points = RAND_GEN.integers(1, population.shape[1], (parents1.shape[0], 1))
    idx_matrix = np.tile(np.arange(population.shape[1]), parents1.shape[0]).reshape(parents1.shape)
    cross_mask = idx_matrix < cross_points

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def cross_2p(population):
    """
    Performs a 2-point crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_points1 = RAND_GEN.integers(1, population.shape[1] - 1, (parents1.shape[0], 1))
    cross_points2 = RAND_GEN.integers(cross_points1 + 1, parents1.shape[1], (parents1.shape[0], 1))

    idx_matrix = np.tile(np.arange(population.shape[1]), parents1.shape[0]).reshape(parents1.shape)
    slice_mask1 = idx_matrix < cross_points1
    slice_mask2 = idx_matrix >= cross_points2
    cross_mask = slice_mask1 | slice_mask2

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def cross_mp(population):
    """
    Performs a multipoint crossover between one half of the population and the rest.
    """

    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    cross_mask = RAND_GEN.uniform(0, 1, parents1.shape) < 0.5

    offspring1 = np.where(cross_mask, parents1, parents2)
    offspring2 = np.where(cross_mask, parents2, parents1)

    return np.concatenate((offspring1, offspring2))[: population.shape[0]]


def weighted_average_cross(population, alpha):
    """
    Performs a weighted average between each individual and a random member of the population.
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0]), :]

    return (1 - alpha) * population + alpha * population_shuffled


def blxalpha(population, alpha, lower_bounds=None, upper_bounds=None):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """

    parent_min = np.min(population, axis=0)
    parent_max = np.max(population, axis=0)
    diff = alpha * (parent_max - parent_min)
    low = population - diff
    high = population + diff

    return RAND_GEN.uniform(0, 1, population.shape) * (high - low) + low


def sbx(population, strength):
    """
    Performs the SBX crossing operator between two vectors.
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0])]

    beta = np.empty_like(population)
    u = RAND_GEN.uniform(0, 1, population.shape)

    _val1 = (2 * u) ** (1 / (strength + 1))
    _val2 = (0.5 * (1 - u)) ** (1 / (strength + 1))

    beta[u <= 0.5] = _val1[u <= 0.5]
    beta[u > 0.5] = _val2[u > 0.5]
    sign = RAND_GEN.choice([-1, 1], size=population.shape[0])

    return 0.5 * (population + population_shuffled) + sign * 0.5 * beta * (population - population_shuffled)


def xor_cross(population):
    """
    Applies the XOR operation between each component of individuals in the population. The crossover is performed
    between the first and second half of the population
    """

    population_shuffled = population[RAND_GEN.permutation(population.shape[0])]

    return population ^ population_shuffled


def multi_cross(population, n_indiv):
    """
    Performs a multipoint crossover between 'n_indiv' randomly chosen individuals for each member of the population.
    """

    n_indiv = np.minimum(n_indiv, population.shape[0])

    indiv_chosen = np.tile(np.arange(population.shape[0]), n_indiv).reshape((n_indiv, population.shape[0]))
    indiv_chosen = RAND_GEN.permuted(indiv_chosen, axis=1).T

    selection_mask = RAND_GEN.integers(0, n_indiv, population.shape)

    components_chosen = indiv_chosen[np.arange(indiv_chosen.shape[0])[:, None], selection_mask]

    return population[components_chosen, np.arange(population.shape[1])]


def cross_inter_avg(population, n_indiv):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population.
    """

    n_indiv = np.minimum(n_indiv, population.shape[0])

    # TODO: individuals should be chosen with replacement
    for i in range(n_indiv):
        population_shuffled = population[RAND_GEN.permutation(population.shape[0]), :]
        population += population_shuffled

    return population / n_indiv