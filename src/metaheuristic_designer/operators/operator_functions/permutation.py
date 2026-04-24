"""
"""

import math
import numpy as np
from ...utils import check_random_state


def permute_mutation(population_matrix, _fitness_array, random_state=None, **kwargs):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    random_state = check_random_state(random_state)

    n = kwargs.get("N", population_matrix.shape[0])
    n = np.clip(n, 2, population_matrix.shape[0])

    mask_pos = np.tile(np.arange(population_matrix.shape[1]), (population_matrix.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)[:, :n]

    if n == 2:
        population_matrix[np.arange(population_matrix.shape[0])[:, None], mask_pos] = population_matrix[np.arange(population_matrix.shape[0])[:, None], mask_pos][:, ::-1]
    else:
        population_matrix[np.arange(population_matrix.shape[0])[:, None], mask_pos] = random_state.permuted(
            population_matrix[np.arange(population_matrix.shape[0])[:, None], mask_pos], axis=1
        )

    return population_matrix


def roll_mutation(population_matrix, _fitness_array, random_state=None, **kwargs):
    """
    Rolls a selection of components of the vector.
    """

    random_state = check_random_state(random_state)

    n = kwargs.get("N", 1)

    roll_start = random_state.integers(0, population_matrix.shape[1] - 2, population_matrix.shape[0])
    roll_end = random_state.integers(roll_start + 2, population_matrix.shape[1] + 1, (population_matrix.shape[0]))

    def roll_individual(indiv, start, end, n):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = np.roll(indiv[start:end], n)
        return indiv_copy

    roll_vec = np.vectorize(roll_individual, signature="(m),(),(),()->(m)")
    population_matrix = roll_vec(population_matrix, roll_start, roll_end, n)
    return population_matrix


def invert_mutation(population_matrix, _fitness_array, random_state=None):
    """
    Inverts the order a selection of components of the vector.
    """

    random_state = check_random_state(random_state)

    invert_start = random_state.integers(0, population_matrix.shape[1] - 2, population_matrix.shape[0])
    invert_end = random_state.integers(invert_start + 2, population_matrix.shape[1] + 1, population_matrix.shape[0])

    def invert_individual(indiv, start, end):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = indiv[start:end][::-1]
        return indiv_copy

    invert_vec = np.vectorize(invert_individual, signature="(m),(),()->(m)")
    population_matrix = invert_vec(population_matrix, invert_start, invert_end)
    return population_matrix


def pmx(population_matrix, _fitness_array, random_state=None):

    random_state = check_random_state(random_state)

    half_size = np.ceil(population_matrix.shape[0] / 2).astype(int)

    new_population = np.empty((2 * half_size, population_matrix.shape[1]), dtype=int)
    for i in range(half_size):
        new_population[i] = pmx_single(population_matrix[i], population_matrix[2 * i], random_state=random_state)
        new_population[i + half_size] = pmx_single(population_matrix[2 * i], population_matrix[i], random_state=random_state)

    return new_population[: population_matrix.shape[0]]


def pmx_single(vector1, vector2, random_state=None):
    """
    Partially mapped crossover.

    Taken from https://github.com/cosminmarina/A1_ComputacionEvolutiva
    """

    random_state = check_random_state(random_state)

    cross_point1 = random_state.integers(0, vector1.size - 2)
    cross_point2 = random_state.integers(cross_point1 + 1, vector1.size)

    # Segmentamos
    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    # Lo que no forma parte del segmento
    remaining = vector1[~seg_mask]
    segment = vector2[seg_mask]

    # Separamos en conjunto dentro y fuera del segmento del genotipo 2
    overlap = np.isin(remaining, segment)
    conflicting = remaining[overlap]
    no_conflict = np.sort(remaining[~overlap])

    # Añadimos los elementos sin conflicto (que no están dentro del segmento del genotipo 2)
    idx_no_conflict = np.where(np.isin(vector2, no_conflict))[0]
    child[idx_no_conflict] = no_conflict

    # Tratamos conflicto
    for elem in conflicting:
        pos = elem.copy()
        while pos != -1:
            genotype_in_pos = pos
            pos = child[np.where(vector2 == genotype_in_pos)][0]
        child[np.where(vector2 == genotype_in_pos)] = elem
    return child


def order_cross(population_matrix, _fitness_array, random_state=None):

    random_state = check_random_state(random_state)

    half_size = population_matrix.shape[0] / 2
    parents1 = population_matrix[: math.ceil(half_size)]
    parents2 = population_matrix[math.floor(half_size) :]

    new_population = np.empty((2 * np.ceil(half_size).astype(int), population_matrix.shape[1]), dtype=int)
    for i in range(math.ceil(half_size)):
        new_population[i] = order_cross_single(parents1[i], parents2[i], random_state=random_state)
        new_population[i + math.ceil(half_size)] = order_cross_single(parents2[i], parents1[i], random_state=random_state)

    return new_population[: population_matrix.shape[0]]


def order_cross_single(vector1, vector2, random_state=None):

    random_state = check_random_state(random_state)

    cross_point1 = random_state.integers(0, vector1.size - 2)
    cross_point2 = random_state.integers(cross_point1, vector1.size)

    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    remianing_unused = np.setdiff1d(vector2, child)
    remianing_unused = np.roll(remianing_unused, cross_point1)

    child[~seg_mask] = remianing_unused

    return child
