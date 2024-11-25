import math
import numpy as np
import scipy as sp
from ...utils import RAND_GEN


def permute_mutation(population, n):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    n = np.clip(n, 2, population.shape[0])

    mask_pos = np.tile(np.arange(population.shape[1]), (population.shape[0], 1))
    mask_pos = RAND_GEN.permuted(mask_pos, axis=1)[:, :n]

    if n == 2:
        population[np.arange(population.shape[0])[:, None], mask_pos] = population[np.arange(population.shape[0])[:, None], mask_pos][:, ::-1]
    else:
        population[np.arange(population.shape[0])[:, None], mask_pos] = RAND_GEN.permuted(
            population[np.arange(population.shape[0])[:, None], mask_pos], axis=1
        )

    return population


def roll_mutation(population, n):
    """
    Rolls a selection of components of the vector.
    """

    roll_start = RAND_GEN.integers(0, population.shape[1] - 2, population.shape[0])
    roll_end = RAND_GEN.integers(roll_start + 2, population.shape[1] + 1, (population.shape[0]))

    def roll_individual(indiv, start, end, n):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = np.roll(indiv[start:end], n)
        return indiv_copy

    roll_vec = np.vectorize(roll_individual, signature="(m),(),(),()->(m)")
    population = roll_vec(population, roll_start, roll_end, n)
    return population


def invert_mutation(population):
    """
    Inverts the order a selection of components of the vector.
    """

    invert_start = RAND_GEN.integers(0, population.shape[1] - 2, population.shape[0])
    invert_end = RAND_GEN.integers(invert_start + 2, population.shape[1] + 1, population.shape[0])

    def invert_individual(indiv, start, end):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = indiv[start:end][::-1]
        return indiv_copy

    invert_vec = np.vectorize(invert_individual, signature="(m),(),()->(m)")
    population = invert_vec(population, invert_start, invert_end)
    return population


def pmx(population):
    half_size = population.shape[0] / 2

    new_population = np.empty((2 * np.ceil(half_size).astype(int), population.shape[1]), dtype=int)
    for i in range(math.ceil(half_size)):
        new_population[i] = pmx_single(population[i], population[2 * i])
        new_population[2 * i] = pmx_single(population[2 * i], population[i])

    return new_population


def pmx_single(vector1, vector2):
    """
    Partially mapped crossover.

    Taken from https://github.com/cosminmarina/A1_ComputacionEvolutiva
    """

    cross_point1 = RAND_GEN.integers(0, vector1.size - 2)
    cross_point2 = RAND_GEN.integers(cross_point1 + 1, vector1.size)

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


def order_cross(population):
    half_size = population.shape[0] / 2
    parents1 = population[: math.ceil(half_size)]
    parents2 = population[math.floor(half_size) :]

    new_population = np.empty((2 * np.ceil(half_size).astype(int), population.shape[1]), dtype=int)
    for i in range(math.ceil(half_size)):
        new_population[i] = order_cross_single(parents1[i], parents2[i])
        new_population[2 * i] = order_cross_single(parents2[i], parents1[i])

    return new_population


def order_cross_single(vector1, vector2):
    cross_point1 = RAND_GEN.integers(0, vector1.size - 2)
    cross_point2 = RAND_GEN.integers(cross_point1, vector1.size)

    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    remianing_unused = np.setdiff1d(vector2, child)
    remianing_unused = np.roll(remianing_unused, cross_point1)

    child[~seg_mask] = remianing_unused

    return child
