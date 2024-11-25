import numpy as np
from ...utils import RAND_GEN


def DE_rand1(population, F, CR):
    """
    Performs the differential evolution operator DE/rand/1
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]

    v = r1 + F * (r2 - r3)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_best1(population, fitness, F, CR):
    """
    Performs the differential evolution operator DE/best/1
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = r_best + F * (r1 - r2)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_rand2(population, F, CR):
    """
    Performs the differential evolution operator DE/rand/2
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]
    r5 = population[RAND_GEN.permutation(population.shape[0])]

    v = r1 + F * (r2 - r3) + F * (r4 - r5)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_best2(population, fitness, F, CR):
    """
    Performs the differential evolution operator DE/best/2
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]

    v = r_best + F * (r1 - r2) + F * (r3 - r4)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_rand1(population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-rand/1
    """

    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]
    r3 = population[RAND_GEN.permutation(population.shape[0])]
    r4 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.random() * (r1 - population) + F * (r2 - r3)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_best1(population, fitness, F, CR):
    """
    Performs the differential evolution operator DE/current-to-best/1
    """

    r_best = population[np.argmax(fitness)][None, :]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.random() * (r_best - population) + F * (r1 - r2)
    mask_pos = RAND_GEN.random(population.shape) <= CR
    population[mask_pos] = v[mask_pos]
    return population


def DE_current_to_pbest1(population, fitness, F, CR, P):
    """
    Performs the differential evolution operator DE/current-to-pbest/1
    """

    n_best_max_idx = np.ceil(population.shape[0] * P).astype(int)
    n_best_idx = np.argsort(fitness)[::-1][:n_best_max_idx]
    chosen_idx = RAND_GEN.choice(n_best_idx, replace=True, size=population.shape[0])

    r_best = population[chosen_idx]
    r1 = population[RAND_GEN.permutation(population.shape[0])]
    r2 = population[RAND_GEN.permutation(population.shape[0])]

    v = population + RAND_GEN.random() * (r_best - population) + F * (r1 - r2)
    mask_pos = RAND_GEN.random(population.shape) <= CR

    population[mask_pos] = v[mask_pos]

    return population
