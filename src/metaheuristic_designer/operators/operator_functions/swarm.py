""" """

import numpy as np
import scipy.spatial.distance as sp_dist
from ...population import Population
from ...utils import check_random_state


def pso_operator(
    population_matrix: np.array,
    population_speed: np.array,
    historical_best: np.array,
    global_best: np.array,
    random_state=None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
):
    """
    Performs a step of the Particle Swarm algorithm
    """

    random_state = check_random_state(random_state)

    c1 = c1 * random_state.random(population_matrix.shape)
    c2 = c2 * random_state.random(population_matrix.shape)

    speed = w * population_speed + c1 * (historical_best - population_matrix) + c2 * (global_best - population_matrix)

    return population_matrix + speed, speed


def firefly_operator(population_matrix, fitness_array, random_state=None, alpha_0=0.2, beta_0=1.0, delta=1.0, gamma=1.0):
    """
    Performs a step of the Firefly algorithm
    """

    random_state = check_random_state(random_state)

    n_components = population_matrix.shape[1]
    dist_matrix_flat = sp_dist.pdist(population_matrix)
    dist_matrix = sp_dist.squareform(dist_matrix_flat)
    fit_order = np.argsort(fitness_array)[::-1]
    for idx_i, i in enumerate(fit_order):
        for idx_j, j in enumerate(fit_order[:idx_i]):
            r = dist_matrix[i, j]

            # if fitness_array[i] > fitness_array[j]:
            alpha = alpha_0 * delta**idx_j
            beta = beta_0 * np.exp(-gamma * r * r)
            population_matrix[i, :] = (
                population_matrix[i, :]
                + beta * (population_matrix[j, :] - population_matrix[i, :])
                + alpha * (random_state.random(n_components) - 0.5)
            )

    return population_matrix


def glowworm_operator(population, luciferin, fitness, random_state, rho, gamma, radial_range, step_size):
    raise NotImplementedError()

    random_state = check_random_state(random_state)

    if np.asarray(radial_range).ndim == 0:
        radial_range = np.full(population.shape[0], radial_range)

    # Update luciferin
    luciferin_next = (1 - rho) * luciferin + gamma * fitness

    # Update movement
    dist_matrix = sp_dist.pdist(population)

    for idx, ind in enumerate(population):
        # Select neighborhood
        neighbor_idx, *_ = np.where(dist_matrix[idx, :] < radial_range[idx])
        sum_dist = dist_matrix[idx, neighbor_idx].sum()
        for idx_n in neighbor_idx:
            p = (luciferin[idx_n] - luciferin[idx]) / sum_dist
            if random_state.random() < p:
                population[idx] += step_size * (population[idx_n] - population[idx]) / sp_dist.euclidean(population[idx_n], population[idx])

    return population, luciferin_next


def pso_operator_wrapper(population: Population, random_state=None, w=0.7, c1=1.5, c2=1.5):
    """ """

    population_encoding = population.encoding
    population_params = population_encoding.decode_params(population.genotype_matrix)
    historical_best_solution = population_encoding.extract_solution(population.historical_best)
    global_best_solution = population_encoding.extract_solution(population.best[None, :])[0]

    population_solutions, population_params["speed"] = pso_operator(
        population.genotype_matrix, population_params["speed"], historical_best_solution, global_best_solution, random_state, w, c1, c2
    )

    population_matrix = population.encoding.encode(population_solutions, population_params)
    return population.update_genotype_matrix(population_matrix)


def firefly_operator_wrapper(population: Population, random_state=None, alpha_0=0.2, beta_0=1.0, delta=1.0, gamma=0.97):
    """ """

    population_encoding = population.encoding
    population_params = population_encoding.decode_params(population.genotype_matrix)

    population_solutions, population_params["speed"] = firefly_operator(
        population.genotype_matrix, population.fitness, random_state, alpha_0, beta_0, delta, gamma
    )

    population_matrix = population.encoding.encode(population_solutions, population_params)
    return population.update_genotype_matrix(population_matrix)


def glowworm_operator_wrapper(population, fitness, luciferin, random_state, rho, gamma, radial_range, step_size):
    raise NotImplementedError()
