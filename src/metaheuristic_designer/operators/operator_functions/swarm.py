""" """

from typing import Optional
import numpy as np
import scipy.spatial.distance as sp_dist
from ...population import Population
from ...initializer import Initializer
from ...utils import check_random_state, RNGLike


def pso_operator(
    population_matrix: np.array,
    population_speed: np.array,
    historical_best: np.array,
    global_best: np.array,
    random_state: Optional[RNGLike] = None,
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


def pso_operator_wrapper(population: Population, _initializer: Initializer, random_state=None, w=0.7, c1=1.5, c2=1.5):
    """ """

    from ...encodings import ParameterExtendingEncoding

    population_encoding: ParameterExtendingEncoding = population.encoding
    population_genotype = population_encoding.decode(population.genotype_matrix)
    population_params = population_encoding.decode_params(population.genotype_matrix)
    historical_best_solution = population_encoding.extract_solution(population.historical_best_matrix)
    global_best_solution = population_encoding.extract_solution(population.best[None, :])[0]

    population_solutions, population_params["speed"] = pso_operator(
        population_genotype, population_params["speed"], historical_best_solution, global_best_solution, random_state=random_state, w=w, c1=c1, c2=c2
    )

    population_matrix = population.encoding.encode(population_solutions, population_params)
    return population.update_genotype(population_matrix)
