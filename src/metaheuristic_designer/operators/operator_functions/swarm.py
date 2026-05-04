""" """

from typing import Optional
import numpy as np
from ...population import Population
from ...initializer import Initializer
from ...encodings import ParameterExtendingEncoding
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


def pso_operator_wrapper(population: Population, _initializer: Initializer, random_state=None, w=0.7, c1=1.5, c2=1.5):
    """ """

    population_encoding = population.encoding
    if (not isinstance(population_encoding, ParameterExtendingEncoding)) or ("speed" not in population_encoding.extended_parameters):
        raise ValueError('Encoding of the population must be a ParameterExtendingEncoding with a "speed" parameter')

    population_genotype = population_encoding.extract_solution(population.genotype_matrix)
    population_params = population_encoding.decode_params(population.genotype_matrix)
    historical_best_solution = population_encoding.extract_solution(population.historical_best_matrix)
    global_best_solution = population_encoding.extract_solution(population.best[None, :])[0]

    population_solutions, population_params["speed"] = pso_operator(
        population_genotype, population_params["speed"], historical_best_solution, global_best_solution, random_state=random_state, w=w, c1=c1, c2=c2
    )

    population_matrix = population.encoding.encode(population_solutions, population_params)
    return population.update_genotype(population_matrix)
