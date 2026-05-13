"""
Mutation operator implementations based on probability distributions.
"""

import logging
import numpy as np
from .probability_distributions_factory import create_prob_distribution
from ...utils import check_random_state

logger = logging.getLogger(__name__)


def mutate_sample(population_matrix, fitness_array, distribution, N, random_state=None, **kwargs):
    random_state = check_random_state(random_state)
    population_size, n_components = population_matrix.shape

    distribution = create_prob_distribution(distribution, population_matrix, **kwargs)

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    rand_samples = distribution.sample(population_matrix.shape, random_state)

    population_matrix[mask_pos] = rand_samples[mask_pos]

    logger.debug("Resampled components of the vector %s, with mask %s", population_matrix[mask_pos], mask_pos.astype(int))

    return population_matrix


def mutate_noise(population_matrix, fitness_array, distribution, F, N, random_state=None, **kwargs):
    random_state = check_random_state(random_state)

    population_size, n_components = population_matrix.shape

    distribution = create_prob_distribution(distribution, population_matrix, **kwargs)

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    rand_samples = distribution.sample(population_matrix.shape, random_state)

    population_matrix[mask_pos] = population_matrix[mask_pos] + (F * rand_samples)[mask_pos]

    logger.debug(
        "Mutated components of the vector:\nvector = %s\nnoise_added = %s\nmask = %s",
        population_matrix[mask_pos],
        (F * rand_samples)[mask_pos],
        mask_pos.astype(int),
    )

    return population_matrix


def rand_sample(population_matrix, fitness_array, distribution, random_state=None, **kwargs):
    random_state = check_random_state(random_state)

    distribution = create_prob_distribution(distribution, population_matrix, **kwargs)

    rand_samples = distribution.sample(population_matrix.shape, random_state)

    logger.debug("Resampled vector %s", rand_samples)

    return rand_samples


def rand_noise(population_matrix, fitness_array, distribution, F, random_state=None, **kwargs):
    random_state = check_random_state(random_state)

    distribution = create_prob_distribution(distribution, population_matrix, **kwargs)

    rand_samples = distribution.sample(population_matrix.shape, random_state)
    result = population_matrix + F * rand_samples

    logger.debug("Added noise to vector %s", result)

    return result


def sample_1_sigma(population, _fitness, random_state=None, **kwargs):
    """
    Replaces 'n' components of the input vector with a value sampled from the mutate 1 sigma function.

    In future, it should be integrated in mutate_sample and sample_distribution functions, considering
    np.exp(tau * N(0,1)) as a distribution function with a minimum value of epsilon.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    sigma = kwargs["sigma"]
    tau = kwargs["tau"]
    n = kwargs["n"]

    mask_pos = np.tile(np.arange(population.shape[1]) < n, (population.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    sampled = np.maximum(epsilon, population * np.exp(tau * random_state.normal(0, 1, sigma.shape[0])))
    population[mask_pos] = sampled[mask_pos]
    return population


def mutate_1_sigma(population, _fitness, random_state=None, **kwargs):
    """
    Mutate a sigma value in base of tau param, where epsilon is de minimum value that a sigma can have.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]

    return np.maximum(epsilon, population * np.exp(tau * random_state.normal(0, 1, population.shape[0])[:, None]))


def mutate_n_sigmas(population, _fitness, random_state=None, **kwargs):
    """
    Mutate a list of sigmas values in base of tau and tau_multiple params, where epsilon is de minimum value that a sigma can have.
    """

    random_state = check_random_state(random_state)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]
    tau_multiple = kwargs["tau_multiple"]

    return np.maximum(
        epsilon,
        population
        * np.exp(
            tau * random_state.normal(0, 1, population.shape[0])[:, None] + tau_multiple * random_state.normal(0, 1, population.shape[0])[:, None]
        ),
    )


def xor_mask(population_matrix, fitness_array, N, mode="byte", random_state=None, **kwargs):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    random_state = check_random_state(random_state)
    population_size, n_components = population_matrix.shape

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)

    match mode:
        case "bin":
            mask = mask_pos
        case "byte":
            mask = random_state.integers(1, 0xFF, size=population_matrix.shape) * mask_pos
        case "int":
            mask = random_state.integers(1, 0xFFFF, size=population_matrix.shape) * mask_pos
        case _:
            mask = 0

    return population_matrix ^ mask
