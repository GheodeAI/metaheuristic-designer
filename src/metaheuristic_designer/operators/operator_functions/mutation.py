"""
Mutation operator implementations based on probability distributions.
"""

import logging
from typing import Optional
from ...utils import MatrixLike, RNGLike, VectorLike, ScalarLike, check_rng
import numpy as np
from .probability_distributions_factory import create_prob_distribution

logger = logging.getLogger(__name__)


def mutate_sample(
    population_matrix: MatrixLike, fitness_array: VectorLike, distribution: str, N: int, rng: Optional[RNGLike] = None, **kwargs
) -> MatrixLike:
    """
    Replace `N` components of each individual with random values.

    The new values are sampled from the probability distribution
    specified by `distribution`.  The remaining components are left
    unchanged.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population of shape ``(N_indiv, N_comp)``.
    fitness_array : VectorLike
        Fitness values (unused; kept for interface consistency).
    distribution : str
        Key of the distribution to use (see :func:`create_prob_distribution`).
    N : int
        Number of components to resample per individual.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :func:`create_prob_distribution` (e.g. ``loc``, ``scale``).

    Returns
    -------
    MatrixLike
        The modified population.
    """

    rng = check_rng(rng)
    population_size, n_components = population_matrix.shape

    distribution = create_prob_distribution(distribution, population_matrix, rng=rng, **kwargs)

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = rng.permuted(mask_pos, axis=1)

    rand_samples = distribution.sample(population_matrix.shape)

    population_matrix[mask_pos] = rand_samples[mask_pos]

    logger.debug("Resampled components of the vector %s, with mask %s", population_matrix[mask_pos], mask_pos.astype(int))

    return population_matrix


def mutate_noise(
    population_matrix: MatrixLike,
    fitness_array: VectorLike,
    distribution: str,
    F: ScalarLike | VectorLike,
    N: int,
    rng: Optional[RNGLike] = None,
    **kwargs,
) -> MatrixLike:
    """
    Add random noise to `N` components of each individual.

    The noise is drawn from `distribution`, multiplied by the
    strength factor `F`, and added to the selected components.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population of shape ``(N_indiv, N_comp)``.
    fitness_array : VectorLike
        Fitness values (unused).
    distribution : str
        Key of the distribution to use.
    F : ScalarLike | VectorLike
        Strength factor (scalar or per-individual array).
    N : int
        Number of components to mutate per individual.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :func:`create_prob_distribution`.

    Returns
    -------
    MatrixLike
        The mutated population.
    """

    rng = check_rng(rng)

    population_size, n_components = population_matrix.shape

    distribution = create_prob_distribution(distribution, population_matrix, rng=rng, **kwargs)

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = rng.permuted(mask_pos, axis=1)

    rand_samples = distribution.sample(population_matrix.shape)

    population_matrix[mask_pos] = population_matrix[mask_pos] + (F * rand_samples)[mask_pos]

    logger.debug(
        "Mutated components of the vector:\nvector = %s\nnoise_added = %s\nmask = %s",
        population_matrix[mask_pos],
        (F * rand_samples)[mask_pos],
        mask_pos.astype(int),
    )

    return population_matrix


def rand_sample(population_matrix: MatrixLike, fitness_array: VectorLike, distribution: str, rng: Optional[RNGLike] = None, **kwargs) -> MatrixLike:
    """
    Replace the entire population with new random values.

    Each element of the genotype matrix is independently resampled
    from `distribution`.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population of shape ``(N_indiv, N_comp)``. Only its shape
        is used.
    fitness_array : VectorLike
        Fitness values (unused).
    distribution : str
        Key of the distribution to use.
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :func:`create_prob_distribution`.

    Returns
    -------
    MatrixLike
        A new matrix of the same shape filled with random samples.
    """

    rng = check_rng(rng)

    distribution = create_prob_distribution(distribution, population_matrix, rng=rng, **kwargs)

    rand_samples = distribution.sample(population_matrix.shape)

    logger.debug("Resampled vector %s", rand_samples)

    return rand_samples


def rand_noise(
    population_matrix: MatrixLike, fitness_array: VectorLike, distribution: str, F: ScalarLike, rng: Optional[RNGLike] = None, **kwargs
) -> MatrixLike:
    """
    Add random noise to the entire population.

    The noise is drawn from `distribution`, scaled by `F`, and
    added to every element of the genotype matrix.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population of shape ``(N_indiv, N_comp)``.
    fitness_array : VectorLike
        Fitness values (unused).
    distribution : str
        Key of the distribution to use.
    F : ScalarLike
        Strength factor (scalar or per-individual array).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Forwarded to :func:`create_prob_distribution`.

    Returns
    -------
    MatrixLike
        Noisy population of the same shape.
    """

    rng = check_rng(rng)

    distribution = create_prob_distribution(distribution, population_matrix, rng=rng, **kwargs)

    rand_samples = distribution.sample(population_matrix.shape)
    result = population_matrix + F * rand_samples

    logger.debug("Added noise to vector %s", result)

    return result


def sample_1_sigma(population_matrix: MatrixLike, fitness_array: VectorLike, rng: Optional[RNGLike] = None, **kwargs) -> MatrixLike:
    """
    Replace `n` components using a log-normal perturbation with a
    stored sigma value.

    This is a self-adaptation helper for Evolution Strategies. The
    sigma values are expected to be passed in ``kwargs``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population.
    fitness_array : VectorLike
        Fitness values (unused).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Must contain ``epsilon``, ``sigma``, ``tau``, ``n``.

    Returns
    -------
    MatrixLike
        The mutated population.
    """

    rng = check_rng(rng)

    epsilon = kwargs["epsilon"]
    sigma = kwargs["sigma"]
    tau = kwargs["tau"]
    n = kwargs["n"]

    mask_pos = np.tile(np.arange(population_matrix.shape[1]) < n, (population_matrix.shape[0], 1))
    mask_pos = rng.permuted(mask_pos, axis=1)

    sampled = np.maximum(epsilon, population_matrix * np.exp(tau * rng.normal(0, 1, sigma.shape[0])))
    population_matrix[mask_pos] = sampled[mask_pos]
    return population_matrix


def mutate_1_sigma(population_matrix: MatrixLike, fitness_array: VectorLike, rng: Optional[RNGLike] = None, **kwargs) -> MatrixLike:
    """
    Mutate a single sigma value using a log-normal update.

    The new sigma is ``max(epsilon, old_sigma * exp(tau * N(0,1)))``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current sigma values (one per individual, or per dimension).
    fitness_array : VectorLike
        Fitness values (unused).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Must contain ``epsilon`` and ``tau``.

    Returns
    -------
    MatrixLike
        Updated sigma values.
    """

    rng = check_rng(rng)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]

    return np.maximum(epsilon, population_matrix * np.exp(tau * rng.normal(0, 1, population_matrix.shape[0])[:, None]))


def mutate_n_sigmas(population_matrix: MatrixLike, fitness_array: VectorLike, rng: Optional[RNGLike] = None, **kwargs) -> MatrixLike:
    """
    Mutate multiple sigma values with global and local learning rates.

    ``max(epsilon, old_sigma * exp(tau*N(0,1) + tau_multiple*N(0,1)))``.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current sigma values.
    fitness_array : VectorLike
        Fitness values (unused).
    rng : RNGLike, optional
        Random number generator.
    **kwargs
        Must contain ``epsilon``, ``tau``, ``tau_multiple``.

    Returns
    -------
    MatrixLike
        Updated sigma values.
    """

    rng = check_rng(rng)

    epsilon = kwargs["epsilon"]
    tau = kwargs["tau"]
    tau_multiple = kwargs["tau_multiple"]

    return np.maximum(
        epsilon,
        population_matrix
        * np.exp(tau * rng.normal(0, 1, population_matrix.shape[0])[:, None] + tau_multiple * rng.normal(0, 1, population_matrix.shape[0])[:, None]),
    )


def xor_mask(
    population_matrix: MatrixLike, fitness_array: VectorLike, N: int, mode: str = "byte", rng: Optional[RNGLike] = None, **kwargs
) -> MatrixLike:
    """
    Apply bitwise XOR with random masks to `N` components per individual.

    The mask is drawn as random bytes, integers, or single bits
    depending on `mode`.

    Parameters
    ----------
    population_matrix : MatrixLike
        Population.
    fitness_array : VectorLike
        Fitness values (unused).
    N : int
        Number of components to mask per individual.
    mode : str, optional
        Mask format: ``"bin"``, ``"byte"``, or ``"int"`` (default ``"byte"``).
    rng : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        The masked population.
    """

    rng = check_rng(rng)
    population_size, n_components = population_matrix.shape

    mask_pos = np.tile(np.arange(n_components) < N, (population_size, 1))
    mask_pos = rng.permuted(mask_pos, axis=1)

    match mode:
        case "bin":
            mask = mask_pos
        case "byte":
            mask = rng.integers(1, 0xFF, size=population_matrix.shape) * mask_pos
        case "int":
            mask = rng.integers(1, 0xFFFF, size=population_matrix.shape) * mask_pos
        case _:
            mask = 0

    return population_matrix ^ mask


def polynomial_mutation(
    population_matrix: MatrixLike,
    fitness_array: VectorLike,
    lower_bound: VectorLike | ScalarLike,
    upper_bound: VectorLike | ScalarLike,
    dist_index: float = 100,
    rng: Optional[RNGLike] = None,
    **kwargs,
):
    """
    Polynomial mutation for real-coded genetic algorithms.

    Performs per-component polynomial mutation on a population matrix.
    The perturbation magnitude depends on the distance from the parent
    to the respective bound, guaranteeing offspring stay within
    the feasible region without clamping.

    Parameters
    ----------
    population_matrix : MatrixLike
        Current population of real-valued solutions.
    fitness_array : VectorLike
        Fitness values (unused in this mutation, kept for compatibility).
    lower_bound : VectorLike | ScalarLike
        Lower bound(s) for each variable.
    upper_bound : VectorLike | ScalarLike
        Upper bound(s) for each variable.
    dist_index : float, optional
        Distribution index (η) controlling the mutation spread.
        Larger values (e.g., 100) give small perturbations (exploitation);
        smaller values (e.g., 20) allow larger jumps (exploration).
        Default is 100.
    rng : Optional[RNGLike], optional
        Random number generator. If None, uses `numpy.random.default_rng()`.
    **kwargs
        Additional arguments (unused, for compatibility).

    Returns
    -------
    population_matrix : ndarray, shape (n_individuals, n_vars)
        Mutated population.

    Notes
    -----
    The polynomial mutation operator was introduced by Deb & Agrawal (1995)
    and is described in detail in Deb & Deb (2012), KanGAL Report 2012016.
    For each variable independently, a random number u ∈ [0,1] is drawn:
        - If u ≤ 0.5:   child = parent + [(2u)^(1/(1+η)) - 1] * (parent - low)
        - If u > 0.5:    child = parent + [1 - (2(1-u))^(1/(1+η))] * (high - parent)
    This formulation ensures child ∈ [low, high] without post-clamping.
    """

    rng = check_rng(rng)

    u = rng.random(population_matrix.shape, dtype=float)

    delta = np.where(u <= 0.5, (2 * u) ** (1 / (1 + dist_index)) - 1, 1 - (2 * (1 - u)) ** (1 / (1 + dist_index)))

    population_matrix = np.where(
        u <= 0.5, population_matrix + delta * (population_matrix - lower_bound), population_matrix + delta * (upper_bound - population_matrix)
    )

    return population_matrix
