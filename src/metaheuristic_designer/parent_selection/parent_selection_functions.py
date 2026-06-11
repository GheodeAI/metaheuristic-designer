"""
Core parent selection functions (tournament, roulette, SUS, best, …) and
fitness scaling helpers.
"""

from typing import Callable, Optional
import warnings
import numpy as np
from ..utils import MaskLike, RNGLike, ScalarLike, VectorLike, check_rng


# ---------------------------------------------
# Population ranking factory logic
# ---------------------------------------------
def fitness_proportional(fitness: VectorLike, scaling_factor: ScalarLike) -> VectorLike:
    """Fitness proportional scaling.

    Shift fitness to be non-negative and add a constant offset.

    Parameters
    ----------
    fitness : VectorLike
        Raw fitness values of the population.
    scaling_factor : ScalarLike
        Offset added after shifting to ensure all weights are positive.

    Returns
    -------
    VectorLike
        unnormalized selection weights.
    """

    return fitness - fitness.min() + scaling_factor


def sigma_scaling(fitness: VectorLike, scaling_factor: ScalarLike) -> VectorLike:
    """Sigma scaling: weight based on standard deviations above the mean.

    Values below ``mean - scaling_factor * std`` are clamped to zero.

    Parameters
    ----------
    fitness : VectorLike
        Raw fitness values.
    scaling_factor : ScalarLike
        Number of standard deviations below the mean to clamp.

    Returns
    -------
    VectorLike
        unnormalized selection weights.
    """

    return np.maximum(fitness - (fitness.mean() - scaling_factor * fitness.std()), 0)


def linear_ranking(fitness: VectorLike, scaling_factor: ScalarLike) -> VectorLike:
    """Linear ranking: weight proportional to rank.

    Rank 0 (worst) receives the smallest weight; rank N-1 (best) the largest.
    The scaling factor is clamped to at most 2.

    Parameters
    ----------
    fitness : VectorLike
        Raw fitness values.
    scaling_factor : ScalarLike
        Selection pressure (clamped to ≤2). Lower values give more
        extreme emphasis on high ranks.

    Returns
    -------
    VectorLike
        unnormalized selection weights.
    """

    scaling_factor = np.minimum(scaling_factor, 2)
    fit_order = np.argsort(np.argsort(fitness))  # Using the double-argsort trick
    n_parents = fitness.shape[0]
    return (2 - scaling_factor) + (2 * fit_order * (scaling_factor - 1)) / (n_parents - 1)


def exponential_ranking(fitness: VectorLike, scaling_factor: ScalarLike) -> VectorLike:
    """Exponential ranking: weight decays exponentially with rank.

    Parameters
    ----------
    fitness : VectorLike
        Raw fitness values.
    scaling_factor : ScalarLike
        Not used directly; included for interface consistency.

    Returns
    -------
    VectorLike
        unnormalized selection weights.
    """

    fit_order = np.argsort(np.argsort(fitness))  # Using the double-argsort trick
    return 1 - np.exp(-fit_order)


def flat_ranking(fitness: VectorLike, scaling_factor: ScalarLike) -> VectorLike:
    """Flat ranking: every individual receives equal weight.

    Parameters
    ----------
    fitness : VectorLike
        Raw fitness values.
    scaling_factor : ScalarLike
        Not used; included for interface consistency.

    Returns
    -------
    VectorLike
        unnormalized weights (all ones).
    """

    return np.ones_like(fitness)


# fmt: off
scaling_map = {
    "fitness_proportional": fitness_proportional,
    "fitness_prop": fitness_proportional,

    "sigma_scaling": sigma_scaling,

    "linear_scaling": linear_ranking,
    "linear_ranking": linear_ranking,
    "linear_rank": linear_ranking,

    "exponential_scaling": exponential_ranking,
    "exponential_ranking": exponential_ranking,
    "exponential_rank": exponential_ranking,

    "flat_scaling": flat_ranking
}
# fmt: on


def create_scaling_fn(method: str, scaling_factor: ScalarLike = 2) -> Callable:
    """Create a callable that computes normalized selection weights.

    Parameters
    ----------
    method : str
        Key into :data:`scaling_map` (e.g., ``"fitness_proportional"``).
    scaling_factor : float, optional
        Factor forwarded to the underlying scaling function.

    Returns
    -------
    callable
        A function ``(fitness) -> weights`` that returns a normalized
        probability vector.
    """

    chosen_fn = scaling_map[method.lower()]

    def wrapper(fitness: VectorLike):
        weights = chosen_fn(fitness, scaling_factor=scaling_factor)

        weight_norm = weights.sum()
        if weight_norm == 0:
            weights += 1
            weight_norm = weights.sum()

        return weights / weight_norm

    return wrapper


# ---------------------------------------------
# Population selection methods
# ---------------------------------------------
def repeating_selection(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None) -> MaskLike:
    """
    Chooses the entire population repeated in order duplicated enough times to reach the specified amount.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be replicated.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    n_individuals = fitness.shape[0]
    repetitions = amount // n_individuals
    selected_idx = np.tile(np.arange(n_individuals), repetitions)

    return selected_idx[:amount]


def select_best(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None) -> MaskLike:
    """
    Selects the best parent of the population as parents.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    # Select the best indices of the best 'n' individuals
    return np.argsort(fitness)[::-1][:amount]


def prob_tournament(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None, tournament_size: int = 3, prob: float = 1) -> MaskLike:
    """
    Selects the parents for the next generation by tournament.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    tournament_size: int
        Amount of individuals that will be chosen for each tournament.
    prob: float
        Probability that a parent with low fitness will win the tournament.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    rng = check_rng(rng)

    n_individuals = fitness.shape[0]

    # Generate the participants of each tournament
    tournament_idx = rng.integers(0, n_individuals, size=(amount, tournament_size))
    tournament_fit = fitness[tournament_idx]

    # Choose the best individual of each tournament
    best_idx = np.argmax(tournament_fit, axis=1)

    # Choose a random individual on each tournament
    random_idx = rng.integers(0, tournament_size, size=amount)

    # Choose the final winner of the tournament
    chosen_idx = np.where(rng.random(amount) < prob, best_idx, random_idx)
    selected_idx = tournament_idx[np.arange(amount), chosen_idx]

    return selected_idx


def uniform_selection(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None) -> MaskLike:
    """
    Chooses a number of individuals from the population at random with replacement.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    rng = check_rng(rng)

    # Take a random sample of individuals
    return rng.integers(0, fitness.shape[0], amount)


def shuffle_population(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None) -> MaskLike:
    """
    Chooses a number of individuals from the population at random without replacement if amount < population_size.
    If we cannot pick without replacement, we at least make sure we pick every individual at least
    :math:`\\left\\lceil \\frac{\\text{amount}}{\\text{population\\_size}} \\right\\rceil` times

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """
    rng = check_rng(rng)

    population_size = fitness.shape[0]

    if amount <= population_size:
        picked_idx = rng.permuted(np.arange(population_size))[:amount]
    else:
        repetitions = np.ceil(amount / population_size).astype(int)
        idx_choice = np.tile(np.arange(population_size), repetitions)
        picked_idx = rng.permuted(idx_choice)[:amount]

    return picked_idx


def roulette(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None, method: str = "flat_scaling", scaling_factor: float = None) -> MaskLike:
    """
    Fitness proportionate parent selection.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.
    method: str, optional
        Indicates how the roulette will be generated.
    f: float, optional
        Parameter passed to some of the roulette generating methods.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    rng = check_rng(rng)

    scaling_fn = create_scaling_fn(method, scaling_factor)
    weights = scaling_fn(fitness)

    if np.any(weights < 0):
        warnings.warn("Some values of fitness resulted in negative selection probabilities in the parent selection step.", stacklevel=2)

    return rng.choice(np.arange(fitness.shape[0]), size=amount, p=weights, axis=0)


def sus(fitness: VectorLike, amount: int, rng: Optional[RNGLike] = None, method: str = "flat_scaling", scaling_factor: float = None) -> MaskLike:
    """
    Stochastic universal sampling parent selection method.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.
    method: str, optional
        Indicates how the roulette will be generated.
    f: float, optional
        Parameter passed to some of the roulette generating methods.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    rng = check_rng(rng)

    scaling_fn = create_scaling_fn(method, scaling_factor)
    weights = scaling_fn(fitness)

    cum_weights = np.cumsum(weights)
    random_offsets = rng.random(amount) / amount
    positions = random_offsets + (np.arange(amount) / amount)
    order = np.searchsorted(cum_weights, positions, side="right") % len(cum_weights)

    return order
