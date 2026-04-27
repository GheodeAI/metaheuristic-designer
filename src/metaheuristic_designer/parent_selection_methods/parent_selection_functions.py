import warnings
import enum
from enum import Enum
import numpy as np
from ..utils import check_random_state


class SelectionDist(Enum):
    FIT_PROP = enum.auto()
    SIGMA_SCALE = enum.auto()
    LIN_RANK = enum.auto()
    EXP_RANK = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in select_dist_map:
            raise ValueError(f'Selection distribution "{str_input}" not defined')

        return select_dist_map[str_input]


select_dist_map = {
    "fitness_prop": SelectionDist.FIT_PROP,
    "sigma_scaling": SelectionDist.SIGMA_SCALE,
    "lin_rank": SelectionDist.LIN_RANK,
    "exp_rank": SelectionDist.EXP_RANK,
}


def selection_distribution(fitness, method, f=None):
    """
    Gives the weights that will be applied to each individual in
    the selection process.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    method: str, optional
        Indicates how the roulette will be generated.
    f: float, optional
        Parameter passed to some of the roulette generating methods.

    Returns
    -------
    weights: ndarray
        Weight assigned to each of the individuals
    """

    if f is None:
        f = 2

    match method:
        case SelectionDist.FIT_PROP:
            weights = fitness - fitness.min() + f
        case SelectionDist.SIGMA_SCALE:
            weights = np.maximum(fitness - (fitness.mean() - f * fitness.std()), 0)
        case SelectionDist.LIN_RANK:
            f = np.minimum(f, 2)
            fit_order = np.argsort(fitness)
            n_parents = fitness.shape[0]
            weights = (2 - f) + (2 * fit_order * (f - 1)) / (n_parents - 1)
        case SelectionDist.EXP_RANK:
            fit_order = np.argsort(fitness)
            weights = 1 - np.exp(-fit_order)
        case _:
            weights = np.ones_like(fitness)

    weight_norm = weights.sum()
    if weight_norm == 0:
        weights += 1
        weight_norm = weights.sum()

    return weights / weight_norm


def select_best(fitness, amount, _random_state=None):
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


def prob_tournament(fitness, amount, random_state=None, tournament_size=3, prob=1):
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

    random_state = check_random_state(random_state)

    n_individuals = fitness.shape[0]

    # Generate the participants of each tournament
    tournament_idx = random_state.integers(0, n_individuals, size=(amount, tournament_size))
    tournament_fit = fitness[tournament_idx]

    # Choose the best individual of each tournament
    best_idx = np.argmax(tournament_fit, axis=1)

    # Choose a random individual on each tournament
    random_idx = random_state.integers(0, tournament_size, size=amount)

    # Choose the final winner of the tournament
    chosen_idx = np.where(random_state.random(amount) < prob, best_idx, random_idx)
    selected_idx = tournament_idx[np.arange(amount), chosen_idx]

    return selected_idx


def uniform_selection(fitness, amount, random_state=None):
    """
    Chooses a number of individuals from the population at random.

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

    random_state = check_random_state(random_state)

    # Take a random sample of individuals
    return random_state.integers(0, fitness.shape[0], amount)


def roulette(fitness, amount, random_state=None, method=None, f=None):
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

    random_state = check_random_state(random_state)

    if method is None:
        method = SelectionDist.FIT_PROP

    weights = selection_distribution(fitness, method, f)

    if np.any(weights < 0):
        warnings.warn("Some values of fitness resulted in negative selection probabilities in the parent selection step.", stacklevel=2)

    return random_state.choice(np.arange(fitness.shape[0]), size=amount, p=weights, axis=0)


def sus(fitness, amount, random_state=None, method=None, f=None):
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

    random_state = check_random_state(random_state)

    if method is None:
        method = SelectionDist.FIT_PROP

    weights = selection_distribution(fitness, method, f)

    cum_weights = np.cumsum(weights)
    random_offsets = random_state.random(amount) / amount
    positions = random_offsets + (np.arange(amount) / amount)
    order = np.searchsorted(cum_weights, positions)

    return order
