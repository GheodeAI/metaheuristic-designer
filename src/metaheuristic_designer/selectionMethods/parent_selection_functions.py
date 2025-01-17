import warnings
import enum
from enum import Enum
import numpy as np
from ..utils import RAND_GEN


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
    "fitnessprop": SelectionDist.FIT_PROP,
    "sigmascaling": SelectionDist.SIGMA_SCALE,
    "linrank": SelectionDist.LIN_RANK,
    "exprank": SelectionDist.EXP_RANK,
}


def select_best(fitness, amount):
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
    order = np.argsort(fitness)[::-1][:amount]

    return order


def prob_tournament(fitness, tourn_size, prob):
    """
    Selects the parents for the next generation by tournament.

    Parameters
    ----------
    population: ndarray
        List of individuals from which the parents will be selected.
    tourn_size: int
        Amount of individuals that will be chosen for each tournament.
    prob: float
        Probability that a parent with low fitness will win the tournament.

    Returns
    -------
    parents: ndarray
        List of individuals chosen as parents.
    """

    # Generate the participants of each tournament
    tournament_idx = RAND_GEN.integers(0, fitness.shape[0], size=(fitness.shape[0], tourn_size))
    tournament_fit = fitness[tournament_idx]

    # Choose the best individual of each tournament
    best_idx = np.argmax(tournament_fit, axis=1)

    # Choose a random individual on each torunament
    random_idx = RAND_GEN.integers(0, tourn_size, size=fitness.shape[0])

    # Choose the final winner of the tournament
    chosen_idx = np.where(random_idx < RAND_GEN.random(fitness.shape[0]), best_idx, random_idx)
    selected_idx = tournament_idx[np.arange(fitness.shape[0]), chosen_idx]

    return selected_idx


def uniform_selection(fitness, amount):
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

    # Take a random sample of individuals
    return RAND_GEN.integers(0, fitness.shape[0], amount)


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
        Weight assinged to each of the individuals
    """

    if f is None:
        f = 2

    if method == SelectionDist.FIT_PROP:
        weights = fitness - fitness.min() + f
    elif method == SelectionDist.SIGMA_SCALE:
        weights = np.maximum(fitness - (fitness.mean() - f * fitness.std()), 0)
    elif method == SelectionDist.LIN_RANK:
        f = np.minimum(f, 2)
        fit_order = np.argsort(fitness)
        n_parents = fitness.shape[0]
        weights = (2 - f) + (2 * fit_order * (f - 1)) / (n_parents - 1)
    elif method == SelectionDist.EXP_RANK:
        fit_order = np.argsort(fitness)
        weights = 1 - np.exp(-fit_order)
    else:
        weights = np.ones_like(fitness)

    weight_norm = weights.sum()
    if weight_norm == 0:
        weights += 1
        weight_norm = weights.sum()

    return weights / weight_norm


def roulette(fitness, amount, method=None, f=None):
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

    if method is None:
        method = SelectionDist.FIT_PROP

    weights = selection_distribution(fitness, method, f)

    if np.any(weights < 0):
        warnings.warn(
            "Some values of fitness resulted in negative selection probabilities in the parent selection step.",
            stacklevel=2,
        )

    return RAND_GEN.choice(np.arange(fitness.shape[0]), size=amount, p=weights, axis=0)


def sus(fitness, amount, method=None, f=None):
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

    if method is None:
        method = SelectionDist.FIT_PROP

    weights = selection_distribution(fitness, method, f)

    cum_weights = np.cumsum(weights)
    random_offsets = RAND_GEN.random(amount) / amount
    positions = (np.arange(amount) / amount)[:, None] + random_offsets[:, None]
    order = np.searchsorted(cum_weights, positions.ravel())[:amount]

    return order
