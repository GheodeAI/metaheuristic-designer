import numpy as np
import random
import warnings
import enum
from enum import Enum


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


def select_best(population, amount):
    """
    Selects the best parent of the population as parents.

    Parameters
    ----------
    population: List[Individual]
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: List[Individual]
        List of individuals chosen as parents.
    """

    # Get the fitness of all the individuals
    fitness_list = np.fromiter(map(lambda x: x.fitness, population), dtype=float)

    # Get the index of the individuals sorted by fitness
    order = np.argsort(fitness_list)[::-1][:amount]

    # Select the 'amount' best individuals
    parents = [population[i] for i in order]

    # return parents, order
    return parents


def prob_tournament(population, tourn_size, prob):
    """
    Selects the parents for the next generation by tournament.

    Parameters
    ----------
    population: List[Individual]
        List of individuals from which the parents will be selected.
    tourn_size: int
        Amount of individuals that will be chosen for each tournament.
    prob: float
        Probability that a parent with low fitness will win the tournament.

    Returns
    -------
    parents: List[Individual]
        List of individuals chosen as parents.
    """

    parent_pool = []
    order = []

    for _ in population:
        # Choose 'tourn_size' individuals for the torunament
        parent_idxs = random.sample(range(len(population)), tourn_size)
        parents = [population[i] for i in parent_idxs]
        fits = [i.fitness for i in parents]

        # Choose one of the individuals
        if random.random() < prob:
            idx = random.randint(0, tourn_size - 1)
        else:
            idx = fits.index(max(fits))

        # Add the individuals to the list
        order.append(parent_idxs[idx])
        parent = parents[idx]
        parent_pool.append(parent)

    # return parent_pool, order
    return parent_pool


def uniform_selection(population, amount):
    """
    Chooses a number of individuals from the population at random.

    Parameters
    ----------
    population: List[Individual]
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.

    Returns
    -------
    parents: List[Individual]
        List of individuals chosen as parents.
    """

    order = random.choices(range(len(population)), k=amount)
    parents = [population[i] for i in order]

    # return parents, order
    return parents


def selection_distribution(population, method, f=2):
    """
    Gives the weights that will be applied to each individual in
    the selection process.

    Parameters
    ----------
    population: List[Individual]
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
    fit_list = np.fromiter((i.fitness for i in population), float)

    if method == SelectionDist.FIT_PROP:
        weights = fit_list
    elif method == SelectionDist.SIGMA_SCALE:
        weights = np.maximum(fit_list - (fit_list.mean() - f * fit_list.std()), 0)
    elif method == SelectionDist.LIN_RANK:
        fit_order = np.argsort(fit_list)
        n_parents = len(population)
        weights = (2 - f) + (2 * fit_order * (f - 1)) / (n_parents - 1)
    elif method == SelectionDist.EXP_RANK:
        fit_order = np.argsort(fit_list)
        weights = 1 - np.exp(-fit_order)

    weight_norm = weights.sum()
    if weight_norm == 0:
        weights += 1
        weight_norm = weights.sum()

    return weights / weight_norm


def roulette(population, amount, method=None, f=None):
    """
    Fitness proportionate parent selection.

    Parameters
    ----------
    population: List[Individual]
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.
    method: str, optional
        Indicates how the roulette will be generated.
    f: float, optional
        Parameter passed to some of the roulette generating methods.

    Returns
    -------
    parents: List[Individual]
        List of individuals chosen as parents.
    """

    if method is None:
        method = "basic"

    if f is None:
        f = 2

    weights = selection_distribution(population, method, f)

    if np.any(weights < 0):
        warnings.warn(
            "Some values of fitness resulted in negative selection probabilities in the parent selection step.",
            stacklevel=2,
        )

    order = random.choices(range(len(population)), k=amount, weights=weights)
    parents = [population[i] for i in order]

    # return parents, order
    return parents


def sus(population, amount, method=None, f=None):
    """
    Stochastic universal sampling parent selection method.

    Parameters
    ----------
    population: List[Individual]
        List of individuals from which the parents will be selected.
    amount: int
        Amount of individuals to be chosen as parents.
    method: str, optional
        Indicates how the roulette will be generated.
    f: float, optional
        Parameter passed to some of the roulette generating methods.

    Returns
    -------
    parents: List[Individual]
        List of individuals chosen as parents.
    """

    if method is None:
        method = "basic"

    if f is None:
        f = 2

    weights = selection_distribution(population, method, f)

    cum_weights = np.cumsum(weights)

    order = []
    current_member = i = 1
    while current_member < amount:
        r = random.random() / amount
        while r <= cum_weights[i] and current_member < amount:
            order.append(i)
            r = r + 1 / amount
            current_member += 1
        i += 1

    parents = [population[idx] for idx in order]

    # return parents, order
    return parents
