from copy import copy
import numpy as np
from ..utils import check_random_state


def generational(population_fitness, offspring_fitness, _random_state):
    return np.arange(population_fitness.shape[0], offspring_fitness.shape[0])


def one_to_one(population_fitness, offspring_fitness, _random_state):
    """
    Compares each new individual with its parent and it replaces it if
    it has a better fitness.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    assert n_parents == n_offspring

    selection_mask = population_fitness <= offspring_fitness
    full_idx = np.arange(n_parents)
    full_idx[selection_mask] += n_parents
    return full_idx


def prob_one_to_one(population_fitness, offspring_fitness, random_state, p):
    """
    Compares each new individual with its parent and it replaces it with a
    probability of p or if it has a better fitness.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.
    p: float
        Probability that an individual will be replaced by its child even if it has a worse fitness.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    random_state = check_random_state(random_state)

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    assert n_parents == n_offspring

    selection_mask = (population_fitness < offspring_fitness) | (random_state.random(n_parents) < p)
    full_idx = np.arange(n_parents)
    full_idx[selection_mask] += n_parents
    return full_idx


def many_to_one(population_fitness, offspring_fitness, _random_state):
    """
    Compares each new individual with its parent and it replaces it if
    it has a better fitness.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]
    n_repetitions = n_offspring // n_parents

    assert (n_offspring % n_parents) == 0

    # Reorder fitness, compare each individual with it's offspring
    reshaped_offspring_fitness = offspring_fitness.reshape((n_repetitions, n_parents))
    fitness_matrix = np.concatenate([population_fitness[None, :], reshaped_offspring_fitness], axis=0)

    # Get the best child or the parent for each individual
    best_individual_idx = np.argmax(fitness_matrix, axis=0)

    # Get indices to use.
    full_idx = np.arange(n_parents)
    full_idx += best_individual_idx * n_parents

    return full_idx


def prob_many_to_one(population_fitness, offspring_fitness, random_state, p):
    """
    Compares each new individual with its parent and it replaces it with a
    probability of p or if it has a better fitness.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.
    p: float
        Probability that an individual will be replaced by its child even if it has a worse fitness.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    random_state = check_random_state(random_state)

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]
    n_repetitions = n_offspring // n_parents

    assert (n_offspring % n_parents) == 0

    # Reorder fitness, compare each individual with it's offspring.
    reshaped_offspring_fitness = offspring_fitness.reshape((n_repetitions, n_parents))
    fitness_matrix = np.concatenate([population_fitness[None, :], reshaped_offspring_fitness], axis=0)

    # Get the best child or the parent for each individual
    best_individual_idx = np.argmax(fitness_matrix, axis=0)

    # Use random individual with probability 'p'.
    random_individual_idx = random_state.integers(0, n_repetitions + 1, n_parents)
    ignore_mask = random_state.random(n_parents) < p
    best_individual_idx[ignore_mask] = random_individual_idx[ignore_mask]

    # Get indices to use.
    full_idx = np.arange(n_parents)
    full_idx += best_individual_idx * n_parents

    return full_idx


def elitism(population_fitness, offspring_fitness, _random_state, amount):
    """
    The offspring is passed to the next generation and a number of the
    parents replace the worst individuals.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.
    amount: int
        Amount of parents from the original population that will be kept.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    n_parents = population_fitness.shape[0]
    amount = min(n_parents, amount)

    parent_order = np.argsort(population_fitness)[::-1][:amount]
    offspring_order = np.argsort(offspring_fitness)[::-1][: n_parents - amount]
    return np.concatenate((parent_order, offspring_order + n_parents))


def cond_elitism(population_fitness, offspring_fitness, _random_state, amount):
    """
    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.
    amount: int
        Amount of parents from the original population that will be kept.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """
    # best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]
    # new_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[: len(popul)]
    # best_offspring = new_offspring[:amount]

    # for idx, val in enumerate(best_parents):
    #     if val.fitness > best_offspring[0].fitness:
    #         new_offspring.pop()
    #         new_offspring = [val] + new_offspring

    # return new_offspring

    n_parents = population_fitness.shape[0]

    parent_order = np.argsort(population_fitness)[::-1][:amount]
    offspring_order = np.argsort(offspring_fitness)[::-1][: n_parents - amount]
    return np.concatenate((parent_order, offspring_order + n_parents))


def lamb_plus_mu(population_fitness, offspring_fitness, _random_state):
    """
    Both the parents and the offspring are considered and the best
    of them will pass to the next generation.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    n_parents = population_fitness.shape[0]

    full_fitness = np.concatenate((population_fitness, offspring_fitness))
    fitness_order = np.argsort(full_fitness)[::-1][:n_parents]

    return fitness_order


def lamb_comma_mu(population_fitness, offspring_fitness, _random_state):
    """
    Only the best individuals in the offsping are selected.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    n_parents = population_fitness.shape[0]

    fitness_order = np.argsort(offspring_fitness)[::-1][:n_parents] + n_parents
    return fitness_order
