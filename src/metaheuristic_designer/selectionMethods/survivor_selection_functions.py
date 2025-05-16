from copy import copy
import numpy as np
from ..utils import RAND_GEN


def one_to_one(population_fitness, offspring_fitness):
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


def prob_one_to_one(population_fitness, offspring_fitness, p):
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
    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    assert n_parents == n_offspring

    selection_mask = (population_fitness < offspring_fitness) | (RAND_GEN.random(n_parents) < p)
    full_idx = np.arange(n_parents)
    full_idx[selection_mask] += n_parents
    return full_idx


def many_to_one(population_fitness, offspring_fitness):
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


def prob_many_to_one(population_fitness, offspring_fitness, p):
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
    random_individual_idx = RAND_GEN.integers(0, n_repetitions + 1, n_parents)
    ignore_mask = RAND_GEN.random(n_parents) < p
    best_individual_idx[ignore_mask] = random_individual_idx[ignore_mask]

    # Get indices to use.
    full_idx = np.arange(n_parents)
    full_idx += best_individual_idx * n_parents

    return full_idx


def elitism(population_fitness, offspring_fitness, amount):
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


def cond_elitism(population_fitness, offspring_fitness, amount):
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


def lamb_plus_mu(population_fitness, offspring_fitness):
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


def lamb_comma_mu(population_fitness, offspring_fitness):
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


def _cro_set_larvae(population_fitness, offspring_fitness, attempts, maxpopsize):
    """
    First step of the CRO selection function.

    Each individual in the offsring tries to settle down into the reef,
    if the spot they find is empty they are accepted, if there is already
    an individual in that spot, the one with the best fitness is kept.
    """

    # new_population = copy(population)
    # for larva in offspring:
    #     attempts_left = attempts
    #     setted = False

    #     while attempts_left > 0 and not setted:
    #         idx = random.randrange(0, maxpopsize)

    #         if setted := (idx >= len(new_population)):
    #             new_population.append(larva)
    #         elif setted := (larva.fitness > new_population[idx].fitness):
    #             new_population[idx] = larva

    #         attempts_left -= 1

    # return new_population

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    idx_to_compare = RAND_GEN.integers(0, n_parents, size=(n_offspring, attempts))
    spot_fitness = population_fitness[idx_to_compare]
    new_fitness = np.tile(offspring_fitness, (attempts, 1)).T


def _cro_depredation(population, Fd, Pd):
    """
    Second step of the CRO selection function.

    A fraction Fd of the worse individuals in the population will be removed
    from the population with a probability of Pd.

    To ensure the integrity of the algorithm at least 2 individuals will always be
    kept.
    """

    # amount = int(len(population) * Fd)

    # fitness_values = [coral.fitness for coral in population]
    # affected_corals = np.argsort(fitness_values)[:amount]

    # alive_count = len(population)
    # dead_list = [False] * len(population)

    # for idx, val in enumerate(affected_corals):
    #     if alive_count <= 2:
    #         break

    #     dies = random.random() <= Pd
    #     dead_list[idx] = dies
    #     if dies:
    #         alive_count -= 1

    # return [c for idx, c in enumerate(population) if not dead_list[idx]]
    return []


def cro_selection(population_fitness, offspring_fitness, Fd, Pd, attempts, maxpopsize):
    """
    Selection method of the Coral Reef Optimization algorithm.
    The offspring first tries to be inserted into the population, then
    the whole resulting population will go through a depredation phase
    where the worse individuals will be removed.

    Parameters
    ----------
    popul: List[Individual]
        Original population of individuals before being operated on.
    offspring: List[Individual]
        Individuals resulting from an iteration of the algorithm.
    Fd: float
        Proportion of individuals with the worse fintess that will go through
        a depredation step.
    Pd: float
        Probability that an individual will be eliminated from the population
        in the depredation step.
    attempts: int
        Maximum number of times a solution can attempt to be inserted into a
        position with an individual with a better fitness value.
    maxpopsize: int
        Maximum size of the population.

    Returns
    -------
    survivors: List[Individual]
        The individuals selected for the next generation.
    """

    # setted_corals = _cro_set_larvae(popul, offspring, attempts, maxpopsize)
    # reduced_population = _cro_depredation(setted_corals, Fd, Pd)
    # return reduced_population

    # DELETEME (debug)
    return lamb_plus_mu(population_fitness, offspring_fitness)
