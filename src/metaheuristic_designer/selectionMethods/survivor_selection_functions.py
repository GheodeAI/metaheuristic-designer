from copy import copy
import random


def argsort(seq):
    """
    Implementation of argsort for python-style lists.
    Source: https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

    Parameters
    ----------
    seq: Iterable
        Iterable for which we want to obtain the order of.

    Returns
    -------
    order: List
        The positions of the original elements of the list in order.
    """

    return sorted(range(len(seq)), key=seq.__getitem__)


def one_to_one(popul, offspring):
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

    new_population = []
    for parent, child in zip(popul, offspring):
        if child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)

    if len(offspring) < len(popul):
        n_leftover = len(popul) - len(offspring)
        new_population += popul[n_leftover:]

    return new_population


def prob_one_to_one(popul, offspring, p):
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

    new_population = []
    for parent, child in zip(popul, offspring):
        if random.random() < p and child.fitness > parent.fitness:
            new_population.append(child)
        else:
            new_population.append(parent)

    if len(offspring) < len(popul):
        n_leftover = len(popul) - len(offspring)
        new_population += popul[n_leftover:]

    return new_population


def elitism(popul, offspring, amount):
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
    selected_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[
        : len(popul) - amount
    ]
    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]

    return best_parents + selected_offspring


def cond_elitism(popul, offspring, amount):
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

    best_parents = sorted(popul, reverse=True, key=lambda x: x.fitness)[:amount]
    new_offspring = sorted(offspring, reverse=True, key=lambda x: x.fitness)[
        : len(popul)
    ]
    best_offspring = new_offspring[:amount]

    for idx, val in enumerate(best_parents):
        if val.fitness > best_offspring[0].fitness:
            new_offspring.pop()
            new_offspring = [val] + new_offspring

    return new_offspring


def lamb_plus_mu(popul, offspring):
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

    population = popul + offspring
    return sorted(population, reverse=True, key=lambda x: x.fitness)[: len(popul)]


def lamb_comma_mu(popul, offspring):
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

    return sorted(offspring, reverse=True, key=lambda x: x.fitness)[: len(popul)]


def _cro_set_larvae(population, offspring, attempts, maxpopsize):
    """
    First step of the CRO selection function.

    Each individual in the offsring tries to settle down into the reef,
    if the spot they find is empty they are accepted, if there is already
    an individual in that spot, the one with the best fitness is kept.
    """

    new_population = copy(population)
    for larva in offspring:
        attempts_left = attempts
        setted = False

        while attempts_left > 0 and not setted:
            idx = random.randrange(0, maxpopsize)

            if setted := (idx >= len(new_population)):
                new_population.append(larva)
            elif setted := (larva.fitness > new_population[idx].fitness):
                new_population[idx] = larva

            attempts_left -= 1

    return new_population


def _cro_depredation(population, Fd, Pd):
    """
    Second step of the CRO selection function.

    A fraction Fd of the worse individuals in the population will be removed
    from the population with a probability of Pd.

    To ensure the integrity of the algorithm at least 2 individuals will always be
    kept.
    """

    amount = int(len(population) * Fd)

    fitness_values = [coral.fitness for coral in population]
    affected_corals = argsort(fitness_values)[:amount]

    alive_count = len(population)
    dead_list = [False] * len(population)

    for idx, val in enumerate(affected_corals):
        if alive_count <= 2:
            break

        dies = random.random() <= Pd
        dead_list[idx] = dies
        if dies:
            alive_count -= 1

    return [c for idx, c in enumerate(population) if not dead_list[idx]]


def cro_selection(popul, offspring, Fd, Pd, attempts, maxpopsize):
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

    setted_corals = _cro_set_larvae(popul, offspring, attempts, maxpopsize)
    reduced_population = _cro_depredation(setted_corals, Fd, Pd)
    return reduced_population
