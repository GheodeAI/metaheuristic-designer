from copy import copy
import numpy as np
from ..utils import check_random_state, VectorLike, MatrixLike, RNGLike


def generational(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike) -> VectorLike:
    """
    Full generational replacement: the entire next generation is formed
    exclusively by the offspring. No parent survives.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population. Only its size is used.
    offspring_fitness : VectorLike
        Fitness values of the offspring population.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals. Offspring indices are offset by
        `len(population_fitness)` so that the caller can distinguish them.
    """

    return np.arange(offspring_fitness.shape[0]) + population_fitness.shape[0]


def one_to_one(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike) -> VectorLike:
    """
    One-to-one competition: each offspring replaces its parent if it has a
    better (higher) fitness. Parent and offspring populations must have the
    same size.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness values of the offspring, one per parent.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals. Indices < n_parents point to
        parents; indices >= n_parents point to offspring.
    """

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    assert n_parents == n_offspring

    selection_mask = population_fitness <= offspring_fitness
    full_idx = np.arange(n_parents)
    full_idx[selection_mask] += n_parents
    return full_idx


def prob_one_to_one(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike, p: float) -> VectorLike:
    """
    Probabilistic one-to-one competition. An offspring replaces its parent
    if it has a better fitness, OR with probability `p` regardless of fitness.
    Populations must be the same size.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness values of the offspring, one per parent.
    random_state : RNGLike
        Seeded random state for the stochastic replacement decision.
    p : float
        Probability of replacing a parent even if the offspring is worse.

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals (parent indices offset when replaced).
    """

    random_state = check_random_state(random_state)

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    assert n_parents == n_offspring

    selection_mask = (population_fitness < offspring_fitness) | (random_state.random(n_parents) < p)
    full_idx = np.arange(n_parents)
    full_idx[selection_mask] += n_parents
    return full_idx


def many_to_one(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike) -> VectorLike:
    """
    Many-to-one competition. Each parent competes against its own block of
    `n_repetitions` offspring (offspring size must be a multiple of parent size).
    The best individual among {parent, offspring_1, …, offspring_k} survives.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness of all offspring, grouped in contiguous blocks of equal size
        (one block per parent).
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals, with offspring indices shifted by
        n_parents for each repetition appropriately.
    """

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]
    n_repetitions = n_offspring // n_parents

    assert (n_offspring % n_parents) == 0

    # Reorder fitness, compare each individual with its offspring
    reshaped_offspring_fitness = offspring_fitness.reshape((n_repetitions, n_parents))
    fitness_matrix = np.concatenate([population_fitness[None, :], reshaped_offspring_fitness], axis=0)

    # Get the best child or the parent for each individual
    best_individual_idx = np.argmax(fitness_matrix, axis=0)

    # Get indices to use.
    full_idx = np.arange(n_parents)
    full_idx += best_individual_idx * n_parents

    return full_idx


def prob_many_to_one(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike, p: float) -> VectorLike:
    """
    Probabilistic many-to-one competition. Like `many_to_one`, but with
    probability `p` the winner is replaced by a uniformly random competitor
    from the pool (parent + its offspring).

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness of all offspring, grouped in contiguous blocks per parent.
    random_state : RNGLike
        Seeded random state.
    p : float
        Probability of ignoring the fitness-based winner and picking a random
        individual from the block.

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals.
    """

    random_state = check_random_state(random_state)

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]
    n_repetitions = n_offspring // n_parents

    assert (n_offspring % n_parents) == 0

    # Reorder fitness, compare each individual with its offspring.
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


def elitism(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike, amount: int) -> VectorLike:
    """
    Standard elitism. The top `amount` parents (highest fitness) survive;
    the remaining slots are filled by the best offspring.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness values of the offspring population.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).
    amount : int
        How many of the best parents are unconditionally preserved.

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals. Parent indices appear as-is;
        offspring indices are shifted by the number of parents.
    """

    n_parents = population_fitness.shape[0]
    amount = min(n_parents, amount)

    parents_selected = np.argsort(population_fitness)[::-1][:amount]
    offspring_selected = np.argsort(offspring_fitness)[::-1][: n_parents - amount]
    return np.concatenate((parents_selected, offspring_selected + n_parents))


def cond_elitism(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike, amount: int) -> VectorLike:
    """
    Conditional (fitness-based) elitism. A parent among the top `amount`
    is kept **only** if its fitness is strictly higher than the best offspring.
    Otherwise the elite slot is given to an offspring.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness of the previous population.
    offspring_fitness : VectorLike
        Fitness of the new offspring.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).
    amount : int
        Maximum number of elite candidates considered.

    Returns
    -------
    survivors : VectorLike
        Indices of the selected individuals (parent indices not shifted,
        offspring indices shifted by n_parents).
    """

    n_parents = population_fitness.shape[0]

    parent_order = np.argsort(population_fitness)[::-1]
    offspring_order = np.argsort(offspring_fitness)[::-1]
    elite_candidates = parent_order[:amount]

    max_offspring_fitness = np.max(offspring_fitness)

    keep_mask = population_fitness[elite_candidates] > max_offspring_fitness
    elites = elite_candidates[keep_mask]
    n_elites = elites.shape[0]

    offspring_selected = offspring_order[: n_parents - n_elites]

    return np.concatenate((elites, offspring_selected + n_parents))


def keep_best(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike) -> VectorLike:
    """
    Combined selection: the best `n_parents` individuals from the union of
    parents and offspring survive. Indices are absolute positions in the
    concatenated array [parents, offspring].

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population.
    offspring_fitness : VectorLike
        Fitness values of the offspring population.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices into the concatenated fitness array (0..n_parents-1 for parents,
        n_parents.. for offspring).
    """

    n_parents = population_fitness.shape[0]

    full_fitness = np.concatenate((population_fitness, offspring_fitness))
    fitness_order = np.argsort(full_fitness)[::-1][:n_parents]

    return fitness_order


def keep_best_offspring(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike) -> VectorLike:
    """
    Offspring-only selection: the best `n_parents` offspring survive.
    Parents are completely discarded.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population (only its length is used).
    offspring_fitness : VectorLike
        Fitness values of the offspring population.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices of the selected offspring, shifted by n_parents so that
        they are distinguishable from parent indices.
    """

    n_parents = population_fitness.shape[0]

    fitness_order = np.argsort(offspring_fitness)[::-1][:n_parents] + n_parents
    return fitness_order


def random_replacement(population_fitness: VectorLike, offspring_fitness: VectorLike, random_state: RNGLike, p: float = 0.5) -> VectorLike:
    """
    Randomly replaces the parents with some of the individuals.

    Parameters
    ----------
    population_fitness : VectorLike
        Fitness values of the parent population (only its length is used).
    offspring_fitness : VectorLike
        Fitness values of the offspring population.
    random_state : RNGLike
        Random state (unused; kept for interface consistency).

    Returns
    -------
    survivors : VectorLike
        Indices of the selected offspring, shifted by n_parents so that
        they are distinguishable from parent indices.
    """

    random_state = check_random_state(random_state)

    n_parents = population_fitness.shape[0]
    n_offspring = offspring_fitness.shape[0]

    replacement_idx = random_state.random(n_parents) > p
    n_chosen = np.count_nonzero(replacement_idx)
    parent_idx = np.arange(n_parents)
    parent_idx[replacement_idx] = random_state.permutation(n_offspring)[:n_chosen]
    return parent_idx
