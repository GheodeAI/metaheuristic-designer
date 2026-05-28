"""Permutation-specific genetic operators (mutations and crossover)."""

from typing import Optional
import numpy as np
from ...utils import MatrixLike, RNGLike, VectorLike, check_random_state
from .crossover import create_pairing_fn


def permute_mutation(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    random_state: Optional[RNGLike] = None,
    N: Optional[int] = None,
) -> MatrixLike:
    """
    Randomly permute ``N`` components of each individual.

    When ``N`` is not given, all components are shuffled (a full permutation).
    The same subset of positions is used for every row, but a different
    random permutation is applied to each individual.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape ``(pop_size, num_components)``.
    fitness_array : VectorLike
        Fitness values (unused; kept for interface consistency).
    random_state : RNGLike, optional
        Random number generator.
    N : int, optional
        Number of components to permute. Clipped between 2 and the number
        of components. Defaults to the population size when ``None``.

    Returns
    -------
    MatrixLike
        The mutated population with permuted components.
    """
    random_state = check_random_state(random_state)

    if N is None:
        N = population_array.shape[0]

    N = np.clip(N, 2, population_array.shape[0])

    mask_pos = np.tile(np.arange(population_array.shape[1]), (population_array.shape[0], 1))
    mask_pos = random_state.permuted(mask_pos, axis=1)[:, :N]

    if N == 2:
        population_array[np.arange(population_array.shape[0])[:, None], mask_pos] = population_array[
            np.arange(population_array.shape[0])[:, None], mask_pos
        ][:, ::-1]
    else:
        population_array[np.arange(population_array.shape[0])[:, None], mask_pos] = random_state.permuted(
            population_array[np.arange(population_array.shape[0])[:, None], mask_pos], axis=1
        )

    return population_array


def roll_mutation(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    random_state: Optional[RNGLike] = None,
    N: int = 1,
) -> MatrixLike:
    """
    Cyclically shift (roll) a random segment of each individual.

    For each solution, a contiguous interval ``[start, end)`` is chosen
    uniformly.  That segment is then rolled by ``N`` positions.
    ``N`` defaults to 1, which effectively moves the first element of the
    segment to the end.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape ``(pop_size, num_components)``.
    fitness_array : VectorLike
        Fitness values (unused).
    random_state : RNGLike, optional
        Random number generator.
    N : int, optional
        Number of positions to roll inside the segment. Default is 1.

    Returns
    -------
    MatrixLike
        The mutated population.
    """
    random_state = check_random_state(random_state)

    roll_start = random_state.integers(0, population_array.shape[1] - 2, population_array.shape[0])
    roll_end = random_state.integers(roll_start + 2, population_array.shape[1] + 1, (population_array.shape[0]))

    def roll_individual(indiv, start, end, n):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = np.roll(indiv[start:end], n)
        return indiv_copy

    roll_vec = np.vectorize(roll_individual, signature="(m),(),(),()->(m)")
    population_array = roll_vec(population_array, roll_start, roll_end, N)
    return population_array


def invert_mutation(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Reverse the order of a random contiguous segment in each individual.

    A segment ``[start, end)`` is selected uniformly for every row,
    and its elements are reversed in place.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape ``(pop_size, num_components)``.
    fitness_array : VectorLike
        Fitness values (unused).
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        The mutated population.
    """
    random_state = check_random_state(random_state)

    invert_start = random_state.integers(0, population_array.shape[1] - 2, population_array.shape[0])
    invert_end = random_state.integers(invert_start + 2, population_array.shape[1] + 1, population_array.shape[0])

    def invert_individual(indiv, start, end):
        indiv_copy = indiv.copy()
        indiv_copy[start:end] = indiv[start:end][::-1]
        return indiv_copy

    invert_vec = np.vectorize(invert_individual, signature="(m),(),()->(m)")
    population_array = invert_vec(population_array, invert_start, invert_end)
    return population_array


def pmx(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Partially Mapped Crossover (PMX) for permutation chromosomes.

    Parents are paired using the given *pairing_method*.  For each pair,
    two children are created by the standard PMX procedure, which
    preserves a randomly chosen segment from one parent and maps the
    remaining positions from the other parent.  With probability
    *crossover_prob* the children are replaced by exact copies of the
    parents.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape ``(N, M)``, where each row is a permutation
        of integers ``0 … M-1``.
    fitness_array : VectorLike
        Fitness values (unused).
    pairing_method : str, optional
        Pairing strategy (``"random"`` or ``"stable"``).
    crossover_prob : float, optional
        Probability of applying crossover to a pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape ``(N, M)``.
    """
    random_state = check_random_state(random_state)

    population_size, n_components = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

    crossed = np.empty((n_parents * 2, n_components))
    for i in range(n_parents):
        if random_state.random() < crossover_prob:
            crossed[i] = pmx_single(parents1[i], parents2[i], random_state=random_state)
            crossed[i + n_parents] = pmx_single(parents2[i], parents1[i], random_state=random_state)
        else:
            crossed[i] = parents1[i]
            crossed[i + n_parents] = parents2[i]

    return crossed[:population_size, :].astype(int)


def pmx_single(
    vector1: VectorLike,
    vector2: VectorLike,
    random_state: Optional[RNGLike] = None,
) -> VectorLike:
    """
    Core PMX operation for a single pair of parents.

    Original implementation found in
    https://github.com/cosminmarina/A1_ComputacionEvolutiva

    Parameters
    ----------
    vector1, vector2 : VectorLike
        Two parent permutations (1-D arrays).
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    VectorLike
        One offspring permutation.
    """

    cross_point1 = random_state.integers(0, vector1.size - 2)
    cross_point2 = random_state.integers(cross_point1 + 1, vector1.size)

    # Segmentamos
    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    # Lo que no forma parte del segmento
    remaining = vector1[~seg_mask]
    segment = vector2[seg_mask]

    # Separamos en conjunto dentro y fuera del segmento del genotipo 2
    overlap = np.isin(remaining, segment)
    conflicting = remaining[overlap]
    no_conflict = np.sort(remaining[~overlap])

    # Añadimos los elementos sin conflicto (que no están dentro del segmento del genotipo 2)
    idx_no_conflict = np.where(np.isin(vector2, no_conflict))[0]
    child[idx_no_conflict] = no_conflict

    # Tratamos conflicto
    for elem in conflicting:
        pos = elem.copy()
        while pos != -1:
            genotype_in_pos = pos
            pos = child[np.where(vector2 == genotype_in_pos)][0]
        child[np.where(vector2 == genotype_in_pos)] = elem
    return child


def order_cross(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Order Crossover (OX) for permutation chromosomes.

    Builds offspring by preserving a randomly chosen segment from one
    parent and filling the remaining positions with the order of the other
    parent.  The pairing and probability logic is identical to
    :func:`pmx`.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape ``(N, M)``.
    fitness_array : VectorLike
        Fitness values (unused).
    pairing_method : str, optional
        Pairing strategy.
    crossover_prob : float, optional
        Probability of applying crossover to a pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape ``(N, M)``.
    """
    random_state = check_random_state(random_state)

    population_size, n_components = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

    crossed = np.empty((n_parents * 2, n_components))
    for i in range(n_parents):
        if random_state.random() < crossover_prob:
            crossed[i] = order_cross_single(parents1[i], parents2[i], random_state=random_state)
            crossed[i + n_parents] = order_cross_single(parents2[i], parents1[i], random_state=random_state)
        else:
            crossed[i] = parents1[i]
            crossed[i + n_parents] = parents2[i]

    return crossed[:population_size, :].astype(int)


def order_cross_single(
    vector1: VectorLike,
    vector2: VectorLike,
    random_state: Optional[RNGLike] = None,
) -> VectorLike:
    """
    Core OX operation for a single pair of parents.

    Parameters
    ----------
    vector1, vector2 : VectorLike
        Two parent permutations.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    VectorLike
        One offspring permutation.
    """
    cross_point1 = random_state.integers(0, vector1.size - 2)
    cross_point2 = random_state.integers(cross_point1, vector1.size)

    child = np.full_like(vector1, -1)
    range_vec = np.arange(vector1.size)
    seg_mask = (range_vec >= cross_point1) & (range_vec <= cross_point2)
    child[seg_mask] = vector1[seg_mask]

    remaining_unused = np.setdiff1d(vector2, child)
    remaining_unused = np.roll(remaining_unused, cross_point1)

    child[~seg_mask] = remaining_unused

    return child
