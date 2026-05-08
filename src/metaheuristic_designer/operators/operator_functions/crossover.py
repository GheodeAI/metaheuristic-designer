import math
import logging
from tkinter import W
from typing import Callable, Optional
import numpy as np
from ...utils import MatrixLike, RNGLike, VectorLike, check_random_state

logger = logging.getLogger(__name__)


# ------------------------------------------------
# Splitting functions for dual-parent crossovers
# ------------------------------------------------
def random_split(population_array: MatrixLike, _fitness_array: VectorLike, random_state: RNGLike) -> tuple[MatrixLike, MatrixLike]:
    population_size = population_array.shape[0]

    total_parent_count = 2 * np.ceil(population_size / 2).astype(int)
    chosen_parents1, chosen_parents2 = np.array_split(random_state.permutation(total_parent_count), 2)
    if population_size % 2 != 0:
        chosen_parents1[chosen_parents1 == total_parent_count - 1] = random_state.choice(chosen_parents2)
        chosen_parents2[chosen_parents2 == total_parent_count - 1] = random_state.choice(chosen_parents1)

    return population_array[chosen_parents1, :], population_array[chosen_parents2, :]


def stable_split(population_array: MatrixLike, _fitness_array: VectorLike, random_state: RNGLike) -> tuple[MatrixLike, MatrixLike]:
    population_size = population_array.shape[0]

    half_size = math.ceil(population_size / 2)
    if population_size % 2 != 0:
        first_half, second_half = np.array_split(np.arange(half_size * 2) % population_size, 2)
        return population_array[first_half], population_array[second_half]
    else:
        return population_array[:half_size], population_array[half_size:]


# fmt: off
pairing_map = {
    "random": random_split,
    "stable": stable_split
}
# fmt: on


def create_pairing_fn(method: str) -> Callable:
    if method not in pairing_map:
        raise ValueError(f"Pairing strategy {method} doesn't exist.")

    return pairing_map[method]


# ------------------------------------------------
# Dual-parent crossover functions
# ------------------------------------------------
def k_point_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    k: int,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]
    n_components = population_array.shape[1]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents = parents1.shape[0]

    cuts = random_state.choice(n_components - 1, size=(n_parents, k), replace=False) + 1
    cuts.sort(axis=1)

    delta = np.zeros((n_parents, n_components), dtype=int)
    delta[np.arange(n_parents)[:, None], cuts] = 1

    cross_mask = np.cumsum(delta, axis=1) % 2 == 0
    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = np.where(cross_mask, parents1, parents2)
    crossed2 = np.where(cross_mask, parents2, parents1)

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    crossed = np.concatenate([offspring1, offspring2], axis=0)

    return crossed[:population_size, :]


def uniform_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Performs an uniform crossover between one half of the population_array and the rest.
    """

    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]
    n_components = population_array.shape[1]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents = parents1.shape[0]

    cross_mask = random_state.random((n_parents, n_components)) < 0.5
    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = np.where(cross_mask, parents1, parents2)
    crossed2 = np.where(cross_mask, parents2, parents1)

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


def averaged_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    alpha: float = 0.5,
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents = parents1.shape[0]

    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = (1 - alpha) * parents1 + alpha * parents2
    crossed2 = (1 - alpha) * parents2 + alpha * parents1

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


def blx_alpha_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    alpha: float = 0.5,
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    full_parents = np.concatenate([parents1, parents2], axis=0)
    n_parents = parents1.shape[0]

    lowest_parent = np.minimum(parents1, parents2)
    highest_parent = np.maximum(parents1, parents2)
    lower_bound = np.tile(lowest_parent - alpha * (highest_parent - lowest_parent), 2)
    upper_bound = np.tile(highest_parent + alpha * (highest_parent - lowest_parent), 2)

    crossed = random_state.uniform(lower_bound, upper_bound)
    pair_mask = np.tile(random_state.random(n_parents) < crossover_prob, 2)

    offspring = np.where(pair_mask, crossed, full_parents)

    return offspring[:population_size, :]


def sbx_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    eta: float = 0.5,
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    eps = np.finfo(population_array.dtype).tiny

    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents = parents1.shape[0]

    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    random_values = np.clip(random_state.random(parents1.shape), eps, 1 - eps)

    exp_factor = 1 / (eta + 1)
    spread_factor = np.empty_like(parents1)
    spread_factor[random_values <= 0.5] = (2 * random_values) ** exp_factor
    spread_factor[random_values > 0.5] = (0.5 / (1 - random_values)) ** exp_factor

    crossed1 = 0.5 * (parents1 + parents2) - 0.5 * spread_factor * np.abs(parents1 - parents2)
    crossed2 = 0.5 * (parents1 + parents2) + 0.5 * spread_factor * np.abs(parents1 - parents2)

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


def bitwise_xor_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Applies the XOR operation between each component of individuals in the population_array. The crossover is performed
    between the first and second half of the population_array
    """

    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents = parents1.shape[0]

    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = parents1 ^ parents2
    crossed2 = parents1 ^ ~parents2

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


# ------------------------------------------------
# Splitting functions for multi-parent crossovers
# ------------------------------------------------
def random_multiparent_split(
    population_array: MatrixLike, _fitness_array: VectorLike, n_parents: int, random_state: RNGLike
) -> tuple[MatrixLike, ...]:
    population_size = population_array.shape[0]

    total_parent_count = n_parents * np.ceil(population_size / n_parents).astype(int)
    parent_list = np.array_split(random_state.permutation(total_parent_count), n_parents)
    if population_size % n_parents != 0:
        remainder = population_size % n_parents
        n_phantom = np.ceil(population_size / n_parents).astype(int) - remainder
        for chosen_parents_i in parent_list:
            chosen_parents_i[chosen_parents_i >= n_phantom] = random_state.integers(0, population_size, remainder)

    return tuple(population_array[idxs, :] for idxs in parent_list)


def stable_multiparent_split(
    population_array: MatrixLike, _fitness_array: VectorLike, n_parents: int, random_state: RNGLike
) -> tuple[MatrixLike, ...]:
    population_size = population_array.shape[0]

    total_parent_count = n_parents * np.ceil(population_size / n_parents).astype(int)
    parent_list = np.array_split(np.arange(total_parent_count) % population_size, n_parents)

    return tuple(population_array[idxs, :] for idxs in parent_list)


# fmt: off
group_selection_map = {
    "random": random_multiparent_split,
    "stable": stable_multiparent_split
}
# fmt: on


def create_group_selection_fn(method: str) -> Callable:
    if method not in group_selection_map:
        raise ValueError(f"Pairing strategy {method} doesn't exist.")

    return group_selection_map[method]


# ------------------------------------------------
# Multiparent crossover
# ------------------------------------------------
def multiparent_discrete_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    n_parents: int = 3,
    grouping_method: str = "random",
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """
    Performs a multipoint crossover between 'n_indiv' randomly chosen individuals for each member of the population_array.
    """

    random_state = check_random_state(random_state)

    population_size = population_array.shape[0]
    n_components = population_array.shape[1]

    if n_parents < population_size:
        raise ValueError(f"Cannot perform recombination between {n_parents} individuals when there are only {population_size} parents.")

    group_selection_fn = create_group_selection_fn(grouping_method)
    parents = group_selection_fn(population_array, fitness_array, n_parents, random_state)
    n_parents = parents[0].shape[0]

    # full_parent_array = np.concatenate(parents, axis=0)

    indiv_chosen = np.tile(np.arange(population_array.shape[0]), (n_parents, 1))
    indiv_chosen = random_state.permuted(indiv_chosen, axis=1).T

    selection_mask = random_state.integers(0, n_parents, population_array.shape)

    components_chosen = indiv_chosen[np.arange(indiv_chosen.shape[0])[:, None], selection_mask]

    return population_array[components_chosen, np.arange(population_array.shape[1])]


def multiparent_intermediate_crossover(population_array, _fitness_array, N=3, random_state=None) -> MatrixLike:
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population_array.
    """

    random_state = check_random_state(random_state)

    n_indiv = np.minimum(N, population_array.shape[0])

    # TODO: individuals should be chosen with replacement
    for i in range(n_indiv):
        population_shuffled = population_array[random_state.permutation(population_array.shape[0]), :]
        population_array += population_shuffled

    return population_array / n_indiv
