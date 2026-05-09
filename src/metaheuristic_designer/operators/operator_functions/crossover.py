import math
import logging
from typing import Callable, Optional
import numpy as np
from ...utils import MatrixLike, RNGLike, VectorLike, check_random_state

logger = logging.getLogger(__name__)


# ------------------------------------------------
# Splitting functions for dual-parent crossovers
# ------------------------------------------------
def random_split(population_array: MatrixLike, fitness_array: VectorLike, random_state: RNGLike) -> tuple[MatrixLike, MatrixLike]:
    population_size, _ = population_array.shape

    total_parent_count = 2 * np.ceil(population_size / 2).astype(int)
    chosen_parents1, chosen_parents2 = np.array_split(random_state.permutation(total_parent_count), 2)
    if population_size % 2 != 0:
        chosen_parents1[chosen_parents1 == total_parent_count - 1] = random_state.choice(chosen_parents2)
        chosen_parents2[chosen_parents2 == total_parent_count - 1] = random_state.choice(chosen_parents1)

    return population_array[chosen_parents1, :], population_array[chosen_parents2, :]


def stable_split(population_array: MatrixLike, fitness_array: VectorLike, random_state: RNGLike) -> tuple[MatrixLike, MatrixLike]:
    population_size, _ = population_array.shape

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

    population_size, n_components = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

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
    random_state = check_random_state(random_state)

    population_size, n_components = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

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

    population_size, _ = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = (1 - alpha) * parents1 + alpha * parents2
    crossed2 = (1 - alpha) * parents2 + alpha * parents1

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


def blend_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    pairing_method: str = "random",
    alpha: float = 0.5,
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    random_state = check_random_state(random_state)

    population_size, _ = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    full_parents = np.concatenate([parents1, parents2], axis=0)
    n_parents, _ = parents1.shape

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

    population_size, _ = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

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
    random_state = check_random_state(random_state)

    population_size, _ = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

    pair_mask = (random_state.random(n_parents) < crossover_prob)[:, None]

    crossed1 = parents1 ^ parents2
    crossed2 = parents1 ^ ~parents2

    offspring1 = np.where(pair_mask, crossed1, parents1)
    offspring2 = np.where(pair_mask, crossed2, parents2)

    offspring = np.concatenate([offspring1, offspring2], axis=0)

    return offspring[:population_size, :]


# ------------------------------------------------
# Multiparent crossover
# ------------------------------------------------
def multiparent_discrete_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike = None,
    k: int = 3,
    crossover_prob: float = 1.0,
    replace: bool = False,
    random_state: Optional[RNGLike] = None
):
    random_state = check_random_state(random_state)
    population_size, n_components = population_array.shape

    if not replace:
        parent_idx = np.argsort(random_state.random((population_size, population_size)), axis=1)[:, :k]
    else:
        parent_idx = random_state.integers(0, population_size, size=(population_size, k))

    parents_selected = population_array[parent_idx]

    gene_choice = random_state.integers(0, k, size=(population_size, n_components))
    crossed = np.take_along_axis(
        parents_selected, gene_choice[:, None, :], axis=1
    ).squeeze(axis=1)

    cross_mask = (random_state.random(population_size) < crossover_prob)[:, None]
    offspring = np.where(cross_mask, crossed, population_array)
    return offspring


def multiparent_intermediate_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike = None,
    k: int = 3,
    crossover_prob: float = 1.0,
    replace: bool = False,
    random_state: Optional[RNGLike] = None
):
    random_state = check_random_state(random_state)
    population_size, _ = population_array.shape

    if not replace:
        parent_idx = np.argsort(random_state.random((population_size, population_size)), axis=1)[:, :k]
    else:
        parent_idx = random_state.integers(0, population_size, size=(population_size, k))

    parent_set = population_array[parent_idx]

    crossed = parent_set.mean(axis=1)

    cross_mask = (random_state.random(population_size) < crossover_prob)[:, None]
    offspring = np.where(cross_mask, crossed, population_array)
    return offspring
