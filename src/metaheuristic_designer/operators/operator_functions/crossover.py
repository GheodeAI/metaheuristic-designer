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
    """Randomly partition the population into two equal-sized groups.

    The population rows are randomly permuted and split in half.
    When the population size is odd, the phantom index (one beyond the
    original length) is replaced by a random valid index from the
    opposite group, ensuring both groups have the same shape and no
    individual is lost.

    Parameters
    ----------
    population_array : MatrixLike
        2D array of shape (N, M) containing the current population.
    fitness_array : VectorLike
        Fitness values (unused in this split, kept for interface consistency).
    random_state : RNGLike
        Random number generator.

    Returns
    -------
    tuple[MatrixLike, MatrixLike]
        Two arrays ``(parents1, parents2)`` of equal shape ``(ceil(N/2), M)``
        representing the randomly paired groups.
    """
    population_size, _ = population_array.shape

    total_parent_count = 2 * np.ceil(population_size / 2).astype(int)
    chosen_parents1, chosen_parents2 = np.array_split(random_state.permutation(total_parent_count), 2)
    if population_size % 2 != 0:
        chosen_parents1[chosen_parents1 == total_parent_count - 1] = random_state.choice(chosen_parents2)
        chosen_parents2[chosen_parents2 == total_parent_count - 1] = random_state.choice(chosen_parents1)

    return population_array[chosen_parents1, :], population_array[chosen_parents2, :]


def stable_split(population_array: MatrixLike, fitness_array: VectorLike, random_state: RNGLike) -> tuple[MatrixLike, MatrixLike]:
    """Deterministically split the population into two halves preserving order.

    For an even population the first and second halves are returned directly.
    For an odd population the arrays are made equal by cyclically wrapping
    indices modulo the original size, i.e. the extra slot is filled with the
    first individual (index 0).  No randomness is used.

    Parameters
    ----------
    population_array : MatrixLike
        2D array of shape (N, M) containing the current population.
    fitness_array : VectorLike
        Fitness values (unused).
    random_state : RNGLike
        Random number generator (kept for API compatibility; not used).

    Returns
    -------
    tuple[MatrixLike, MatrixLike]
        Two arrays ``(parents1, parents2)`` of equal shape ``(ceil(N/2), M)``.
    """
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
    """Retrieve a pairing function by name.

    Parameters
    ----------
    method : str
        Key into the :data:`pairing_map` dictionary. Supported values are
        ``"random"`` and ``"stable"``.

    Returns
    -------
    Callable
        A function with signature
        ``(population_array, fitness_array, random_state) -> (parents1, parents2)``.

    Raises
    ------
    ValueError
        If *method* is not present in :data:`pairing_map`.
    """
    if method not in pairing_map:
        raise ValueError(f"Pairing strategy {method} doesn't exist.")

    return pairing_map[method]


# ------------------------------------------------
# Dual-parent crossover functions
# ------------------------------------------------
def k_point_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike,
    k: int = 1,
    pairing_method: str = "random",
    crossover_prob: float = 1,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """k-point crossover with per-pair probability.

    The population is split into paired halves using *pairing_method*.
    For each pair, *k* distinct crossover points are drawn uniformly from
    :math:`\{1, \dots, M-1\}` (sorted).  The alternating mask built from
    these points determines which parent contributes each gene.

    With probability *crossover_prob* the children are formed using the
    mask; otherwise the parents are copied unchanged.  The operator returns
    exactly *N* offspring.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike
        Fitness values (unused by this operator).
    k : int
        Number of crossover points. Must satisfy ``1 <= k < M``.
    pairing_method : str, optional
        Pairing strategy (``"random"`` or ``"stable"``).
    crossover_prob : float, optional
        Probability of applying the crossover to a given pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
    random_state = check_random_state(random_state)

    population_size, n_components = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    n_parents, _ = parents1.shape

    random_samples = random_state.random((n_parents, n_components - 1))
    random_order = np.argsort(random_samples, axis=1)
    cuts = random_order[:, :k] + 1

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
    """Uniform crossover with per-pair probability.

    For each gene of a pair, the contributing parent is chosen independently
    with probability 0.5.  The per-pair decision follows the same pattern as
    :func:`k_point_crossover`: with probability *crossover_prob* the pair
    undergoes crossover, otherwise the parents are kept unchanged.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
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
        Offspring population of shape (N, M).
    """
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
    """Arithmetic (averaged) crossover with per-pair probability.

    For a pair of parents :math:`p_1, p_2` the two children are defined as

    .. math::

        c_1 &= (1-\alpha)\,p_1 + \alpha\,p_2,\\
        c_2 &= (1-\alpha)\,p_2 + \alpha\,p_1,

    where :math:`\alpha \in [0,1]` controls the blend.  With probability
    *crossover_prob* the pair is recombined; otherwise the parents are
    copied unchanged.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike
        Fitness values (unused).
    pairing_method : str, optional
        Pairing strategy.
    alpha : float, optional
        Blend factor.  ``alpha=0`` gives pure parent 1; ``alpha=1`` gives
        pure parent 2; ``alpha=0.5`` gives the midpoint.
    crossover_prob : float, optional
        Probability of applying crossover to a pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
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
    """Blend crossover (BLX-:math:`\alpha`) with per-pair probability.

    For a pair of parents the smaller/larger values per gene are taken as
    :math:`x_{\min}` and :math:`x_{\max}`.  Each child gene is then sampled
    uniformly from the expanded interval

    .. math::

        [x_{\min} - \alpha\,(x_{\max}-x_{\min}),\;
         x_{\max} + \alpha\,(x_{\max}-x_{\min})].

    The pair undergoes crossover with probability *crossover_prob*; otherwise
    the parents are kept intact.

    Reference
    ---------
    Eshelman & Schaffer (1993): Real-coded genetic algorithms and interval
    schemata.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike
        Fitness values (unused).
    pairing_method : str, optional
        Pairing strategy.
    alpha : float, optional
        Expansion factor (>=0).
    crossover_prob : float, optional
        Probability of applying crossover to a pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
    random_state = check_random_state(random_state)

    population_size, _ = population_array.shape

    pairing_fn = create_pairing_fn(pairing_method)
    parents1, parents2 = pairing_fn(population_array, fitness_array, random_state)
    full_parents = np.concatenate([parents1, parents2], axis=0)
    n_parents, _ = parents1.shape

    lowest_parent = np.minimum(parents1, parents2)
    highest_parent = np.maximum(parents1, parents2)
    lower_bound = np.tile(lowest_parent - alpha * (highest_parent - lowest_parent), (2, 1))
    upper_bound = np.tile(highest_parent + alpha * (highest_parent - lowest_parent), (2, 1))

    crossed = random_state.uniform(lower_bound, upper_bound)
    pair_mask = np.tile(random_state.random(n_parents) < crossover_prob, 2)[:, None]

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
    """Simulated Binary Crossover (SBX) with per-pair probability.

    For a pair of parents :math:`p_1, p_2` the children are computed as

    .. math::

        c_1 &= 0.5(p_1+p_2) - 0.5\,\beta\,|p_1-p_2|,\\
        c_2 &= 0.5(p_1+p_2) + 0.5\,\beta\,|p_1-p_2|,

    where the spread factor :math:`\beta` is drawn from a polynomial
    distribution with index :math:`\eta`:

    .. math::

        \beta = \begin{cases}
            (2u)^{1/(\eta+1)}, & u \le 0.5,\\
            \bigl(\frac{1}{2(1-u)}\bigr)^{1/(\eta+1)}, & u > 0.5,
        \end{cases}

    with :math:`u \sim \mathcal{U}(0,1)`.  Larger *eta* keeps children
    closer to the parents.

    Reference
    ---------
    Deb & Agrawal (1995): Simulated binary crossover for continuous search
    space.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike
        Fitness values (unused).
    pairing_method : str, optional
        Pairing strategy.
    eta : float, optional
        Distribution index for the spread factor (>=0).
    crossover_prob : float, optional
        Probability of applying crossover to a pair.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
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
    mask1 = random_values <= 0.5
    mask2 = random_values > 0.5
    spread_factor[mask1] = (2 * random_values[mask1]) ** exp_factor
    spread_factor[mask2] = (0.5 / (1 - random_values[mask2])) ** exp_factor

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
    """Bitwise XOR crossover for binary-valued populations.

    For a pair of parents :math:`p_1, p_2` the two children are

    .. math::

        c_1 &= p_1 \oplus p_2,\\
        c_2 &= p_1 \oplus \neg p_2,

    where :math:`\oplus` denotes bitwise XOR and :math:`\neg` is bitwise
    NOT.  This operator is intended for Boolean arrays (0/1).  With
    probability *crossover_prob* the pair is crossed; otherwise the parents
    are kept unchanged.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).  Should be of a Boolean or integer type
        where bitwise operations are meaningful.
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
        Offspring population of shape (N, M).
    """
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
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """Multi-parent discrete crossover (uniform scanning).

    For each of the *N* offspring, *k* parents are drawn (with or without
    replacement) from the whole population.  Every gene of the offspring is
    then taken uniformly at random from one of those *k* parents.

    With probability *crossover_prob* an offspring is produced by
    recombination; otherwise it is a direct copy of the original individual
    at the same index.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike, optional
        Fitness values (unused).
    k : int, optional
        Number of parents per offspring.
    crossover_prob : float, optional
        Probability of applying crossover to an individual.
    replace : bool, optional
        If False (default), the *k* parents are distinct (no replacement).
        If True, parents are sampled independently (with replacement).
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
    random_state = check_random_state(random_state)
    population_size, n_components = population_array.shape

    if not replace:
        parent_idx = np.argsort(random_state.random((population_size, population_size)), axis=1)[:, :k]
    else:
        parent_idx = random_state.integers(0, population_size, size=(population_size, k))

    parents_selected = population_array[parent_idx]

    gene_choice = random_state.integers(0, k, size=(population_size, n_components))
    crossed = np.take_along_axis(parents_selected, gene_choice[:, None, :], axis=1).squeeze(axis=1)

    cross_mask = (random_state.random(population_size) < crossover_prob)[:, None]
    offspring = np.where(cross_mask, crossed, population_array)
    return offspring


def multiparent_intermediate_crossover(
    population_array: MatrixLike,
    fitness_array: VectorLike = None,
    k: int = 3,
    crossover_prob: float = 1.0,
    replace: bool = False,
    random_state: Optional[RNGLike] = None,
) -> MatrixLike:
    """Multi-parent intermediate crossover (averaging recombination).

    For each offspring, *k* parents are drawn (with or without replacement).
    The offspring is the arithmetic mean of those *k* parents.

    With probability *crossover_prob* the offspring is the averaged vector;
    otherwise it is the original individual at the same index.

    Parameters
    ----------
    population_array : MatrixLike
        Population of shape (N, M).
    fitness_array : VectorLike, optional
        Fitness values (unused).
    k : int, optional
        Number of parents per offspring.
    crossover_prob : float, optional
        Probability of applying crossover to an individual.
    replace : bool, optional
        If False (default), the *k* parents are distinct.
        If True, parents are sampled with replacement.
    random_state : RNGLike, optional
        Random number generator.

    Returns
    -------
    MatrixLike
        Offspring population of shape (N, M).
    """
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
