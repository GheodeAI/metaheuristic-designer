"""Initializers that insert predefined solutions into the population."""

from __future__ import annotations
from typing import Iterable, Optional
from ..population import Population
from ..utils import MatrixLike, VectorLike, RNGLike
from ..initializer import Initializer
from .composite_initializer import CompositeInitializer, FixedCompositeInitializer
from .direct_initializer import DirectInitializer


class SeededInitializer(CompositeInitializer):
    """
    Initializer that inserts a predefined solution with a given probability.

    With probability `insert_prob`, a randomly chosen solution from the
    provided set is used; otherwise a random individual is generated
    by the fallback initializer.

    Parameters
    ----------
    default_init : Initializer
        Fallback initializer for random individuals.
    solutions : Population, Iterable[VectorLike] or MatrixLike
        Set of predefined solutions to draw from.
    insert_prob : float, optional
        Probability of using a predefined solution (default 0.1).
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(
        self,
        default_init: Initializer,
        solutions: Population | Iterable[VectorLike] | MatrixLike,
        insert_prob: float = 0.1,
        population_size: int = None,
        rng: Optional[RNGLike] = None,
    ):
        assert len(solutions) > 0, "The solution set should not be empty."
        if isinstance(solutions, Population):
            inferred_dimension = solutions.genotype_matrix.shape[1]
        else:
            inferred_dimension = solutions[0].shape[0]
        assert inferred_dimension == default_init.dimension, "Dimension of the default initializer and the solutions must match."

        if population_size is None:
            population_size = len(solutions)

        seed_initializer = DirectInitializer(
            default_init=default_init, solutions=solutions, encoding=default_init.encoding, rng=rng
        )

        super().__init__(
            dimension=inferred_dimension,
            initializers=[seed_initializer, default_init],
            weights=[insert_prob, 1 - insert_prob],
            population_size=population_size,
            rng=rng,
        )


class FixedSeededInitializer(FixedCompositeInitializer):
    """
    Initializer that inserts a fixed number of predefined solutions.

    The first `n_to_insert` individuals generated are taken from the
    solution set (cycled if necessary); the remaining are created by
    the fallback initializer.

    Parameters
    ----------
    default_init : Initializer
        Fallback initializer for random individuals.
    solutions : Population, Iterable[VectorLike] or MatrixLike
        Set of predefined solutions to draw from.
    n_to_insert : int, optional
        Exact number of predefined solutions to insert.  Defaults to
        the size of the solution set.
    rng : RNGLike, optional
        Random number generator.
    """

    def __init__(
        self,
        default_init: Initializer,
        solutions: Population | Iterable[VectorLike] | MatrixLike,
        n_to_insert: int = None,
        population_size: int = None,
        rng: Optional[RNGLike] = None,
    ):
        assert len(solutions) > 0, "The solution set should not be empty."
        if isinstance(solutions, Population):
            inferred_dimension = solutions.dimension
        else:
            inferred_dimension = solutions[0].shape[0]
        assert inferred_dimension == default_init.dimension, "Dimension of the default initializer and the solutions must match."

        if n_to_insert is None:
            n_to_insert = len(solutions)

        if population_size is None:
            population_size = len(solutions)

        seed_initializer = DirectInitializer(
            default_init=default_init, solutions=solutions, encoding=default_init.encoding, rng=rng
        )

        super().__init__(
            dimension=inferred_dimension,
            population_size=population_size,
            initializers=[seed_initializer, default_init],
            amounts=[n_to_insert, max(population_size - n_to_insert, 0)],
            rng=rng,
        )
