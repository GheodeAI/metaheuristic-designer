from copy import copy
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ...utils import ScalarLike, VectorLike
from ...population import Population
from ...initializer import Initializer


def dummy_op(
    population_matrix: np.ndarray, fitness_array: np.ndarray, random_state: Optional[np.random.Generator] = None, f: ScalarLike = 0
) -> np.ndarray:
    """Return a matrix of constant value *f* with the same shape as the input.

    This operator is intended **only for debugging and testing**.  It ignores
    the population contents and produces an array where every gene equals
    *f*.

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)`` (ignored, but its shape is used).
    _fitness_array : np.ndarray
        Fitness values (unused, kept for interface compatibility).
    random_state : np.random.Generator, optional
        Random number generator (unused).
    f : ScalarLike, optional
        Value to fill the array with. Default is ``0``.

    Returns
    -------
    np.ndarray
        A ``(N, M)`` array filled with the constant *f*.
    """
    return np.full_like(population_matrix, f)


def add_const(
    population_matrix: np.ndarray, fitness_array: np.ndarray, random_state: Optional[np.random.Generator] = None, f: VectorLike | ScalarLike = 0
) -> np.ndarray:
    """Add a constant (or vector) to every gene of the population.

    Useful as a trivial baseline operator: it simply returns
    ``population_matrix + f``, where *f* can be a scalar (added to all genes)
    or a vector (added per gene).

    Parameters
    ----------
    population_matrix : np.ndarray
        2-D array of shape ``(N, M)``.
    fitness_array : np.ndarray
        Fitness values (unused).
    random_state : np.random.Generator, optional
        Random number generator (unused).
    f : VectorLike or ScalarLike, optional
        Value(s) to add.  A scalar is broadcast to every gene; a 1-D array
        of length *M* is added per gene. Default is ``0``.

    Returns
    -------
    np.ndarray
        ``population_matrix + f``.
    """
    return population_matrix + f


# ------------------------------------------------------------------
# Wrapper dataclasses – they adapt your plain functions into the
# Operator interface used by the GA loop.
# ------------------------------------------------------------------


@dataclass
class OperatorFnDef:
    """Bridge a matrix-to-matrix operator function into an :class:`Operator`.

    This wrapper accepts a callable that operates on a genotype matrix,
    fitness array, and random state, and turns it into an object that can
    be used directly on a :class:`Population`.  It merges user-supplied
    keyword arguments with stored defaults and forced parameters, then
    invokes the underlying function and updates the population's genotype.

    Parameters
    ----------
    operator_fn : callable
        Function with signature
        ``(population_matrix, fitness_array, random_state, **kwargs) -> np.ndarray``.
    params : dict, optional
        Default keyword arguments for the operator.
    forced_params : dict, optional
        Keyword arguments that **always** override user-supplied ones.
    """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state: Optional[np.random.Generator] = None, **kwargs) -> Population:
        """Execute the wrapped operator and return a new population.

        Parameters
        ----------
        population : Population
            The current population (its genotype matrix and fitness are used).
        initializer : Initializer
            Population initializer (forwarded to the operator if needed).
        random_state : np.random.Generator, optional
            Random number generator.
        **kwargs
            Additional keyword arguments passed to the operator function.
            They are combined with, but overridden by, :attr:`forced_params`.

        Returns
        -------
        Population
            A new population with the genotype updated by the operator.
        """
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(
            self.operator_fn(copy(population.genotype_matrix), population.fitness, random_state=random_state, **modified_kwargs)
        )


@dataclass
class OperatorRandomDef:
    """Bridge a random-style operator function into an :class:`Operator`.

    This wrapper is intended for operators that **replace** the genotype
    with entirely new random values (e.g., uniform sampling, initializer-based
    reset).  It passes the population's genotype matrix, the initializer,
    and the random state to the underlying function.

    Parameters
    ----------
    operator_fn : callable
        Function with signature
        ``(population_matrix, initializer, random_state, **kwargs) -> np.ndarray``.
    params : dict, optional
        Default keyword arguments.
    forced_params : dict, optional
        Keyword arguments that override any user-supplied ones.
    """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state: Optional[np.random.Generator] = None, **kwargs) -> Population:
        """Execute the random operator and return a new population.

        Parameters
        ----------
        population : Population
            The current population (its genotype matrix is used as a shape reference).
        initializer : Initializer
            Population initializer, forwarded to the operator.
        random_state : np.random.Generator, optional
            Random number generator.
        **kwargs
            Additional keyword arguments for the operator function.

        Returns
        -------
        Population
            A new population where the genotype has been replaced by
            the operator's output.
        """
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(self.operator_fn(population.genotype_matrix, initializer, random_state=random_state, **modified_kwargs))


@dataclass
class ObtainStatisticDef:
    """Wrap a statistic‑computing function into an :class:`Operator`.

    This adapter is used for functions that compute a single summary
    vector (e.g., population mean, median, standard deviation) and store
    it as the new genotype (usually a single-row population).

    Parameters
    ----------
    operator_fn : callable
        Function with signature
        ``(population_matrix, random_state, **kwargs) -> np.ndarray``.
    params : dict, optional
        Default keyword arguments.
    forced_params : dict, optional
        Keyword arguments that override user-supplied ones.
    """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state: Optional[np.random.Generator] = None, **kwargs) -> Population:
        """Compute a statistic and replace the population’s genotype.

        Parameters
        ----------
        population : Population
            The current population (its genotype matrix is analysed).
        initializer : Initializer
            Not used by this wrapper (included for interface compatibility).
        random_state : np.random.Generator, optional
            Random number generator.
        **kwargs
            Additional keyword arguments for the operator function.

        Returns
        -------
        Population
            A population whose genotype has been replaced by the
            computed statistic.
        """
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(self.operator_fn(population.genotype_matrix, random_state, **modified_kwargs))


@dataclass
class OperatorSwarmDef:
    """Bridge a swarm operator function into an :class:`Operator`.

    This wrapper is designed for operators that directly receive the
    whole :class:`Population` object and the initializer, and are
    responsible for returning an updated :class:`Population` **themselves**
    (e.g., PSO operators that need access to historical bests).

    Parameters
    ----------
    operator_fn : callable
        Function with signature
        ``(population, initializer, random_state, **kwargs) -> Population``.
    params : dict, optional
        Default keyword arguments.
    forced_params : dict, optional
        Keyword arguments that override user-supplied ones.
    """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state: np.random.Generator, **kwargs) -> Population:
        """Execute the swarm operator and return the new population.

        Parameters
        ----------
        population : Population
            The current population, passed directly to the operator.
        initializer : Initializer
            Population initializer, passed to the operator.
        random_state : np.random.Generator
            Random number generator (required for swarm operators).
        **kwargs
            Additional keyword arguments for the operator function.

        Returns
        -------
        Population
            The population returned by the operator (may be the same
            object or a new one, depending on the operator).
        """
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return self.operator_fn(population, initializer, random_state=random_state, **modified_kwargs)
