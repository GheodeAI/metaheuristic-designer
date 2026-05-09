from copy import copy
from dataclasses import dataclass, field
import numpy as np
from ...utils import ScalarLike, VectorLike
from ...population import Population
from ...initializer import Initializer


def dummy_op(population_matrix, _fitness_array, random_state=None, f: ScalarLike = 0):
    """
    Replaces the vector with one consisting of all the same value

    Only for testing, not useful for real applications

    Parameters
    ----------
    population_matrix: numpy.array
        Matrix containing the set of tentative solutions.
    _fitness_array: numpy.array
        Array containing the fitness of the individuals. (unused, kept for compatibility with other operator functions).
    F: float, optional
        Value to set as the value for the components.

    Returns
    -------
        Vector with all the components consisting of the same value.
    """

    return np.full_like(population_matrix, f)


def add_const(population_matrix, fitness_array, random_state=None, f: VectorLike | ScalarLike = 0):
    return population_matrix + f


@dataclass
class OperatorVectorDef:
    """ """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state=None, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(self.operator_fn(copy(population.genotype_matrix), population.fitness, random_state=random_state, **modified_kwargs))


@dataclass
class OperatorRandomDef:
    """ """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state=None, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(self.operator_fn(population=population.genotype_matrix, initializer=initializer, random_state=random_state, **modified_kwargs))


@dataclass
class ObtainStatisticDef:
    """ """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state=None, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype(self.operator_fn(population.genotype_matrix, random_state, **modified_kwargs))


@dataclass
class OperatorSwarmDef:
    """ """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, random_state, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return self.operator_fn(population, initializer, random_state=random_state, **modified_kwargs)
