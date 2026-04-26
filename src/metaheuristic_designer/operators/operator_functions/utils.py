from copy import copy
from dataclasses import dataclass, field
import numpy as np
from ...population import Population
from ...initializer import Initializer


def dummy_op(population_matrix, _fitness_array, f=0):
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

        return population.update_genotype_matrix(
            self.operator_fn(copy(population.genotype_matrix), population.fitness, random_state, **modified_kwargs)
        )


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

        return population.update_genotype_matrix(self.operator_fn(population.genotype_matrix, initializer, random_state**modified_kwargs))


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

        return population.update_genotype_matrix(self.operator_fn(population.genotype_matrix, random_state, **modified_kwargs))


@dataclass
class OperatorSwarmFuncDef:
    """ """

    operator_fn: callable
    params: dict = field(default_factory=dict)
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, random_state, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)

        return population.update_genotype_matrix(self.operator_fn(population, random_state, **modified_kwargs))
