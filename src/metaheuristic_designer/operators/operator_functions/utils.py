from dataclasses import dataclass, field
from copy import copy
from ...population import Population
from ...initializer import Initializer

@dataclass
class OperatorVectorDef:
    """

    """

    operator_fn: callable
    params: dict = field(default_factory=dict) 
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)
        
        return population.update_genotype_matrix(self.operator_fn(copy(population.genotype_matrix), population.fitness, **modified_kwargs))


@dataclass
class OperatorRandomDef:
    """

    """

    operator_fn: callable
    params: dict = field(default_factory=dict) 
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)
        
        return population.update_genotype_matrix(self.operator_fn(population.genotype_matrix, initializer, **modified_kwargs))

@dataclass
class ObtainStatisticDef:
    """

    """

    operator_fn: callable
    params: dict = field(default_factory=dict) 
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, initializer: Initializer, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)
        
        return population.update_genotype_matrix(self.operator_fn(population.genotype_matrix, **modified_kwargs))

@dataclass
class OperatorSwarmFuncDef:
    """

    """

    operator_fn: callable
    params: dict = field(default_factory=dict) 
    forced_params: dict = field(default_factory=dict)

    def __call__(self, population: Population, **kwargs):
        modified_kwargs = {}
        modified_kwargs.update(self.params)
        modified_kwargs.update(kwargs)
        modified_kwargs.update(self.forced_params)
        
        return population.update_genotype_matrix(self.operator_fn(population.genotype_matrix, **modified_kwargs))