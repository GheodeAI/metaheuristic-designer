from __future__ import annotations
from copy import copy
import numpy as np
from ..operator import Operator, NullOperator
from ..encoding import ExtendedEncoding
from .meta_operator import MetaOperator
from ..param_scheduler import ParamScheduler


class AdaptativeOperator(Operator):
    """
    Operator class that allow algorithms to self-adapt by mutating the operator's parameters.

    Parameters
    ----------
        base_operator: Operator
            Operator that will be applied to the solution we are evaluating.
        param_operator: Operator
            Operator that will be applied to the parameters of the base operator.
        param_encoding: AdaptionEncoding
            Encoding that divides the genotype into the solution and the operator's parameters.
        params: Union[ParamScheduler, dict]
            Optional parameters that are used by the operator.
        name: str
            Name of the operator.
    """

    def __init__(
        self,
        base_operator: Operator,
        param_operator: dict,
        param_encoding: ExtendedEncoding,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the OperatorAdaptative class
        """

        super().__init__(params, base_operator.name + "-Adaptative", use_params=True)

        vecmask = np.concatenate([np.zeros(param_encoding.vecsize), np.ones(param_encoding.nparams)])
        null_op = OperatorNull()

        self.base_operator = base_operator
        self.param_operator = param_operator
        self.main_operator = OperatorMeta("Split", [base_operator, param_operator], {"mask": vecmask})
        self.param_encoding = param_encoding

    def evolve(self, population, initializer=None):
        # Update operator parameters
        params = self.param_encoding.decode_params(population.genotype_matrix)
        self.base_operator.params.update(params)

        # Evolve population
        new_population = self.main_operator.evolve(population)

        return new_population

    def step(self, progress: float):
        super().step(progress)
        self.base_operator.step(progress)
        self.param_operator.step(progress)
