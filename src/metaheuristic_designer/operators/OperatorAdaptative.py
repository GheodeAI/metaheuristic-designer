from __future__ import annotations
from copy import copy
import numpy as np
from ..Operator import Operator
from ..encodings import AdaptionEncoding
from .OperatorMeta import OperatorMeta
from .OperatorNull import OperatorNull
from ..ParamScheduler import ParamScheduler


class OperatorAdaptative(Operator):
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
        param_operator: Operator,
        param_encoding: AdaptionEncoding,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the OperatorAdaptative class
        """

        super().__init__(params, base_operator.name + "-Adaptative")

        vecmask = np.concatenate([np.zeros(param_encoding.vecsize), np.ones(param_encoding.nparams)])
        null_op = OperatorNull()

        self.base_operator = base_operator
        self.param_operator = param_operator
        self.base_operator_split = OperatorMeta("Split", [base_operator, null_op], {"mask": vecmask})
        self.param_operator_split = OperatorMeta("Split", [null_op, param_operator], {"mask": vecmask})
        self.param_encoding = param_encoding

    def evolve(self, population, initializer=None):
        population.decode()

        raise NotImplementedError

    def step(self, progress: float):
        super().step(progress)
        self.base_operator.step(progress)
