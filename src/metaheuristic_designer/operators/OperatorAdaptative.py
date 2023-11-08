from __future__ import annotations
import random
from ..Operator import Operator
from ..encodings import AdaptionEncoding
from .OperatorMeta import OperatorMeta
from .OperatorReal import OperatorReal
from .OperatorNull import OperatorNull
from copy import copy
import numpy as np
import enum
from enum import Enum


class OperatorAdaptative(Operator):
    """
    Operator class that utilizes a list of operators to modify individuals.

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    op_list: List[Operator]
        List of operators that will be used.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(
        self,
        base_operator: Operator,
        param_operator: Operator,
        param_encoding: AdaptionEncoding,
        params: Union[ParamScheduler, dict] = None,
        name: str = None,
    ):
        """
        Constructor for the Operator class
        """

        super().__init__(params, base_operator.name + "-Adaptative")

        vecmask = np.concatenate([np.zeros(param_encoding.vecsize), np.ones(param_encoding.nparams)])
        null_op = OperatorNull()

        self.base_operator = base_operator
        self.param_operator = param_operator
        self.base_operator_split = OperatorMeta("Split", [base_operator, null_op], {"mask": vecmask})
        self.param_operator_split = OperatorMeta("Split", [null_op, param_operator], {"mask": vecmask})
        self.param_encoding = param_encoding

    def evolve(self, indiv, population, objfunc, global_best, initializer=None):
        # Evolve only parameters
        indiv_conf_param = self.param_operator_split.evolve(indiv, population, objfunc, global_best, initializer)

        # Apply parameters to operator
        indiv_params = self.param_encoding.decode_param(indiv_conf_param.genotype)
        self.base_operator.params.update(indiv_params)

        # Evolve solution
        new_indiv = self.base_operator_split.evolve(indiv_conf_param, population, objfunc, global_best, initializer)
        return new_indiv

    def step(self, progress: float):
        super().step(progress)
        self.base_operator.step(progress)
