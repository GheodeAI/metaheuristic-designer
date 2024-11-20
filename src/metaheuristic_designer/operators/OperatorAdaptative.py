from __future__ import annotations
import random
from ..Operator import Operator
from ..encodings import AdaptionEncoding
from .OperatorMeta import OperatorMeta
from .OperatorVector import OperatorVector
from .OperatorNull import OperatorNull
from copy import copy
import numpy as np
import enum
from enum import Enum


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
        params: Union[ParamScheduler, dict] = None,
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
        new_population = [self.evolve_single(indiv, population, objfunc, initializer, global_best) for idx, indiv in enumerate(population)]
        return new_population
    
    def evolve_single(self, indiv, population, initializer=None):
        # Evolve only parameters
        indiv_conf_param = self.param_operator_split.evolve_single(indiv, population, objfunc, global_best, initializer)

        # Apply parameters to operator
        indiv_params = self.param_encoding.decode_param(indiv_conf_param.genotype)
        self.base_operator.params.update(indiv_params)

        # Evolve solution
        new_indiv = self.base_operator_split.evolve_single(indiv_conf_param, population, objfunc, global_best, initializer)
        return new_indiv

    def step(self, progress: float):
        super().step(progress)
        self.base_operator.step(progress)
