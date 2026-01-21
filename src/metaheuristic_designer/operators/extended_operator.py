from __future__ import annotations
import numpy as np
from ..operator import Operator, NullOperator
from ..encodings import ExtendedEncoding
from .composite_operator import CompositeOperator


class ExtendedOperator(Operator):
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
        param_operators: dict,
        encoding: ExtendedEncoding,
        params: ParamScheduler | dict = None,
        name: str = None,
    ):
        """
        Constructor for the OperatorAdaptative class
        """

        super().__init__(params=params, name=base_operator.name, encoding=encoding)

        vecmask = np.zeros(encoding.vecsize + encoding.nparams)

        counter = encoding.vecsize
        for idx, (_, param_num) in enumerate(encoding.param_sizes):
            vecmask[counter:counter + param_num] = idx + 1
            counter = counter+param_num

        self.base_operator = base_operator
        self.param_operators = param_operators
        operator_list = [base_operator] + [param_operators[param_name] for idx, (param_name, _) in enumerate(encoding.param_sizes)]

        self.main_operator = SplitOperator(operator_list, {"mask": vecmask})
        self.vecmask = vecmask
        self.param_encoding = encoding 

    def evolve(self, population, initializer=None):
        return self.main_operator.evolve(population, initializer=initializer)

    def step(self, progress: float):
        super().step(progress)

        self.base_operator.step(progress)

        for _, op in self.param_operators.items():
            op.step(progress)
