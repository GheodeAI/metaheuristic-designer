from __future__ import annotations
from copy import copy
import numpy as np
# from ..operator import Operator, NullOperator, ExtendedOperator
from ..operator import Operator, NullOperator
from ..encoding import ExtendedEncoding
from .meta_operator import MetaOperator
from ..param_scheduler import ParamScheduler


# class AdaptativeOperator(Operator):
#     """
#     Operator class that allow algorithms to self-adapt by mutating the operator's parameters.

#     Parameters
#     ----------
#         base_operator: Operator
#             Operator that will be applied to the solution we are evaluating.
#         param_operator: Operator
#             Operator that will be applied to the parameters of the base operator.
#         param_encoding: AdaptionEncoding
#             Encoding that divides the genotype into the solution and the operator's parameters.
#         params: Union[ParamScheduler, dict]
#             Optional parameters that are used by the operator.
#         name: str
#             Name of the operator.
#     """

#     def __init__(
#         self,
#         base_operator: Operator,
#         param_operator: dict,
#         param_encoding: ExtendedEncoding,
#         params: ParamScheduler | dict = None,
#         name: str = None,
#     ):
#         """
#         Constructor for the OperatorAdaptative class
#         """

#         super().__init__(params, base_operator.name + "-Adaptative", use_params=True)

#         vecmask = np.concatenate([np.zeros(param_encoding.vecsize), np.ones(param_encoding.nparams)])
#         null_op = NullOperator()

#         self.base_operator = base_operator
#         self.param_operator = param_operator
#         self.main_operator = MetaOperator("Split", [base_operator, param_operator], {"mask": vecmask})
#         self.param_encoding = param_encoding

#     def evolve(self, population, initializer=None):
#         # Update operator parameters
#         params = self.param_encoding.decode_params(population.genotype_matrix)
#         self.base_operator.params.update(params)

#         # Evolve population
#         new_population = self.main_operator.evolve(population)

#         return new_population

#     def step(self, progress: float):
#         super().step(progress)
#         self.base_operator.step(progress)
#         self.param_operator.step(progress)


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

        print(operator_list)
        print(vecmask)
        self.main_operator = MetaOperator("Split", operator_list, {"mask": vecmask})
        self.vecmask = vecmask
        self.param_encoding = encoding 

    def evolve(self, population, initializer=None):
        return self.main_operator.evolve(population, initializer=initializer)

    def step(self, progress: float):
        super().step(progress)

        self.base_operator.step(progress)

        for _, op in self.param_operators.items():
            op.step(progress)

class AdaptativeOperator(ExtendedOperator):
    def evolve(self, population, initializer=None):
        print(self.vecmask)
        print(population.genotype_matrix)
        # Update operator parameters
        params = self.param_encoding.decode_params(population.genotype_matrix)
        self.base_operator.params.update(params)

        # Evolve population
        return super().evolve(population=population, initializer=initializer)