from __future__ import annotations
from typing import Any
from copy import copy
from abc import ABC, abstractmethod
import numpy as np
from .param_scheduler import ParamScheduler
from .encoding import Encoding, DefaultEncoding
from .objective_function import ObjectiveFunc
from .population import Population
from .initializer import Initializer


class Operator(ABC):
    """
    Abstract Operator class.

    This class modifies the genotype of one individual in order to perform some optimization task.

    Parameters
    ----------
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    encoding: Encoding, optional
        Postprocessing to the operator output.
    """

    _last_id = 0

    def __init__(self, params: ParamScheduler | dict = None, name: str = None, use_params : bool = False, encoding: Encoding = None):
        """
        Constructor for the Operator class.
        """

        self.id = Operator._last_id
        Operator._last_id += 1

        self.param_scheduler = None

        self.name = name

        self.use_params = use_params

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

        if params is None:
            self.params = {}
        elif params == "default":
            self.params = {
                "F": 0.5,
                "Cr": 0.8,
                "N": 5,
                "Nindiv": 5,
                "P": 0.1,
                "distrib": "gauss",
                "temp_ch": 10,
                "iter": 20,
                "Low": -10,
                "Up": 10,
                "p": 0.5,
                "mu": 2,
                "epsilon": 0.1,
                "tau": 0.1,
                "tau_multiple": 0.1,
                "a": 0.1,
                "b": 0.1,
                "d": 0.1,
                "g": 0.1,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "function": lambda x, y, z: x,
            }
        else:
            if "method" in params:
                params["method"] = params["method"].lower()

            if isinstance(params, ParamScheduler):
                self.param_scheduler = params
                self.params = self.param_scheduler.get_params()
            else:
                self.params = params

    def __call__(
        self,
        population: Population,
        initializer: Initializer = None,
    ) -> Population:
        """
        A shorthand for calling the 'evolve' method.
        """

        return self.evolve(population, initializer)

    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists.

        Parameters
        ----------
        progress: float
            Indicator of how close it the algorithm to finishing, 1 means the algorithm should be stopped.
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()

    def get_state(self) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Returns
        -------
        state: dict
            The complete state of the operator.
        """

        data = {"name": self.name}

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
            data["params"].pop("function", None)
        elif self.params:
            data["params"] = self.params
            data["params"].pop("function", None)

        return data

    @abstractmethod
    def evolve(
        self,
        population: Population,
        initializer: Initializer = None,
    ) -> Population:
        """
        Evolves an population using a given strategy.

        Parameters
        ----------
        population: Population
            The population that will be used.
        initializer: Initialize, optional
            The population initializer of the algorithm (used for randomly generating individuals).

        Returns
        -------
        new_population: Population
            The modified population.
        """


class NullOperator(Operator):
    """
    Operator class that returns the individual without changes.
    Surprisingly useful.

    Parameters
    ----------
    fn: callable
        Function that will be applied when operating on an individual.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, name: str = None):
        """
        Constructor for the OperatorNull class
        """

        if name is None:
            name = "Nothing"

        super().__init__({}, name)

    def evolve(self, population, *args):
        return copy(population)


class OperatorFromLambda(Operator):
    """
    Operator class that applies a custom operator specified as a function.

    Parameters
    ----------
    fn: callable
        Function that will be applied when operating on an individual.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, operator_fn: callable, params: ParamScheduler | dict = None, name: str = None, vectorized: bool = True):
        """
        Constructor for the OperatorLambda class
        """

        self.operator_fn = operator_fn
        self.vectorized = vectorized

        if name is None:
            name = operator_fn.__name__

        super().__init__(params, name)

    def evolve(self, population, initializer=None):
        if self.vectorized:
            population_matrix = copy(population.genotype_matrix)
            population_matrix = self.operator_fn(population_matrix, **self.params)
        else:
            population_cpy = copy(population)
            population_matrix = np.asarray([self.operator_fn(indiv, population, initializer) for indiv in population_cpy])
        # print(population_matrix)

        # return population.update_genotype_matrix(population_matrix)
        return population.update_genotype_matrix(population_matrix)


# from .operators import MetaOperator

# class ExtendedOperator(Operator):
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
#         param_operators: dict,
#         encoding: ExtendedEncoding,
#         params: ParamScheduler | dict = None,
#         name: str = None,
#     ):
#         """
#         Constructor for the OperatorAdaptative class
#         """

#         super().__init__(params=params, name=base_operator.name)

#         vecmask = np.zeros(encoding.vecsize + encoding.nparams)

#         counter = encoding.vecsize
#         for idx, (_, param_num) in enumerate(encoding.param_sizes):
#             vecmask[counter:counter + param_num] = idx + 1
#             counter = counter+param_num

#         self.base_operator = base_operator
#         self.param_operators = param_operators
#         operator_list = [base_operator] + [param_operators[param_name] for idx, (param_name, _) in enumerate(encoding.param_sizes)]

#         self.main_operator = MetaOperator("Split", operator_list, {"mask": vecmask})
#         self.param_encoding = param_encoding 

#     def evolve(self, population, initializer=None):
#         return self.main_operator.evolve(population, initializer=initializer)

#     def step(self, progress: float):
#         super().step(progress)

#         self.base_operator.step(progress)

#         for op in self.param_operators:
#             op.step(progress)