from __future__ import annotations
from copy import copy
import enum
from enum import Enum
import numpy as np

from .operator_functions.mutation import *
from .operator_functions.crossover import *
from .operator_functions.permutation import *
from .operator_functions.differential_evolution import *
from .operator_functions.swarm import *
from ..operator import Operator
from ..param_scheduler import ParamScheduler
from ..encoding import Encoding, ExtendedEncoding
from ..utils import RAND_GEN


class SwarmOpMethods(Enum):
    PSO = enum.auto()
    FIREFLY = enum.auto()
    GLOWWORM = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in vector_ops_map:
            raise ValueError(f'Vector operator "{str_input}" not defined')

        return vector_ops_map[str_input]


swarm_ops_map = {
    "pso": SwarmOpMethods.PSO,
    "firefly": SwarmOpMethods.FIREFLY,
    "glowworm": SwarmOpMethods.GLOWWORM,
}


class SwarmOperator(Operator):
    """
    Operator class that has mutation and cross methods for real coded vectors

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, method: str, params: ParamScheduler | dict = None, name: str = None, use_params : bool = False, encoding: Encoding = None):
        """
        Constructor for the OperatorPerm class
        """

        if name is None:
            name = method

        super().__init__(params=params, name=name, use_params=use_params, encoding=encoding)

        self.method = SwarmOpMethods.from_str(method)

        assert isinstance(encoding, ExtendedEncoding), "To use swarm operators, an extended encoding must be indicated."

    def evolve(self, population, initializer=None):
        new_population = None
        population_matrix_full = copy(population.genotype_matrix)
        population_encoding = population.encoding

        # Only evolve solution parameters, the rest is managed in a specific way by each operator
        if isinstance(population.encoding, ExtendedEncoding):
            population_matrix = population_encoding.extract_solution(population_matrix_full)
        else:
            population_matrix = population_matrix_full
        encoded_params = None

        fitness_array = copy(population.fitness)
        global_best = population.best
        historical_best = population.historical_best_matrix

        params = copy(self.params)

        # if "Cr" in params and "N" not in params:
        #     params["N"] = np.count_nonzero(RAND_GEN.random(population_matrix.shape[1]) < params["Cr"])

        # if "N" in params:
        #     params["N"] = round(params["N"])
        #     params["N"] = min(params["N"], population_matrix.shape[1])

        # Perform one of the methods (switch-case like structure)
        match self.method:
            case VectorOpMethods.PSO:
                population_params = population_encoding.decode_params(population_matrix_full)
                historical_best_solution = population_encoding.extract_solution(historical_best)
                global_best_solution = population_encoding.extract_solution(global_best[None, :])[0]

                population_matrix, population_params["speed"] = pso_operator(
                    population_matrix, population_params["speed"], historical_best_solution , global_best_solution , params["w"], params["c1"], params["c2"]
                )
                encoded_params = population_encoding.encode_params(population_params) 

            case VectorOpMethods.FIREFLY:
                raise NotImplementedError
                population_matrix = firefly(
                    population_matrix,
                    fitness_array,
                    population.objfunc,
                    params["alpha"],
                    params["beta"],
                    params["delta"],
                    params["gamma"],
                )

            case VectorOpMethods.GLOWWORM:
                raise NotImplementedError
                population_params = population_encoding.decode_params(population_matrix_full)
                population_matrix, luciferin = glowworm(
                    population_matrix,
                    fitness_array,
                    population_params["luciferin"],
                )
                encoded_params = population_encoding.encode_params(population_params) 

        if new_population is None:
            population_matrix = self.encoding.encode(population_matrix)

            # Only evolve solution parameters, the rest is managed in a specific way by each operator
            if isinstance(population.encoding, ExtendedEncoding):
                new_population_matrix = population_matrix_full
                new_population_matrix[:, :population.encoding.vecsize] = population_matrix[:, :population.encoding.vecsize]
                if encoded_params is not None:
                    new_population_matrix[:, population.encoding.vecsize:] = encoded_params
            else:
                new_population_matrix = population_matrix

            new_population = population.update_genotype_matrix(new_population_matrix)

        return new_population
