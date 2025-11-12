from __future__ import annotations
from copy import copy
import enum
from enum import Enum
from ..Operator import Operator
from .operator_functions.permutation import *
from ..ParamScheduler import ParamScheduler
from ..utils import RAND_GEN


class PermOpMethods(Enum):
    SWAP = enum.auto()
    INSERT = enum.auto()
    SCRAMBLE = enum.auto()
    INVERT = enum.auto()
    ROLL = enum.auto()
    PMX = enum.auto()
    ORDERCROSS = enum.auto()
    RANDOM = enum.auto()
    DUMMY = enum.auto()
    CUSTOM = enum.auto()
    NOTHING = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in perm_ops_map:
            raise ValueError(f'Permutation operator "{str_input}" not defined')

        return perm_ops_map[str_input]


perm_ops_map = {
    "swap": PermOpMethods.SWAP,
    "insert": PermOpMethods.INSERT,
    "scramble": PermOpMethods.SCRAMBLE,
    "perm": PermOpMethods.SCRAMBLE,
    "invert": PermOpMethods.INVERT,
    "roll": PermOpMethods.ROLL,
    "pmx": PermOpMethods.PMX,
    "ordercross": PermOpMethods.ORDERCROSS,
    "random": PermOpMethods.RANDOM,
    "dummy": PermOpMethods.DUMMY,
    "custom": PermOpMethods.CUSTOM,
    "nothing": PermOpMethods.NOTHING,
}


class PermOperator(Operator):
    """
    Operator class that has mutation and cross methods for permutations.

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, method: str, params: ParamScheduler | dict = None, name: str = None):
        """
        Constructor for the OperatorPerm class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = PermOpMethods.from_str(method)

    def evolve(self, population, initializer=None):
        new_population = None
        population_matrix = population.genotype_matrix

        # Only evolve solution parameters, the rest is managed in a specific way by each operator
        if isinstance(population.encoding, ExtendedEncoding):
            population_matrix = population_encoding.extract_solution(population_matrix_full)
        else:
            population_matrix = population_matrix_full
        encoded_params = None

        params = copy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(RAND_GEN.random(population_matrix.shape[1]) < params["Cr"])

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], population_matrix.shape[1])

        # Perform one of the methods (switch-case like structure)
        match self.method:
            case PermOpMethods.SWAP:
                population_matrix = permute_mutation(population_matrix, 2)

            case PermOpMethods.SCRAMBLE:
                population_matrix = permute_mutation(population_matrix, params["N"])

            case PermOpMethods.INSERT:
                population_matrix = roll_mutation(population_matrix, 1)

            case PermOpMethods.ROLL:
                population_matrix = roll_mutation(population_matrix, params["N"])

            case PermOpMethods.INVERT:
                population_matrix = invert_mutation(population_matrix)

            case PermOpMethods.PMX:
                population_matrix = pmx(population_matrix)

            case PermOpMethods.ORDERCROSS:
                population_matrix = order_cross(population_matrix)

            case PermOpMethods.RANDOM:
                new_population = initializer.generate_population(population.objfunc)

            case PermOpMethods.DUMMY:
                population_matrix = np.tile(
                    np.arange(population_matrix.shape[1]),
                    (population_matrix.shape[0], 1),
                )

            case PermOpMethods.CUSTOM:
                fn = params["function"]
                population_matrix = fn(population_matrix, population.objfunc, params)

            case PermOpMethods.NOTHING:
                new_population = copy(population)

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

            new_population = population.update_genotype_matrix(population_matrix)

        return new_population
