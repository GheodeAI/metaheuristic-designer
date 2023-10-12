from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from copy import copy
import enum
from enum import Enum
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


class OperatorPerm(Operator):
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

    def __init__(
        self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None
    ):
        """
        Constructor for the Operator class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = PermOpMethods.from_str(method)

    def evolve(self, indiv, population, objfunc, global_best, initializer):
        new_indiv = copy(indiv)
        others = [i for i in population if i != indiv]
        if len(others) == 0:
            indiv2 = indiv
            others = [indiv]
        elif len(others) == 1:
            indiv2 = indiv
        else:
            indiv2 = random.choice(others)

        if global_best is None:
            global_best = indiv

        params = copy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(
                RAND_GEN.random(indiv.genotype.size) < params["Cr"]
            )

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], new_indiv.genotype.size)

        if self.method == PermOpMethods.SWAP:
            new_indiv.genotype = permutation(new_indiv.genotype, 2)

        elif self.method == PermOpMethods.SCRAMBLE:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == PermOpMethods.INSERT:
            new_indiv.genotype = roll(new_indiv.genotype, 1)

        elif self.method == PermOpMethods.ROLL:
            new_indiv.genotype = roll(new_indiv.genotype, params["N"])

        elif self.method == PermOpMethods.INVERT:
            new_indiv.genotype = invert_mutation(new_indiv.genotype)

        elif self.method == PermOpMethods.PMX:
            new_indiv.genotype = pmx(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == PermOpMethods.ORDERCROSS:
            new_indiv.genotype = order_cross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == PermOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == PermOpMethods.DUMMY:
            new_indiv.genotype = np.arange(indiv.genotype.size)

        elif self.method == PermOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        return new_indiv
