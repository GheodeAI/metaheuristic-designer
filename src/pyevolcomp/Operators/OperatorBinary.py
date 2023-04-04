from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from copy import copy
from enum import Enum


class BinOpMethods(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    MULTIPOINT = 3
    MULTICROSS = 4
    XOR = 5
    XOR_CROSS = 6
    PERM = 7
    MUTSAMPLE = 8
    RANDSAMPLE = 9
    RANDOM = 10
    RANDOM_MASK = 11
    DUMMY = 12
    CUSTOM = 13
    NOTHING = 14

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in bin_ops_map:
            raise ValueError(f"Binary operator \"{str_input}\" not defined")

        return bin_ops_map[str_input]


bin_ops_map = {
    "1point": BinOpMethods.ONE_POINT,
    "2point": BinOpMethods.TWO_POINT,
    "multipoint": BinOpMethods.MULTIPOINT,
    "multicross": BinOpMethods.MULTICROSS,
    "xor": BinOpMethods.XOR,
    "fliprandom": BinOpMethods.XOR,
    "xorcross": BinOpMethods.XOR_CROSS,
    "flipcross": BinOpMethods.XOR_CROSS,
    "perm": BinOpMethods.PERM,
    "mutrand": BinOpMethods.MUTSAMPLE,
    "mutnoise": BinOpMethods.MUTSAMPLE,
    "mutsample": BinOpMethods.MUTSAMPLE,
    "randnoise": BinOpMethods.RANDSAMPLE,
    "randsample": BinOpMethods.RANDSAMPLE,
    "random": BinOpMethods.RANDOM,
    "randommask": BinOpMethods.RANDOM_MASK,
    "dummy": BinOpMethods.DUMMY,
    "custom": BinOpMethods.CUSTOM,
    "nothing": BinOpMethods.NOTHING,
}


class OperatorBinary(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the Operator class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = BinOpMethods.from_str(method)

    def evolve(self, indiv, population, objfunc, global_best):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        new_indiv = copy(indiv)
        others = [i for i in population if i != indiv]
        if len(others) > 1:
            indiv2 = random.choice(others)
        else:
            indiv2 = indiv

        if global_best is None:
            global_best = indiv

        params = copy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(np.random.random(indiv.genotype.size) < params["Cr"])

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], new_indiv.genotype.size)

        if self.method == BinOpMethods.ONE_POINT:
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.TWO_POINT:
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MULTIPOINT:
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MULTICROSS:
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["Nindiv"])

        elif self.method == BinOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == BinOpMethods.XOR:
            new_indiv.genotype = xorMask(new_indiv.genotype, params["N"], mode="bin")

        elif self.method == BinOpMethods.XOR_CROSS:
            new_indiv.genotype = xorCross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MUTSAMPLE:
            params["method"] = "bernouli"
            new_indiv.genotype = mutateSample(new_indiv.genotype, population, params)

        elif self.method == BinOpMethods.RANDSAMPLE:
            params["method"] = "bernouli"
            new_indiv.genotype = randSample(new_indiv.genotype, population, params)

        elif self.method == BinOpMethods.RANDOM:
            new_indiv.genotype = objfunc.random_solution()

        elif self.method == BinOpMethods.RANDOM_MASK:
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = objfunc.random_solution()[mask_pos]

        elif self.method == BinOpMethods.DUMMY:
            new_indiv.genotype = dummyOp(new_indiv.genotype, params["F"])

        elif self.method == BinOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        new_indiv.genotype = (new_indiv.genotype != 0).astype(np.int32)
        return new_indiv
