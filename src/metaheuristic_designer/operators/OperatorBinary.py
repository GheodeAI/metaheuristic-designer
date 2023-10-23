from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from copy import copy
import enum
from enum import Enum


class BinOpMethods(Enum):
    ONE_POINT = enum.auto()
    TWO_POINT = enum.auto()
    MULTIPOINT = enum.auto()
    MULTICROSS = enum.auto()
    XOR = enum.auto()
    XOR_CROSS = enum.auto()
    PERM = enum.auto()
    MUTSAMPLE = enum.auto()
    RANDSAMPLE = enum.auto()
    RANDOM = enum.auto()
    RANDOM_MASK = enum.auto()
    DUMMY = enum.auto()
    CUSTOM = enum.auto()
    NOTHING = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in bin_ops_map:
            raise ValueError(f'Binary operator "{str_input}" not defined')

        return bin_ops_map[str_input]


bin_ops_map = {
    "1point": BinOpMethods.ONE_POINT,
    "2point": BinOpMethods.TWO_POINT,
    "multipoint": BinOpMethods.MULTIPOINT,
    "multicross": BinOpMethods.MULTICROSS,
    "xor": BinOpMethods.XOR,
    "flip": BinOpMethods.XOR,
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
    Operator class that has binary mutation and cross methods

    Parameters
    ----------
    method: str
        Type of operator that will be applied
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

        self.method = BinOpMethods.from_str(method)

        if self.method in [BinOpMethods.MUTSAMPLE, BinOpMethods.RANDSAMPLE]:
            self.params["method"] = ProbDist.BERNOULLI

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

        if self.method == BinOpMethods.ONE_POINT:
            new_indiv.genotype = cross_1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.TWO_POINT:
            new_indiv.genotype = cross_2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MULTIPOINT:
            new_indiv.genotype = cross_mp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MULTICROSS:
            new_indiv.genotype = multi_cross(
                new_indiv.genotype, others, params["Nindiv"]
            )

        elif self.method == BinOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == BinOpMethods.XOR:
            new_indiv.genotype = xor_mask(new_indiv.genotype, params["N"], mode="bin")

        elif self.method == BinOpMethods.XOR_CROSS:
            new_indiv.genotype = xor_cross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == BinOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutate_sample(new_indiv.genotype, population, params)

        elif self.method == BinOpMethods.RANDSAMPLE:
            new_indiv.genotype = rand_sample(new_indiv.genotype, population, params)

        elif self.method == BinOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == BinOpMethods.RANDOM_MASK:
            mask_pos = np.hstack(
                [np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]
            ).astype(bool)
            RAND_GEN.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = initializer.generate_random(
                objfunc
            ).genotype[mask_pos]

        elif self.method == BinOpMethods.DUMMY:
            new_indiv.genotype = dummy_op(new_indiv.genotype, params["F"])

        elif self.method == BinOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        new_indiv.genotype = (new_indiv.genotype != 0).astype(np.int32)
        return new_indiv
