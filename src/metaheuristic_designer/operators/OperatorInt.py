from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class IntOpMethods(Enum):
    ONE_POINT = enum.auto()
    TWO_POINT = enum.auto()
    MULTIPOINT = enum.auto()
    WEIGHTED_AVG = enum.auto()
    BLXALPHA = enum.auto()
    MULTICROSS = enum.auto()
    XOR = enum.auto()
    XOR_CROSS = enum.auto()
    CROSSINTERAVG = enum.auto()
    PERM = enum.auto()
    GAUSS = enum.auto()
    LAPLACE = enum.auto()
    CAUCHY = enum.auto()
    UNIFORM = enum.auto()
    POISSON = enum.auto()
    MUTNOISE = enum.auto()
    MUTSAMPLE = enum.auto()
    RANDNOISE = enum.auto()
    RANDSAMPLE = enum.auto()
    DE_RAND_1 = enum.auto()
    DE_BEST_1 = enum.auto()
    DE_RAND_2 = enum.auto()
    DE_BEST_2 = enum.auto()
    DE_CTRAND_1 = enum.auto()
    DE_CTBEST_1 = enum.auto()
    DE_CTPBEST_1 = enum.auto()
    PSO = enum.auto()
    FIREFLY = enum.auto()
    RANDOM = enum.auto()
    RANDOM_MASK = enum.auto()
    DUMMY = enum.auto()
    CUSTOM = enum.auto()
    NOTHING = enum.auto()
    RANDRESET = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in int_ops_map:
            raise ValueError(f'Integer operator "{str_input}" not defined')

        return int_ops_map[str_input]


int_ops_map = {
    "1point": IntOpMethods.ONE_POINT,
    "2point": IntOpMethods.TWO_POINT,
    "multipoint": IntOpMethods.MULTIPOINT,
    "weightedavg": IntOpMethods.WEIGHTED_AVG,
    "blxalpha": IntOpMethods.BLXALPHA,
    "multicross": IntOpMethods.MULTICROSS,
    "xor": IntOpMethods.XOR,
    "xorcross": IntOpMethods.XOR_CROSS,
    "crossinteravg": IntOpMethods.CROSSINTERAVG,
    "perm": IntOpMethods.PERM,
    "gauss": IntOpMethods.GAUSS,
    "laplace": IntOpMethods.LAPLACE,
    "cauchy": IntOpMethods.CAUCHY,
    "uniform": IntOpMethods.UNIFORM,
    "poisson": IntOpMethods.POISSON,
    "mutrand": IntOpMethods.MUTNOISE,
    "mutnoise": IntOpMethods.MUTNOISE,
    "mutsample": IntOpMethods.MUTSAMPLE,
    "randnoise": IntOpMethods.RANDNOISE,
    "randsample": IntOpMethods.RANDSAMPLE,
    "randreset": IntOpMethods.RANDRESET,
    "randomreset": IntOpMethods.RANDRESET,
    "de/rand/1": IntOpMethods.DE_RAND_1,
    "de/best/1": IntOpMethods.DE_BEST_1,
    "de/rand/2": IntOpMethods.DE_RAND_2,
    "de/best/2": IntOpMethods.DE_BEST_2,
    "de/current-to-rand/1": IntOpMethods.DE_CTRAND_1,
    "de/current-to-best/1": IntOpMethods.DE_CTBEST_1,
    "de/current-to-pbest/1": IntOpMethods.DE_CTPBEST_1,
    "pso": IntOpMethods.PSO,
    "firefly": IntOpMethods.FIREFLY,
    "random": IntOpMethods.RANDOM,
    "randommask": IntOpMethods.RANDOM_MASK,
    "dummy": IntOpMethods.DUMMY,
    "custom": IntOpMethods.CUSTOM,
    "nothing": IntOpMethods.NOTHING,
}


class OperatorInt(Operator):
    """
    Operator class that has discrete mutation and cross methods.

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

        self.method = IntOpMethods.from_str(method)

        if self.method in [
            IntOpMethods.MUTNOISE,
            IntOpMethods.MUTSAMPLE,
            IntOpMethods.RANDNOISE,
            IntOpMethods.RANDSAMPLE,
        ]:
            self.params["method"] = ProbDist.from_str(self.params["method"])

        elif self.method == IntOpMethods.RANDRESET:
            self.params["method"] = ProbDist.UNIFORM

            if "Low" not in self.params:
                self.params["Low"] = 0

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

        if self.method == IntOpMethods.ONE_POINT:
            new_indiv.genotype = cross_1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.TWO_POINT:
            new_indiv.genotype = cross_2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.MULTIPOINT:
            new_indiv.genotype = cross_mp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.WEIGHTED_AVG:
            new_indiv.genotype = weighted_average(
                new_indiv.genotype, indiv2.genotype.copy(), params["F"]
            )

        elif self.method == IntOpMethods.BLXALPHA:
            new_indiv.genotype = blxalpha(
                new_indiv.genotype, indiv2.genotype.copy(), params["Cr"]
            )

        elif self.method == IntOpMethods.MULTICROSS:
            new_indiv.genotype = multi_cross(
                new_indiv.genotype, others, params["Nindiv"]
            )

        elif self.method == IntOpMethods.XOR:
            new_indiv.genotype = xor_mask(new_indiv.genotype, params["N"])

        elif self.method == IntOpMethods.XOR_CROSS:
            new_indiv.genotype = xor_cross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.CROSSINTERAVG:
            new_indiv.genotype = cross_inter_avg(
                new_indiv.genotype, others, params["N"]
            )

        elif self.method == IntOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == IntOpMethods.GAUSS:
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.LAPLACE:
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.CAUCHY:
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.UNIFORM:
            new_indiv.genotype = uniform(
                new_indiv.genotype, params["Low"], params["Up"]
            )

        elif self.method == IntOpMethods.POISSON:
            new_indiv.genotype = poisson(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.MUTNOISE:
            new_indiv.genotype = mutate_rand(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutate_sample(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.RANDNOISE:
            new_indiv.genotype = rand_noise(new_indiv.genotype, params)

        elif self.method == IntOpMethods.RANDSAMPLE:
            new_indiv.genotype = rand_sample(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.RANDRESET:
            new_indiv.genotype = mutate_sample(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.DE_RAND_1:
            new_indiv.genotype = DE_rand1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_BEST_1:
            new_indiv.genotype = DE_best1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_RAND_2:
            new_indiv.genotype = DE_rand2(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_BEST_2:
            new_indiv.genotype = DE_best2(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_CTRAND_1:
            new_indiv.genotype = DE_current_to_rand1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_CTBEST_1:
            new_indiv.genotype = DE_current_to_best1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == IntOpMethods.DE_CTPBEST_1:
            new_indiv.genotype = DE_current_to_pbest1(
                new_indiv.genotype, others, params["F"], params["Cr"], params["P"]
            )

        elif self.method == IntOpMethods.PSO:
            new_indiv = pso_operator(
                indiv, others, global_best, params["w"], params["c1"], params["c2"]
            )

        elif self.method == IntOpMethods.FIREFLY:
            new_indiv.genotype = firefly(
                indiv,
                others,
                objfunc,
                params["a"],
                params["b"],
                params["d"],
                params["g"],
            )

        elif self.method == IntOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == IntOpMethods.RANDOM_MASK:
            mask_pos = np.hstack(
                [np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]
            ).astype(bool)
            RAND_GEN.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = initializer.generate_random(
                objfunc
            ).genotype[mask_pos]

        elif self.method == IntOpMethods.DUMMY:
            new_indiv.genotype = dummy_op(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        new_indiv.genotype = np.round(new_indiv.genotype)
        return new_indiv
