from __future__ import annotations
import numpy as np
from ..Operator import Operator
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class RealOpMethods(Enum):
    ONE_POINT = enum.auto()
    TWO_POINT = enum.auto()
    MULTIPOINT = enum.auto()
    WEIGHTED_AVG = enum.auto()
    BLXALPHA = enum.auto()
    SBX = enum.auto()
    MULTICROSS = enum.auto()
    CROSSINTERAVG = enum.auto()
    MUTATE1SIGMA = enum.auto()
    MUTATENSIGMAS = enum.auto()
    SAMPLESIGMA = enum.auto()
    PERM = enum.auto()
    GAUSS = enum.auto()
    LAPLACE = enum.auto()
    CAUCHY = enum.auto()
    UNIFORM = enum.auto()
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

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in real_ops_map:
            raise ValueError(f'Real operator "{str_input}" not defined')

        return real_ops_map[str_input]


real_ops_map = {
    "1point": RealOpMethods.ONE_POINT,
    "2point": RealOpMethods.TWO_POINT,
    "multipoint": RealOpMethods.MULTIPOINT,
    "weightedavg": RealOpMethods.WEIGHTED_AVG,
    "blxalpha": RealOpMethods.BLXALPHA,
    "sbx": RealOpMethods.SBX,
    "multicross": RealOpMethods.MULTICROSS,
    "crossinteravg": RealOpMethods.CROSSINTERAVG,
    "mutate1sigma": RealOpMethods.MUTATE1SIGMA,
    "mutatensigmas": RealOpMethods.MUTATENSIGMAS,
    "samplesigma": RealOpMethods.SAMPLESIGMA,
    "perm": RealOpMethods.PERM,
    "gauss": RealOpMethods.GAUSS,
    "laplace": RealOpMethods.LAPLACE,
    "cauchy": RealOpMethods.CAUCHY,
    "uniform": RealOpMethods.UNIFORM,
    "mutrand": RealOpMethods.MUTNOISE,
    "mutnoise": RealOpMethods.MUTNOISE,
    "mutsample": RealOpMethods.MUTSAMPLE,
    "randnoise": RealOpMethods.RANDNOISE,
    "randsample": RealOpMethods.RANDSAMPLE,
    "de/rand/1": RealOpMethods.DE_RAND_1,
    "de/best/1": RealOpMethods.DE_BEST_1,
    "de/rand/2": RealOpMethods.DE_RAND_2,
    "de/best/2": RealOpMethods.DE_BEST_2,
    "de/current-to-rand/1": RealOpMethods.DE_CTRAND_1,
    "de/current-to-best/1": RealOpMethods.DE_CTBEST_1,
    "de/current-to-pbest/1": RealOpMethods.DE_CTPBEST_1,
    "pso": RealOpMethods.PSO,
    "firefly": RealOpMethods.FIREFLY,
    "random": RealOpMethods.RANDOM,
    "randommask": RealOpMethods.RANDOM_MASK,
    "dummy": RealOpMethods.DUMMY,
    "custom": RealOpMethods.CUSTOM,
    "nothing": RealOpMethods.NOTHING,
}


class OperatorReal(Operator):
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

    def __init__(
        self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None
    ):
        """
        Constructor for the OperatorReal class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = RealOpMethods.from_str(method)

        if self.method in [
            RealOpMethods.MUTNOISE,
            RealOpMethods.MUTSAMPLE,
            RealOpMethods.RANDNOISE,
            RealOpMethods.RANDSAMPLE,
        ]:
            self.params["method"] = ProbDist.from_str(self.params["method"])

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

        if self.method == RealOpMethods.ONE_POINT:
            new_indiv.genotype = cross_1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.TWO_POINT:
            new_indiv.genotype = cross_2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.MULTIPOINT:
            new_indiv.genotype = cross_mp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.WEIGHTED_AVG:
            new_indiv.genotype = weighted_average(
                new_indiv.genotype, indiv2.genotype.copy(), params["F"]
            )

        elif self.method == RealOpMethods.BLXALPHA:
            new_indiv.genotype = blxalpha(
                new_indiv.genotype, indiv2.genotype.copy(), params["Cr"]
            )

        elif self.method == RealOpMethods.SBX:
            new_indiv.genotype = sbx(
                new_indiv.genotype, indiv2.genotype.copy(), params["Cr"]
            )

        elif self.method == RealOpMethods.MULTICROSS:
            new_indiv.genotype = multi_cross(
                new_indiv.genotype, others, params["Nindiv"]
            )

        elif self.method == RealOpMethods.CROSSINTERAVG:
            new_indiv.genotype = cross_inter_avg(
                new_indiv.genotype, others, params["N"]
            )

        elif self.method == RealOpMethods.MUTATE1SIGMA:
            new_indiv.genotype = mutate_1_sigma(
                new_indiv.genotype[0], params["epsilon"], params["tau"]
            )

        elif self.method == RealOpMethods.MUTATENSIGMAS:
            new_indiv.genotype = mutate_n_sigmas(
                new_indiv.genotype,
                params["epsilon"],
                params["tau"],
                params["tau_multiple"],
            )

        elif self.method == RealOpMethods.SAMPLESIGMA:
            new_indiv.genotype = sample_1_sigma(
                new_indiv.genotype, params["N"], params["epsilon"], params["tau"]
            )

        elif self.method == RealOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == RealOpMethods.GAUSS:
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.LAPLACE:
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.CAUCHY:
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.UNIFORM:
            new_indiv.genotype = uniform(
                new_indiv.genotype, params["Low"], params["Up"]
            )

        elif self.method == RealOpMethods.MUTNOISE:
            new_indiv.genotype = mutate_rand(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutate_sample(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.RANDNOISE:
            new_indiv.genotype = rand_noise(new_indiv.genotype, params)

        elif self.method == RealOpMethods.RANDSAMPLE:
            new_indiv.genotype = rand_sample(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.DE_RAND_1:
            new_indiv.genotype = DE_rand1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_BEST_1:
            new_indiv.genotype = DE_best1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_RAND_2:
            new_indiv.genotype = DE_rand2(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_BEST_2:
            new_indiv.genotype = DE_best2(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_CTRAND_1:
            new_indiv.genotype = DE_current_to_rand1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_CTBEST_1:
            new_indiv.genotype = DE_current_to_best1(
                new_indiv.genotype, others, params["F"], params["Cr"]
            )

        elif self.method == RealOpMethods.DE_CTPBEST_1:
            new_indiv.genotype = DE_current_to_pbest1(
                new_indiv.genotype, others, params["F"], params["Cr"], params["P"]
            )

        elif self.method == RealOpMethods.PSO:
            new_indiv = pso_operator(
                indiv, others, global_best, params["w"], params["c1"], params["c2"]
            )

        elif self.method == RealOpMethods.FIREFLY:
            new_indiv.genotype = firefly(
                indiv,
                others,
                objfunc,
                params["a"],
                params["b"],
                params["d"],
                params["g"],
            )

        elif self.method == RealOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == RealOpMethods.RANDOM_MASK:
            mask_pos = np.hstack(
                [np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]
            ).astype(bool)
            RAND_GEN.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = initializer.generate_random(
                objfunc
            ).genotype[mask_pos]

        elif self.method == RealOpMethods.DUMMY:
            new_indiv.genotype = dummy_op(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        return new_indiv
