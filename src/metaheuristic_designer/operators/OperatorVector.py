from __future__ import annotations
import numpy as np
from ..Operator import Operator
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from ..Encoding import Encoding
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class VectorOpMethods(Enum):
    ONE_POINT = enum.auto()
    TWO_POINT = enum.auto()
    MULTIPOINT = enum.auto()
    WEIGHTED_AVG = enum.auto()
    BLXALPHA = enum.auto()
    SBX = enum.auto()
    MULTICROSS = enum.auto()
    XOR_CROSS = enum.auto()
    CROSSINTERAVG = enum.auto()

    MUTATE1SIGMA = enum.auto()
    MUTATENSIGMAS = enum.auto()
    SAMPLESIGMA = enum.auto()

    PERM = enum.auto()
    XOR = enum.auto()
    GAUSS = enum.auto()
    LAPLACE = enum.auto()
    CAUCHY = enum.auto()
    UNIFORM = enum.auto()
    POISSON = enum.auto()

    MUTNOISE = enum.auto()
    MUTSAMPLE = enum.auto()
    RANDNOISE = enum.auto()
    RANDSAMPLE = enum.auto()
    RANDRESET = enum.auto()
    GENERATE = enum.auto()

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

    CUSTOM = enum.auto()
    DUMMY = enum.auto()
    NOTHING = enum.auto()

    @staticmethod
    def from_str(str_input):
        str_input = str_input.lower()

        if str_input not in real_ops_map:
            raise ValueError(f'Vector operator "{str_input}" not defined')

        return real_ops_map[str_input]


vector_ops_map = {
    "1point": VectorOpMethods.ONE_POINT,
    "2point": VectorOpMethods.TWO_POINT,
    "multipoint": VectorOpMethods.MULTIPOINT,
    "weightedavg": VectorOpMethods.WEIGHTED_AVG,
    "blxalpha": VectorOpMethods.BLXALPHA,
    "sbx": VectorOpMethods.SBX,
    "multicross": VectorOpMethods.MULTICROSS,
    "xor": VectorOpMethods.XOR,
    "flip": VectorOpMethods.XOR,
    "xorcross": VectorOpMethods.XOR_CROSS,
    "flipcross": VectorOpMethods.XOR_CROSS,
    "crossinteravg": VectorOpMethods.CROSSINTERAVG,
    "mutate1sigma": VectorOpMethods.MUTATE1SIGMA,
    "mutatensigmas": VectorOpMethods.MUTATENSIGMAS,
    "samplesigma": VectorOpMethods.SAMPLESIGMA,
    "perm": VectorOpMethods.PERM,
    "gauss": VectorOpMethods.GAUSS,
    "laplace": VectorOpMethods.LAPLACE,
    "cauchy": VectorOpMethods.CAUCHY,
    "uniform": VectorOpMethods.UNIFORM,
    "poisson": VectorOpMethods.POISSON,
    "mutrand": VectorOpMethods.MUTNOISE,
    "mutnoise": VectorOpMethods.MUTNOISE,
    "mutsample": VectorOpMethods.MUTSAMPLE,
    "randnoise": VectorOpMethods.RANDNOISE,
    "randsample": VectorOpMethods.RANDSAMPLE,
    "randomreset": VectorOpMethods.RANDRESET,
    "generate": VectorOpMethods.GENERATE,
    "de/rand/1": VectorOpMethods.DE_RAND_1,
    "de/best/1": VectorOpMethods.DE_BEST_1,
    "de/rand/2": VectorOpMethods.DE_RAND_2,
    "de/best/2": VectorOpMethods.DE_BEST_2,
    "de/current-to-rand/1": VectorOpMethods.DE_CTRAND_1,
    "de/current-to-best/1": VectorOpMethods.DE_CTBEST_1,
    "de/current-to-pbest/1": VectorOpMethods.DE_CTPBEST_1,
    "pso": VectorOpMethods.PSO,
    "firefly": VectorOpMethods.FIREFLY,
    "random": VectorOpMethods.RANDOM,
    "randommask": VectorOpMethods.RANDOM_MASK,
    "dummy": VectorOpMethods.DUMMY,
    "custom": VectorOpMethods.CUSTOM,
    "nothing": VectorOpMethods.NOTHING,
}


class OperatorVector(Operator):
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

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None, encoding: Encoding = None):
        """
        Constructor for the OperatorPerm class
        """

        if name is None:
            name = method

        super().__init__(params=params, name=name, encoding=encoding)

        self.method = VectorOpMethods.from_str(method)

        if self.method in [
            VectorOpMethods.MUTNOISE,
            VectorOpMethods.MUTSAMPLE,
            VectorOpMethods.RANDNOISE,
            VectorOpMethods.RANDSAMPLE,
        ]:
            self.params["distrib"] = ProbDist.from_str(self.params["distrib"])
        elif self.method == VectorOpMethods.RANDRESET:
            self.params["distrib"] = ProbDist.UNIFORM

            if "Low" not in self.params:
                self.params["Low"] = 0

    def evolve(self, population, objfunc, global_best, initializer):
        new_population = [self.evolve_single(indiv, population, objfunc, global_best, intializer) for indiv in population]

        return new_population

    def evolve_single(self, indiv, population, objfunc, global_best, initializer):
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
            params["N"] = np.count_nonzero(RAND_GEN.random(indiv.genotype.size) < params["Cr"])

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], new_indiv.genotype.size)

        if self.method == VectorOpMethods.ONE_POINT:
            new_indiv.genotype = cross_1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == VectorOpMethods.TWO_POINT:
            new_indiv.genotype = cross_2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == VectorOpMethods.MULTIPOINT:
            new_indiv.genotype = cross_mp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == VectorOpMethods.WEIGHTED_AVG:
            new_indiv.genotype = weighted_average(new_indiv.genotype, indiv2.genotype.copy(), params["F"])

        elif self.method == VectorOpMethods.BLXALPHA:
            new_indiv.genotype = blxalpha(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])

        elif self.method == VectorOpMethods.SBX:
            new_indiv.genotype = sbx(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])

        elif self.method == VectorOpMethods.MULTICROSS:
            new_indiv.genotype = multi_cross(new_indiv.genotype, others, params["Nindiv"])

        elif self.method == VectorOpMethods.XOR:
            new_indiv.genotype = xor_mask(new_indiv.genotype, params["N"])

        elif self.method == VectorOpMethods.XOR_CROSS:
            new_indiv.genotype = xor_cross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == VectorOpMethods.CROSSINTERAVG:
            new_indiv.genotype = cross_inter_avg(new_indiv.genotype, others, params["N"])

        elif self.method == VectorOpMethods.MUTATE1SIGMA:
            new_indiv.genotype = mutate_1_sigma(new_indiv.genotype, params["epsilon"], params["tau"])

        elif self.method == VectorOpMethods.MUTATENSIGMAS:
            new_indiv.genotype = mutate_n_sigmas(
                new_indiv.genotype,
                params["epsilon"],
                params["tau"],
                params["tau_multiple"],
            )

        elif self.method == VectorOpMethods.SAMPLESIGMA:
            new_indiv.genotype = sample_1_sigma(new_indiv.genotype, params["N"], params["epsilon"], params["tau"])

        elif self.method == VectorOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == VectorOpMethods.GAUSS:
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == VectorOpMethods.LAPLACE:
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])

        elif self.method == VectorOpMethods.CAUCHY:
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == VectorOpMethods.UNIFORM:
            new_indiv.genotype = uniform(new_indiv.genotype, params["Low"], params["Up"])

        elif self.method == VectorOpMethods.MUTNOISE:
            new_indiv.genotype = mutate_noise(new_indiv.genotype, params)

        elif self.method == VectorOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutate_sample(new_indiv.genotype, others, params)

        elif self.method == VectorOpMethods.RANDNOISE:
            new_indiv.genotype = rand_noise(new_indiv.genotype, params)

        elif self.method == VectorOpMethods.RANDSAMPLE:
            new_indiv.genotype = rand_sample(new_indiv.genotype, others, params)

        elif self.method == VectorOpMethods.GENERATE:
            new_indiv.genotype = generate_statistic(new_indiv.genotype, others, params)

        elif self.method == VectorOpMethods.DE_RAND_1:
            new_indiv.genotype = DE_rand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_BEST_1:
            new_indiv.genotype = DE_best1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_RAND_2:
            new_indiv.genotype = DE_rand2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_BEST_2:
            new_indiv.genotype = DE_best2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTRAND_1:
            new_indiv.genotype = DE_current_to_rand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTBEST_1:
            new_indiv.genotype = DE_current_to_best1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTPBEST_1:
            new_indiv.genotype = DE_current_to_pbest1(new_indiv.genotype, others, params["F"], params["Cr"], params["P"])

        elif self.method == VectorOpMethods.PSO:
            new_indiv = pso_operator(indiv, others, global_best, params["w"], params["c1"], params["c2"])

        elif self.method == VectorOpMethods.FIREFLY:
            new_indiv.genotype = firefly(
                indiv,
                others,
                objfunc,
                params["a"],
                params["b"],
                params["d"],
                params["g"],
            )

        elif self.method == VectorOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == VectorOpMethods.RANDOM_MASK:
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            RAND_GEN.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = initializer.generate_random(objfunc).genotype[mask_pos]

        elif self.method == VectorOpMethods.DUMMY:
            new_indiv.genotype = dummy_op(new_indiv.genotype, params["F"])

        elif self.method == VectorOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        elif self.method == VectorOpMethods.NOTHING:
            new_indiv = indiv
        
        new_inidiv.genotype = self.encoding.encode(new_inidiv.genotype)

        return new_indiv
