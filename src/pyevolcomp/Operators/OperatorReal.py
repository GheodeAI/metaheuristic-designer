from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from ..ParamScheduler import ParamScheduler
from copy import copy
from enum import Enum


class RealOpMethods(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    MULTIPOINT = 3
    WEIGHTED_AVG = 4
    BLXALPHA = 5
    SBX = 6
    MULTICROSS = 7
    CROSSINTERAVG = 8
    MUTATE1SIGMA = 9
    MUTATENSIGMAS = 10
    SAMPLESIGMA = 11
    PERM = 12
    GAUSS = 13
    LAPLACE = 14
    CAUCHY = 15
    UNIFORM = 16
    MUTNOISE = 17
    MUTSAMPLE = 18
    RANDNOISE = 19
    RANDSAMPLE = 20
    DE_RAND_1 = 21
    DE_BEST_1 = 22
    DE_RAND_2 = 23
    DE_BEST_2 = 24
    DE_CTRAND_1 = 25
    DE_CTBEST_1 = 26
    DE_CTPBEST_1 = 27
    PSO = 28
    FIREFLY = 29
    RANDOM = 30
    RANDOM_MASK = 31
    DUMMY = 32
    CUSTOM = 33
    NOTHING = 34

    @staticmethod
    def from_str(str_input):

        str_input = str_input.lower()

        if str_input not in real_ops_map:
            raise ValueError(f"Real operator \"{str_input}\" not defined")

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
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the OperatorReal class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = RealOpMethods.from_str(method)

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

        if self.method == RealOpMethods.ONE_POINT:
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.TWO_POINT:
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.MULTIPOINT:
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == RealOpMethods.WEIGHTED_AVG:
            new_indiv.genotype = weightedAverage(new_indiv.genotype, indiv2.genotype.copy(), params["F"])

        elif self.method == RealOpMethods.BLXALPHA:
            new_indiv.genotype = blxalpha(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])

        elif self.method == RealOpMethods.SBX:
            new_indiv.genotype = sbx(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])

        elif self.method == RealOpMethods.MULTICROSS:
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["Nindiv"])

        elif self.method == RealOpMethods.CROSSINTERAVG:
            new_indiv.genotype = crossInterAvg(new_indiv.genotype, others, params["N"])

        elif self.method == RealOpMethods.MUTATE1SIGMA:
            new_indiv.genotype = mutate_1_sigma(new_indiv.genotype[0], params["epsilon"], params["tau"])

        elif self.method == RealOpMethods.MUTATENSIGMAS:
            new_indiv.genotype = mutate_n_sigmas(new_indiv.genotype, params["epsilon"], params["tau"], params["tau_multiple"])

        elif self.method == RealOpMethods.SAMPLESIGMA:
            new_indiv.genotype = sample_1_sigma(new_indiv.genotype, params["N"], params["epsilon"], params["tau"])

        elif self.method == RealOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == RealOpMethods.GAUSS:
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.LAPLACE:
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.CAUCHY:
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.UNIFORM:
            new_indiv.genotype = uniform(new_indiv.genotype, params["Low"], params["Up"])

        elif self.method == RealOpMethods.MUTNOISE:
            new_indiv.genotype = mutateRand(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutateSample(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.RANDNOISE:
            new_indiv.genotype = randNoise(new_indiv.genotype, params)

        elif self.method == RealOpMethods.RANDNOISE:
            new_indiv.genotype = randSample(new_indiv.genotype, others, params)

        elif self.method == RealOpMethods.DE_RAND_1:
            new_indiv.genotype = DERand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_BEST_1:
            new_indiv.genotype = DEBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_RAND_2:
            new_indiv.genotype = DERand2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_BEST_2:
            new_indiv.genotype = DEBest2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_CTRAND_1:
            new_indiv.genotype = DECurrentToRand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_CTBEST_1:
            new_indiv.genotype = DECurrentToBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == RealOpMethods.DE_CTPBEST_1:
            new_indiv.genotype = DECurrentToPBest1(new_indiv.genotype, others, params["F"], params["Cr"], params["P"])

        elif self.method == RealOpMethods.PSO:
            new_indiv = pso_operator(indiv, others, global_best, params["w"], params["c1"], params["c2"])

        elif self.method == RealOpMethods.FIREFLY:
            new_indiv.genotype = firefly(indiv, others, objfunc, params["a"], params["b"], params["d"], params["g"])

        elif self.method == RealOpMethods.RANDOM:
            new_indiv.genotype = objfunc.random_solution()

        elif self.method == RealOpMethods.RANDOM_MASK:
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = objfunc.decoder.encode(objfunc.random_solution())[mask_pos]

        elif self.method == RealOpMethods.DUMMY:
            new_indiv.genotype = dummyOp(new_indiv.genotype, params["F"])

        elif self.method == RealOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        return new_indiv
