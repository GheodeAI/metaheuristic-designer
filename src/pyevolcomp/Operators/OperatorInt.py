from __future__ import annotations
from ..Operator import Operator
from .vector_operator_functions import *
from copy import copy
from enum import Enum


class IntOpMethods(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    MULTIPOINT = 3
    WEIGHTED_AVG = 4
    BLXALPHA = 4
    MULTICROSS = 5
    XOR = 6
    XOR_CROSS = 7
    CROSSINTERAVG = 8
    PERM = 11
    GAUSS = 12
    LAPLACE = 13
    CAUCHY = 14
    UNIFORM = 15
    POISSON = 16
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

        if str_input not in int_ops_map:
            raise ValueError(f"Integer operator \"{str_input}\" not defined")

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
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict] = None, name: str = None):
        """
        Constructor for the Operator class
        """

        if name is None:
            name = method

        super().__init__(params, name)

        self.method = IntOpMethods.from_str(method)

    def evolve(self, indiv, population, objfunc, global_best, initializer):
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

        if self.method == IntOpMethods.ONE_POINT:
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.TWO_POINT:
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.MULTIPOINT:
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.WEIGHTED_AVG:
            new_indiv.genotype = weightedAverage(new_indiv.genotype, indiv2.genotype.copy(), params["F"])

        elif self.method == IntOpMethods.BLXALPHA:
            new_indiv.genotype = blxalpha(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])

        elif self.method == IntOpMethods.MULTICROSS:
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["Nindiv"])

        elif self.method == IntOpMethods.XOR:
            new_indiv.genotype = xorMask(new_indiv.genotype, params["N"])

        elif self.method == IntOpMethods.XOR_CROSS:
            new_indiv.genotype = xorCross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == IntOpMethods.CROSSINTERAVG:
            new_indiv.genotype = crossInterAvg(new_indiv.genotype, others, params["N"])

        elif self.method == IntOpMethods.PERM:
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == IntOpMethods.GAUSS:
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.LAPLACE:
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.CAUCHY:
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.UNIFORM:
            new_indiv.genotype = uniform(new_indiv.genotype, params["Low"], params["Up"])

        elif self.method == IntOpMethods.POISSON:
            new_indiv.genotype = poisson(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.MUTNOISE:
            new_indiv.genotype = mutateRand(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.MUTSAMPLE:
            new_indiv.genotype = mutateSample(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.RANDNOISE:
            new_indiv.genotype = randNoise(new_indiv.genotype, params)

        elif self.method == IntOpMethods.RANDNOISE:
            new_indiv.genotype = randSample(new_indiv.genotype, others, params)

        elif self.method == IntOpMethods.DE_RAND_1:
            new_indiv.genotype = DERand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_BEST_1:
            new_indiv.genotype = DEBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_RAND_2:
            new_indiv.genotype = DERand2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_BEST_2:
            new_indiv.genotype = DEBest2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_CTRAND_1:
            new_indiv.genotype = DECurrentToRand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_CTBEST_1:
            new_indiv.genotype = DECurrentToBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == IntOpMethods.DE_CTPBEST_1:
            new_indiv.genotype = DECurrentToPBest1(new_indiv.genotype, others, params["F"], params["Cr"], params["P"])

        elif self.method == IntOpMethods.PSO:
            new_indiv = pso_operator(indiv, others, global_best, params["w"], params["c1"], params["c2"])

        elif self.method == IntOpMethods.FIREFLY:
            new_indiv.genotype = firefly(indiv, others, objfunc, params["a"], params["b"], params["d"], params["g"])
        
        elif self.method == IntOpMethods.RANDOM:
            new_indiv = initializer.generate_random(objfunc)

        elif self.method == IntOpMethods.RANDOM_MASK:
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = initializer.generate_random(objfunc).genotype[mask_pos]

        elif self.method == IntOpMethods.DUMMY:
            new_indiv.genotype = dummyOp(new_indiv.genotype, params["F"])

        elif self.method == IntOpMethods.CUSTOM:
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        new_indiv.genotype = np.round(new_indiv.genotype)
        return new_indiv
