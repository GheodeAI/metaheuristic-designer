from __future__ import annotations
import numpy as np
from ..Operator import Operator

from .crossover import *
from .mutation import *
from .permutation_mutation import *
from .differential_evolution import *
from .swarm import *

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

        if str_input not in vector_ops_map:
            raise ValueError(f'Vector operator "{str_input}" not defined')

        return vector_ops_map[str_input]


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

    def evolve_single(self, indiv, population, objfunc, global_best, initializer):
        raise Exception("LMAO what?")

    def evolve(self, population, objfunc, initializer=None, global_best=None):
        new_population = None
        population_matrix = np.array([indiv.genotype for indiv in population])
        fitness_array = np.array([indiv.fitness for indiv in population])
        speed = None

        params = copy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(RAND_GEN.random(population_matrix.shape[1]) < params["Cr"])

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], population_matrix.shape[1])

        # Perform one of the methods (switch-case like structure)

        ## Cross operations
        if self.method == VectorOpMethods.ONE_POINT:
            population_matrix = cross_1p(population_matrix)

        elif self.method == VectorOpMethods.TWO_POINT:
            population_matrix = cross_2p(population_matrix)

        elif self.method == VectorOpMethods.MULTIPOINT:
            population_matrix = cross_mp(population_matrix)

        elif self.method == VectorOpMethods.WEIGHTED_AVG:
            population_matrix = weighted_average_cross(population_matrix, params["F"])

        elif self.method == VectorOpMethods.BLXALPHA:
            population_matrix = blxalpha(population_matrix, params["Cr"])

        elif self.method == VectorOpMethods.SBX:
            population_matrix = sbx(population_matrix, params["Cr"])

        elif self.method == VectorOpMethods.XOR_CROSS:
            population_matrix = xor_cross(population_matrix)

        elif self.method == VectorOpMethods.MULTICROSS:
            population_matrix = multi_cross(population_matrix, params["Nindiv"])

        elif self.method == VectorOpMethods.CROSSINTERAVG:
            population_matrix = cross_inter_avg(population_matrix, params["N"])

        ## Adaptative mutations
        elif self.method == VectorOpMethods.MUTATE1SIGMA:
            population_matrix = mutate_1_sigma(population_matrix, params["epsilon"], params["tau"])

        elif self.method == VectorOpMethods.MUTATENSIGMAS:
            population_matrix = mutate_n_sigmas(
                population_matrix,
                params["epsilon"],
                params["tau"],
                params["tau_multiple"],
            )

        elif self.method == VectorOpMethods.SAMPLESIGMA:
            population_matrix = sample_1_sigma(population_matrix, params["N"], params["epsilon"], params["tau"])

        ## Mutation operators
        elif self.method == VectorOpMethods.XOR:
            population_matrix = xor_mask(population_matrix, params["N"], params.get("BinRep", "byte"))

        elif self.method == VectorOpMethods.PERM:
            population_matrix = permute_mutation(population_matrix, params["N"])

        elif self.method == VectorOpMethods.GAUSS:
            population_matrix = gaussian_mutation(population_matrix, params["F"])

        elif self.method == VectorOpMethods.LAPLACE:
            population_matrix = laplace_mutation(population_matrix, params["F"])

        elif self.method == VectorOpMethods.CAUCHY:
            population_matrix = cauchy_mutation(population_matrix, params["F"])

        elif self.method == VectorOpMethods.UNIFORM:
            population_matrix = uniform_mutation(population_matrix, params["F"])

        elif self.method == VectorOpMethods.UNIFORM:
            population_matrix = poisson_mutation(population_matrix, params["F"], params["mu"])

        elif self.method == VectorOpMethods.MUTNOISE:
            population_matrix = mutate_noise(population_matrix, **params)

        elif self.method == VectorOpMethods.MUTSAMPLE:
            population_matrix = mutate_sample(population_matrix, **params)

        elif self.method == VectorOpMethods.RANDNOISE:
            population_matrix = rand_noise(population_matrix, **params)

        elif self.method == VectorOpMethods.RANDSAMPLE:
            population_matrix = rand_sample(population_matrix, **params)

        elif self.method == VectorOpMethods.GENERATE:
            population_row = generate_statistic(population_matrix, **params)
            population_matrix = np.tile(population_row, population_matrix.shape[0]).reshape(population_matrix.shape)

        ## Differential evolution operators
        elif self.method == VectorOpMethods.DE_RAND_1:
            population_matrix = DE_rand1(population_matrix, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_BEST_1:
            population_matrix = DE_best1(population_matrix, fitness_array,  params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_RAND_2:
            population_matrix = DE_rand2(population_matrix, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_BEST_2:
            population_matrix = DE_best2(population_matrix, fitness_array, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTRAND_1:
            population_matrix = DE_current_to_rand1(population_matrix, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTBEST_1:
            population_matrix = DE_current_to_best1(population_matrix, fitness_array, params["F"], params["Cr"])

        elif self.method == VectorOpMethods.DE_CTPBEST_1:
            population_matrix = DE_current_to_pbest1(population_matrix, fitness_array, params["F"], params["Cr"], params["P"])

        ## Swarm based algorithms
        elif self.method == VectorOpMethods.PSO:
            population_speed = np.array([indiv.speed for indiv in population])
            historical_best = np.array([indiv.best for indiv in population])
            population_matrix, speed = pso_operator(population_matrix, population_speed, historical_best, global_best.genotype, params["w"], params["c1"], params["c2"])

        elif self.method == VectorOpMethods.FIREFLY:
            population_matrix = firefly(
                indiv,
                others,
                objfunc,
                params["a"],
                params["b"],
                params["d"],
                params["g"],
            )

        ## Other operators
        elif self.method == VectorOpMethods.RANDOM:
            new_population = initializer.generate_population(objfunc, len(population))

        elif self.method == VectorOpMethods.RANDOM_MASK:
            mask_pos = np.tile(np.arange(population_matrix.shape[1]) < params["N"], population_matrix.shape[0]).reshape(population_matrix.shape)
            mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

            random_population = initializer.generate_population(objfunc, len(population))
            random_population_matrix = np.array([indiv.genotype for indiv in random_population])

            population_matrix[mask_pos] = random_population_matrix[mask_pos]

        elif self.method == VectorOpMethods.DUMMY:
            population_matrix = dummy_op(population_matrix, params["F"])

        elif self.method == VectorOpMethods.CUSTOM:
            fn = params["function"]
            population_matrix = fn(population, objfunc, params)

        elif self.method == VectorOpMethods.NOTHING:
            population_matrix = population_matrix
        
        if new_population is None:
            if speed is None:
                new_population = [indiv.change_genotype(self.encoding.encode(population_matrix[idx, :])) for idx, indiv in enumerate(population)]
            else:
                new_population = [indiv.change_genotype(self.encoding.encode(population_matrix[idx, :]), speed[idx, :]) for idx, indiv in enumerate(population)]
        
        return new_population