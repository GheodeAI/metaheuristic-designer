from __future__ import annotations
from copy import copy
import enum
from enum import Enum
import numpy as np

from .operator_functions.mutation import *
from .operator_functions.crossover import *
from .operator_functions.permutation import *
from .operator_functions.differential_evolution import *
from .operator_functions.swarm import *
from ..operator import Operator
from ..param_scheduler import ParamScheduler
from ..encoding import Encoding
from ..encodings import ParameterExtendingEncoding
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

    MUTATE1SIGMA = enum.auto()
    MUTATENSIGMAS = enum.auto()
    SAMPLESIGMA = enum.auto()

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
    "mutate1sigma": VectorOpMethods.MUTATE1SIGMA,
    "mutatensigmas": VectorOpMethods.MUTATENSIGMAS,
    "samplesigma": VectorOpMethods.SAMPLESIGMA,
    "random": VectorOpMethods.RANDOM,
    "randommask": VectorOpMethods.RANDOM_MASK,
    "dummy": VectorOpMethods.DUMMY,
    "custom": VectorOpMethods.CUSTOM,
    "nothing": VectorOpMethods.NOTHING,
}


class VectorOperator(Operator):
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

    def __init__(self, method: str, params: ParamScheduler | dict = None, name: str = None, use_params : bool = False, encoding: Encoding = None):
        """
        Constructor for the OperatorPerm class
        """

        if name is None:
            name = method

        if params == "default":
            params = {
                "F": 0.5,
                "Cr": 0.8,
                "N": 5,
                "p": 0.5,
                "P": 0.1,
                "distrib": "gauss",
                "Low": -10,
                "Up": 10,
                "mu": 2,
                "epsilon": 0.1,
                "tau": 0.1,
                "tau_multiple": 0.1,
                "function": lambda x, y, **z: x,
            }

        super().__init__(params=params, name=name, use_params=use_params, encoding=encoding)

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

    def evolve(self, population, initializer=None):
        new_population = None
        population_matrix = copy(population.genotype_matrix)
        fitness_array = copy(population.fitness)
        global_best = population.best
        historical_best = population.historical_best_matrix

        params = copy(self.params)

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(RAND_GEN.random(population_matrix.shape[1]) < params["Cr"])

        if "N" in params:
            params["N"] = round(params["N"])
            params["N"] = min(params["N"], population_matrix.shape[1])

        # Perform one of the methods (switch-case like structure)
        match self.method:
            ## Cross operations
            case VectorOpMethods.ONE_POINT:
                population_matrix = cross_1p(population_matrix, fitness_array, **params)

            case VectorOpMethods.TWO_POINT:
                population_matrix = cross_2p(population_matrix, fitness_array, **params)

            case VectorOpMethods.MULTIPOINT:
                population_matrix = cross_mp(population_matrix, fitness_array, **params)

            case VectorOpMethods.WEIGHTED_AVG:
                population_matrix = weighted_average_cross(population_matrix, fitness_array, **params)

            case VectorOpMethods.BLXALPHA:
                population_matrix = blxalpha(population_matrix, fitness_array, **params)

            case VectorOpMethods.SBX:
                population_matrix = sbx(population_matrix, fitness_array, **params)

            case VectorOpMethods.XOR_CROSS:
                population_matrix = xor_cross(population_matrix, fitness_array, **params)

            case VectorOpMethods.MULTICROSS:
                population_matrix = multi_cross(population_matrix, fitness_array, **params)

            case VectorOpMethods.CROSSINTERAVG:
                population_matrix = cross_inter_avg(population_matrix, fitness_array, **params)

            ## Mutation operators
            case VectorOpMethods.XOR:
                population_matrix = xor_mask(population_matrix, fitness_array, **params)

            case VectorOpMethods.PERM:
                population_matrix = permute_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.GAUSS:
                population_matrix = gaussian_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.LAPLACE:
                population_matrix = laplace_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.CAUCHY:
                population_matrix = cauchy_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.UNIFORM:
                population_matrix = uniform_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.POISSON:
                population_matrix = poisson_mutation(population_matrix, fitness_array, **params)

            case VectorOpMethods.MUTNOISE:
                population_matrix = mutate_noise(population_matrix, fitness_array, **params)

            case VectorOpMethods.MUTSAMPLE:
                population_matrix = mutate_sample(population_matrix, fitness_array, **params)

            case VectorOpMethods.RANDNOISE:
                population_matrix = rand_noise(population_matrix, fitness_array, **params)

            case VectorOpMethods.RANDSAMPLE:
                population_matrix = rand_sample(population_matrix, fitness_array, **params)

            case VectorOpMethods.GENERATE:
                population_row = generate_statistic(population_matrix, fitness_array, **params)
                population_matrix = np.tile(population_row, population_matrix.shape[0]).reshape(population_matrix.shape)

            ## Differential evolution operators
            case VectorOpMethods.DE_RAND_1:
                population_matrix = DE_rand1(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_BEST_1:
                population_matrix = DE_best1(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_RAND_2:
                population_matrix = DE_rand2(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_BEST_2:
                population_matrix = DE_best2(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_CTRAND_1:
                population_matrix = DE_current_to_rand1(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_CTBEST_1:
                population_matrix = DE_current_to_best1(population_matrix, fitness_array, **params)

            case VectorOpMethods.DE_CTPBEST_1:
                population_matrix = DE_current_to_pbest1(population_matrix, fitness_array, **params)

            ## Adaptative mutations
            case VectorOpMethods.MUTATE1SIGMA:
                population_matrix = mutate_1_sigma(population_matrix, fitness_array, **params)

            case VectorOpMethods.MUTATENSIGMAS:
                population_matrix = mutate_n_sigmas(population_matrix, fitness_array, **params)

            case VectorOpMethods.SAMPLESIGMA:
                population_matrix = sample_1_sigma(population_matrix, fitness_array, **params)

            ## Other operators
            case VectorOpMethods.RANDOM:
                new_population = initializer.generate_population(population.objfunc, len(population))

            case VectorOpMethods.RANDOM_MASK:
                mask_pos = np.tile(
                    np.arange(population_matrix.shape[1]) < params["N"],
                    population_matrix.shape[0],
                ).reshape(population_matrix.shape)
                mask_pos = RAND_GEN.permuted(mask_pos, axis=1)

                random_population = initializer.generate_population(population.objfunc, len(population))

                population_matrix[mask_pos] = random_population.genotype_matrix[mask_pos]

            case VectorOpMethods.DUMMY:
                population_matrix = dummy_op(population_matrix, fitness_array, **params)

            case VectorOpMethods.CUSTOM:
                fn = params["function"]
                population_matrix = fn(population_matrix, fitness_array, **params)

            case VectorOpMethods.NOTHING:
                new_population = copy(population)

        if new_population is None:
            population_matrix = self.encoding.encode(population_matrix)

            # Only evolve solution parameters, the rest is managed in a specific way by each operator
            new_population = population.update_genotype_matrix(population_matrix)

        return new_population
