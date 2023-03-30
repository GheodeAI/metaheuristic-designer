from ..Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .vector_operator_functions import *

_int_ops = [
    "1point",
    "2point",
    "multipoint",
    "weightedAvg",
    "blxalpha",
    "multicross",
    "xor",
    "xorcross",
    "crossinteravg",
    "perm",
    "gauss",
    "laplace",
    "cauchy",
    "uniform",
    "poisson",
    "mutrand",
    "mutnoise",
    "randnoise",
    "randsample",
    "mutsample",
    "de/rand/1",
    "de/best/1",
    "de/rand/2",
    "de/best/2",
    "de/current-to-rand/1",
    "de/current-to-best/1",
    "de/current-to-pbest/1",
    "pso",
    "firefly",
    "random",
    "randommask",
    "dummy",
    "custom",
    "nothing"
]

class OperatorInt(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict]=None, name=None):
        """
        Constructor for the Operator class
        """

        super().__init__(method, params, name)

        if self.method not in _int_ops:
            raise ValueError(f"Integer operator \"{self.method}\" not defined")
    
    
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
        
        if "N" in params:
            params["N"] = round(params["N"])

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(np.random.random(indiv.genotype.size) < params["Cr"])

        params["N"] = round(params["N"])
        

        if self.method == "1point":
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "2point":
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "multipoint":
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "weightedAvg":
            new_indiv.genotype = weightedAverage(new_indiv.genotype, indiv2.genotype.copy(), params["F"])

        elif self.method == "blxalpha":
            new_indiv.genotype = blxalpha(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])
            
        elif self.method == "multicross":
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["N"])
        
        elif self.method == "xor":
            new_indiv.genotype = xorMask(new_indiv.genotype, self.params["N"])

        elif self.method == "xorcross":
            new_indiv.genotype = xorCross(new_indiv.genotype, solution2.genotype.copy())

        elif self.method == "crossinteravg":
            new_indiv.genotype = crossInterAvg(new_indiv.genotype, others, params["N"])

        elif self.method == "perm":
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == "gauss":
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.method == "laplace":
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])
            
        elif self.method == "cauchy":
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.method == "uniform":
            new_indiv.genotype = uniform(new_indiv.genotype, params["Low"], params["Up"])
        
        elif self.method == "poisson":
            new_indiv.genotype = poisson(new_indiv.genotype, self.params["F"])

        elif self.method == "mutrand" or self.method == "mutnoise":
            new_indiv.genotype = mutateRand(new_indiv.genotype, others, params)

        elif self.method == "randnoise":
            new_indiv.genotype = randNoise(new_indiv.genotype, params)

        elif self.method == "randsample":
            new_indiv.genotype = randSample(new_indiv.genotype, others, params)

        elif self.method == "mutsample":
            new_indiv.genotype = mutateSample(new_indiv.genotype, others, params)

        elif self.method == "de/rand/1":
            new_indiv.genotype = DERand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/best/1":
            new_indiv.genotype = DEBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/rand/2":
            new_indiv.genotype = DERand2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/best/2":
            new_indiv.genotype = DEBest2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/current-to-rand/1":
            new_indiv.genotype = DECurrentToRand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/current-to-best/1":
            new_indiv.genotype = DECurrentToBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.method == "de/current-to-pbest/1":
            new_indiv.genotype = DECurrentToPBest1(new_indiv.genotype, others, params["F"], params["Cr"], params["P"])

        elif self.method == "random":
            new_indiv.genotype = objfunc.random_solution()

        elif self.method == "randommask":
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = objfunc.random_solution()[mask_pos]

        elif self.method == "dummy":
            new_indiv.genotype = dummyOp(new_indiv.genotype, params["F"])

        elif self.method == "custom":
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)
        
            
        new_indiv.genotype = np.round(new_indiv.genotype)
        return new_indiv