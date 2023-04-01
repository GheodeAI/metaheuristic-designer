from ..Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .vector_operator_functions import *

_bin_ops = [
    "1point",
    "2point",
    "multipoint",
    "multicross",
    "xor",
    "xorcross",
    "perm",
    "randsample",
    "mutsample",
    "random",
    "randommask",
    "dummy",
    "custom",
    "nothing"
]

class OperatorBinary(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the Operator class
        """

        super().__init__(method, params)

        if self.method not in _bin_ops:
            raise ValueError(f"Binary operator \"{self.method}\" not defined")
    
    
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
        
        
        
        if self.method == "1point":
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "2point":
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "multipoint":
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "multicross":
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["Nindiv"])

        elif self.method == "perm":
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.method == "xor" or self.method == "fliprandom":
            new_indiv.genotype = xorMask(new_indiv.genotype, params["N"], mode="bin")

        elif self.method == "xorcross" or self.method == "flipcross":
            new_indiv.genotype = xorCross(new_indiv.genotype, indiv2.genotype.copy())

        elif self.method == "randsample":
            params["method"] = "bernouli"
            new_indiv.genotype = randSample(new_indiv.genotype, population, params)

        elif self.method == "mutsample":
            params["method"] = "bernouli"
            new_indiv.genotype = mutateSample(new_indiv.genotype, population, params)
        
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

        new_indiv.genotype = (new_indiv.genotype != 0).astype(np.int32)
        return new_indiv