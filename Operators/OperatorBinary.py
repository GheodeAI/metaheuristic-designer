from .Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .vector_operator_functions import *


class OperatorBinary(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the Operator class
        """

        super().__init__(name, params)
    
    
    def evolve(self, indiv, population, objfunc, global_best=None):
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
        
        
        if self.name == "1point":
            new_indiv.genotype = cross1p(new_indiv.genotype, solution2.genotype.copy())

        elif self.name == "2point":
            new_indiv.genotype = cross2p(new_indiv.genotype, solution2.genotype.copy())

        elif self.name == "multipoint":
            new_indiv.genotype = crossMp(new_indiv.genotype, solution2.genotype.copy())

        elif self.name == "multicross":
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["N"])

        elif self.name == "perm":
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.name == "xor" or self.name == "fliprandom":
            new_indiv.genotype = xorMask(new_indiv.genotype, params["N"], mode="bin")

        elif self.name == "xorcross" or self.name == "flipcross":
            new_indiv.genotype = xorCross(new_indiv.genotype, solution2.genotype.copy())

        elif self.name == "randsample":
            params["method"] = "bernouli"
            new_indiv.genotype = randSample(new_indiv.genotype, population, params)

        elif self.name == "mutsample":
            params["method"] = "bernouli"
            new_indiv.genotype = mutateSample(new_indiv.genotype, population, params)
        
        elif self.name == "random":
            new_indiv.genotype = objfunc.random_solution()
        
        elif self.name == "randommask":
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.genotype.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.genotype[mask_pos] = objfunc.random_solution()[mask_pos]

        elif self.name == "dummy":
            new_indiv.genotype = dummyOp(new_indiv.genotype, params["F"])

        elif self.name == "nothing":
            pass

        elif self.name == "custom":
            fn = params["function"]
            new_indiv.genotype = fn(indiv, population, objfunc, params)

        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)

        new_indiv.genotype = (new_indiv.genotype != 0).astype(np.int32)
        return new_indiv