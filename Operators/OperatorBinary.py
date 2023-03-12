from .Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy
from .operator_functions import *


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
            params["N"] = np.count_nonzero(np.random.random(indiv.vector.size) < params["Cr"])

        params["N"] = round(params["N"])
        
        
        if self.name == "1point":
            new_indiv.vector = cross1p(new_indiv.vector, solution2.vector.copy())

        elif self.name == "2point":
            new_indiv.vector = cross2p(new_indiv.vector, solution2.vector.copy())

        elif self.name == "multipoint":
            new_indiv.vector = crossMp(new_indiv.vector, solution2.vector.copy())

        elif self.name == "multicross":
            new_indiv.vector = multiCross(new_indiv.vector, others, params["N"])

        elif self.name == "perm":
            new_indiv.vector = permutation(new_indiv.vector, params["N"])

        elif self.name == "xor" or self.name == "fliprandom":
            new_indiv.vector = xorMask(new_indiv.vector, params["N"], mode="bin")

        elif self.name == "xorcross" or self.name == "flipcross":
            new_indiv.vector = xorCross(new_indiv.vector, solution2.vector.copy())

        elif self.name == "randsample":
            params["method"] = "bernouli"
            new_indiv.vector = randSample(new_indiv.vector, population, params)

        elif self.name == "mutsample":
            params["method"] = "bernouli"
            new_indiv.vector = mutateSample(new_indiv.vector, population, params)
        
        elif self.name == "random":
            new_indiv.vector = objfunc.random_solution()
        
        elif self.name == "randommask":
            mask_pos = np.hstack([np.ones(params["N"]), np.zeros(new_indiv.vector.size - params["N"])]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.vector[mask_pos] = objfunc.random_solution()[mask_pos]

        elif self.name == "dummy":
            new_indiv.vector = dummyOp(new_indiv.vector, params["F"])

        elif self.name == "nothing":
            pass

        elif self.name == "custom":
            fn = params["function"]
            new_indiv.vector = fn(indiv, population, objfunc, params)

        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)

        new_indiv.vector = (new_indiv.vector != 0).astype(np.int32)
        return new_indiv