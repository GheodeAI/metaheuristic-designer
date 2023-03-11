from .Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
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
        
        if self.name == "1point":
            new_indiv.vector = cross1p(new_indiv.vector, solution2.vector.copy())
        elif self.name == "2point":
            new_indiv.vector = cross2p(new_indiv.vector, solution2.vector.copy())
        elif self.name == "multipoint":
            new_indiv.vector = crossMp(new_indiv.vector, solution2.vector.copy())
        elif self.name == "multicross":
            new_indiv.vector = multiCross(new_indiv.vector, others, self.params["N"])
        elif self.name == "perm":
            new_indiv.vector = permutation(new_indiv.vector, self.params["N"])
        elif self.name == "xor" or self.name == "fliprandom":
            new_indiv.vector = xorMask(new_indiv.vector, self.params["N"], mode="bin")
        elif self.name == "xorcross" or self.name == "flipcross":
            new_indiv.vector = xorCross(new_indiv.vector, solution2.vector.copy())
        elif self.name == "randsample":
            self.params["method"] = "Bernouli"
            new_indiv.vector = randSample(new_indiv.vector, population, self.params)
        elif self.name == "mutsample":
            self.params["method"] = "Bernouli"
            new_indiv.vector = mutateSample(new_indiv.vector, population, self.params)
        elif self.name == "random":
            new_indiv.vector = objfunc.random_solution()
        elif self.name == "dummy":
            new_indiv.vector = dummyOp(new_indiv.vector, self.params["F"])
        elif self.name == "nothing":
            new_indiv.vector = new_indiv.vector
        elif self.name == "custom":
            fn = self.params["function"]
            new_indiv.vector = fn(indiv, population, objfunc, self.params)
        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)

        return (new_indiv.vector != 0).astype(np.int32)