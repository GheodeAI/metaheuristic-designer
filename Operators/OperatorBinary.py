from .Operator import Operator
from .operatorFunctions import *


class OperatorBinary(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, name, params = None):
        """
        Constructor for the Operator class
        """

        super().__init__(name, params)
    
    
    def evolve(self, solution, population, objfunc):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """

        result = None
        others = [i for i in population if i != solution]
        if len(others) > 1:
            solution2 = random.choice(others)
        else:
            solution2 = solution
        
        if self.name == "1point":
            result = cross1p(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "2point":
            result = cross2p(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "multipoint":
            result = crossMp(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "xor" or self.name == "fliprandom":
            result = xorMask(solution.vector.copy(), self.params["N"], mode="bin")
        elif self.name == "xorcross" or self.name == "flipcross":
            result = xorCross(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "randsample":
            self.params["method"] = "Bernouli"
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "mutsample":
            self.params["method"] = "Bernouli"
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "dummy":
            result = dummyOp(solution.vector.copy(), self.params["F"])
        elif self.name == "nothing":
            result = solution.vector.copy()
        elif self.name == "custom":
            fn = self.params["function"]
            result = fn(solution, population, objfunc, self.params)
        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)

        return (result != 0).astype(np.int32)