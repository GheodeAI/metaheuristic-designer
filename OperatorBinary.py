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

        self.name = name
        super().__init__(self.name, params)
    
    
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
        elif self.name == "Multipoint":
            result = crossMp(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "Multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "Perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "Xor" or self.name == "FlipRandom":
            result = xorMask(solution.vector.copy(), self.params["N"], mode="bin")
        elif self.name == "XorCross" or self.name == "FlipCross":
            result = xorCross(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "RandSample":
            self.params["method"] = "Bernouli"
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "MutSample":
            self.params["method"] = "Bernouli"
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "Dummy":
            result = dummyOp(solution.vector.copy(), self.params["F"])
        elif self.name == "Nothing":
            result = solution.vector.copy()
        elif self.name == "Custom":
            fn = self.params["function"]
            result = fn(solution, population, objfunc, self.params)
        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)

        return (result != 0).astype(np.int32)