from .Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from .operator_functions import *


class OperatorInt(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the Operator class
        """

        super().__init__(name, params)
    
    
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
        
        if self.name == "1point":
            new_indiv.vector = cross1p(new_indiv.vector, solution2.vector.copy())
        elif self.name == "2point":
            new_indiv.vector = cross2p(new_indiv.vector, solution2.vector.copy())
        elif self.name == "multipoint":
            new_indiv.vector = crossMp(new_indiv.vector, solution2.vector.copy())
        elif self.name == "weightedAvg":
            new_indiv.vector = weightedAverage(new_indiv.vector, solution2.vector.copy(), self.params["F"])
        elif self.name == "blxalpha":
            new_indiv.vector = blxalpha(new_indiv.vector, solution2.vector.copy(), self.params["Cr"])
        elif self.name == "multicross":
            new_indiv.vector = multiCross(new_indiv.vector, others, self.params["N"])
        elif self.name == "crossinteravg":
            new_indiv.vector = crossInterAvg(new_indiv.vector, others, self.params["N"])
        elif self.name == "perm":
            new_indiv.vector = permutation(new_indiv.vector, self.params["N"])
        elif self.name == "xor":
            new_indiv.vector = xorMask(new_indiv.vector, self.params["N"])
        elif self.name == "xorcross":
            new_indiv.vector = xorCross(new_indiv.vector, solution2.vector.copy())
        elif self.name == "mutrand":
            new_indiv.vector = mutateRand(new_indiv.vector, population, self.params)
        elif self.name == "randnoise":
            new_indiv.vector = randNoise(new_indiv.vector, self.params)
        elif self.name == "randsample":
            new_indiv.vector = randSample(new_indiv.vector, population, self.params)
        elif self.name == "mutsample":
            new_indiv.vector = mutateSample(new_indiv.vector, population, self.params)
        elif self.name == "gauss":
            new_indiv.vector = gaussian(new_indiv.vector, self.params["F"])
        elif self.name == "laplace":
            new_indiv.vector = laplace(new_indiv.vector, self.params["F"])
        elif self.name == "cauchy":
            new_indiv.vector = cauchy(new_indiv.vector, self.params["F"])
        elif self.name == "uniform":
            new_indiv.vector = uniform(new_indiv.vector, self.params["Low"], self.params["Up"])
        elif self.name == "poisson":
            new_indiv.vector = poisson(new_indiv.vector, self.params["F"])
        elif self.name == "de/rand/1":
            new_indiv.vector = DERand1(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/best/1":
            new_indiv.vector = DEBest1(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/rand/2":
            new_indiv.vector = DERand2(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/best/2":
            new_indiv.vector = DEBest2(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-rand/1":
            new_indiv.vector = DECurrentToRand1(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-best/1":
            new_indiv.vector = DECurrentToBest1(new_indiv.vector, others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-pbest/1":
            new_indiv.vector = DECurrentToPBest1(new_indiv.vector, others, self.params["F"], self.params["Cr"], self.params["P"])
        elif self.name == "lshade":
            self.params["Cr"] = np.random.normal(self.params["Cr"], 0.1)
            self.params["F"] = np.random.normal(self.params["F"], 0.1)

            self.params["Cr"] = np.clip(self.params["Cr"], 0, 1)
            self.params["F"] = np.clip(self.params["F"], 0, 1)

            new_indiv.vector = DECurrentToPBest1(new_indiv.vector, others, self.params["F"], self.params["Cr"]) 
        elif self.name == "sa":
            new_indiv.vector = simAnnealing(indiv, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.name == "hs":
            new_indiv.vector = harmonySearch(new_indiv.vector, population, self.params["F"], self.params["Cr"], self.params["Par"])
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

        return np.round(new_indiv.vector)