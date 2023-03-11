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

        params = copy(self.params)
        
        if "N" in params:
            params["N"] = round(params["N"])

        if "Cr" in params and "N" not in params:
            params["N"] = np.count_nonzero(np.random.random(indiv.vector.size) < params["Cr"])

        params["N"] = round(params["N"])
        

        if self.name == "1point":
            new_indiv.vector = cross1p(new_indiv.vector, indiv2.vector.copy())

        elif self.name == "2point":
            new_indiv.vector = cross2p(new_indiv.vector, indiv2.vector.copy())

        elif self.name == "multipoint":
            new_indiv.vector = crossMp(new_indiv.vector, indiv2.vector.copy())

        elif self.name == "weightedAvg":
            new_indiv.vector = weightedAverage(new_indiv.vector, indiv2.vector.copy(), params["F"])

        elif self.name == "blxalpha":
            new_indiv.vector = blxalpha(new_indiv.vector, indiv2.vector.copy(), params["Cr"])
            
        elif self.name == "multicross":
            new_indiv.vector = multiCross(new_indiv.vector, others, params["N"])
        
        elif self.name == "xor":
            new_indiv.vector = xorMask(new_indiv.vector, self.params["N"])

        elif self.name == "xorcross":
            new_indiv.vector = xorCross(new_indiv.vector, solution2.vector.copy())

        elif self.name == "crossinteravg":
            new_indiv.vector = crossInterAvg(new_indiv.vector, others, params["N"])

        elif self.name == "perm":
            new_indiv.vector = permutation(new_indiv.vector, params["N"])

        elif self.name == "gauss":
            new_indiv.vector = gaussian(new_indiv.vector, params["F"])

        elif self.name == "laplace":
            new_indiv.vector = laplace(new_indiv.vector, params["F"])
            
        elif self.name == "cauchy":
            new_indiv.vector = cauchy(new_indiv.vector, params["F"])

        elif self.name == "uniform":
            new_indiv.vector = uniform(new_indiv.vector, params["Low"], params["Up"])
        
        elif self.name == "poisson":
            new_indiv.vector = poisson(new_indiv.vector, self.params["F"])

        elif self.name == "mutrand" or self.name == "mutnoise":
            new_indiv.vector = mutateRand(new_indiv.vector, others, params)

        elif self.name == "randnoise":
            new_indiv.vector = randNoise(new_indiv.vector, params)

        elif self.name == "randsample":
            new_indiv.vector = randSample(new_indiv.vector, others, params)

        elif self.name == "mutsample":
            new_indiv.vector = mutateSample(new_indiv.vector, others, params)

        elif self.name == "de/rand/1":
            new_indiv.vector = DERand1(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/best/1":
            new_indiv.vector = DEBest1(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/rand/2":
            new_indiv.vector = DERand2(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/best/2":
            new_indiv.vector = DEBest2(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-rand/1":
            new_indiv.vector = DECurrentToRand1(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-best/1":
            new_indiv.vector = DECurrentToBest1(new_indiv.vector, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-pbest/1":
            new_indiv.vector = DECurrentToPBest1(new_indiv.vector, others, params["F"], params["Cr"], params["P"])

        elif self.name == "lshade":
            params["Cr"] = np.clip(np.random.normal(params["Cr"], 0.1), 0, 1)
            params["F"] = np.clip(np.random.normal(params["F"], 0.1), 0, 1)

            new_indiv.vector = DECurrentToPBest1(new_indiv.vector, others, params["F"], params["Cr"])      

        elif self.name == "sa":
            new_indiv.vector = simAnnealing(indiv, params["F"], objfunc, params["temp_ch"], params["iter"])

        elif self.name == "hs":
            new_indiv.vector = harmonySearch(new_indiv.vector, others, params["F"], params["Cr"], params["Par"])

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
        
            
        return np.round(new_indiv)