from .Operator import Operator
from ..ParamScheduler import ParamScheduler
from typing import Union
from .vector_operator_functions import *


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
            params["N"] = np.count_nonzero(np.random.random(indiv.genotype.size) < params["Cr"])

        params["N"] = round(params["N"])
        

        if self.name == "1point":
            new_indiv.genotype = cross1p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.name == "2point":
            new_indiv.genotype = cross2p(new_indiv.genotype, indiv2.genotype.copy())

        elif self.name == "multipoint":
            new_indiv.genotype = crossMp(new_indiv.genotype, indiv2.genotype.copy())

        elif self.name == "weightedAvg":
            new_indiv.genotype = weightedAverage(new_indiv.genotype, indiv2.genotype.copy(), params["F"])

        elif self.name == "blxalpha":
            new_indiv.genotype = blxalpha(new_indiv.genotype, indiv2.genotype.copy(), params["Cr"])
            
        elif self.name == "multicross":
            new_indiv.genotype = multiCross(new_indiv.genotype, others, params["N"])
        
        elif self.name == "xor":
            new_indiv.genotype = xorMask(new_indiv.genotype, self.params["N"])

        elif self.name == "xorcross":
            new_indiv.genotype = xorCross(new_indiv.genotype, solution2.genotype.copy())

        elif self.name == "crossinteravg":
            new_indiv.genotype = crossInterAvg(new_indiv.genotype, others, params["N"])

        elif self.name == "perm":
            new_indiv.genotype = permutation(new_indiv.genotype, params["N"])

        elif self.name == "gauss":
            new_indiv.genotype = gaussian(new_indiv.genotype, params["F"])

        elif self.name == "laplace":
            new_indiv.genotype = laplace(new_indiv.genotype, params["F"])
            
        elif self.name == "cauchy":
            new_indiv.genotype = cauchy(new_indiv.genotype, params["F"])

        elif self.name == "uniform":
            new_indiv.genotype = uniform(new_indiv.genotype, params["Low"], params["Up"])
        
        elif self.name == "poisson":
            new_indiv.genotype = poisson(new_indiv.genotype, self.params["F"])

        elif self.name == "mutrand" or self.name == "mutnoise":
            new_indiv.genotype = mutateRand(new_indiv.genotype, others, params)

        elif self.name == "randnoise":
            new_indiv.genotype = randNoise(new_indiv.genotype, params)

        elif self.name == "randsample":
            new_indiv.genotype = randSample(new_indiv.genotype, others, params)

        elif self.name == "mutsample":
            new_indiv.genotype = mutateSample(new_indiv.genotype, others, params)

        elif self.name == "de/rand/1":
            new_indiv.genotype = DERand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/best/1":
            new_indiv.genotype = DEBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/rand/2":
            new_indiv.genotype = DERand2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/best/2":
            new_indiv.genotype = DEBest2(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-rand/1":
            new_indiv.genotype = DECurrentToRand1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-best/1":
            new_indiv.genotype = DECurrentToBest1(new_indiv.genotype, others, params["F"], params["Cr"])

        elif self.name == "de/current-to-pbest/1":
            new_indiv.genotype = DECurrentToPBest1(new_indiv.genotype, others, params["F"], params["Cr"], params["P"])

        elif self.name == "lshade":
            params["Cr"] = np.clip(np.random.normal(params["Cr"], 0.1), 0, 1)
            params["F"] = np.clip(np.random.normal(params["F"], 0.1), 0, 1)

            new_indiv.genotype = DECurrentToPBest1(new_indiv.genotype, others, params["F"], params["Cr"])      

        elif self.name == "sa":
            new_indiv.genotype = simAnnealing(indiv, params["F"], objfunc, params["temp_ch"], params["iter"])

        elif self.name == "hs":
            new_indiv.genotype = harmonySearch(new_indiv.genotype, others, params["F"], params["Cr"], params["Par"])

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
        
            
        return np.round(new_indiv)