from .Operator import Operator
from .operator_functions import *
from ..ParamScheduler import ParamScheduler
from typing import Union
from copy import copy

class OperatorReal(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name: str, params: Union[ParamScheduler, dict]=None):
        """
        Constructor for the OperatorReal class
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
            new_indiv.vector = cross1p(new_indiv.vector, indiv2.vector.copy())
        elif self.name == "2point":
            new_indiv.vector = cross2p(new_indiv.vector, indiv2.vector.copy())
        elif self.name == "multipoint":
            new_indiv.vector = crossMp(new_indiv.vector, indiv2.vector.copy())
        elif self.name == "weightedAvg":
            new_indiv.vector = weightedAverage(new_indiv.vector, indiv2.vector.copy(), self.params["F"])
        elif self.name == "blxalpha":
            new_indiv.vector = blxalpha(new_indiv.vector, indiv2.vector.copy(), self.params["Cr"])
        elif self.name == "sbx":
            new_indiv.vector = sbx(new_indiv.vector, indiv2.vector.copy(), self.params["Cr"])
        elif self.name == "multicross":
            new_indiv.vector = multiCross(new_indiv.vector, others, self.params["N"])
        elif self.name == "crossinteravg":
            new_indiv.vector = crossInterAvg(new_indiv.vector, others, self.params["N"])
        elif self.name == "mutate1sigma":
            new_indiv.vector = mutate_1_sigma(new_indiv.vector[0], self.params["epsilon"], self.params["tau"])
        elif self.name == "mutatensigmas":
            new_indiv.vector = mutate_n_sigmas(new_indiv.vector, self.params["epsilon"], self.params["tau"], self.params["tau_multiple"])
        elif self.name == "samplesigma":
            new_indiv.vector = sample_1_sigma(new_indiv.vector, self.params["N"], self.params["epsilon"], self.params["tau"])
        elif self.name == "perm":
            new_indiv.vector = permutation(new_indiv.vector, self.params["N"])
        elif self.name == "gauss":
            new_indiv.vector = gaussian(new_indiv.vector, self.params["F"])
        elif self.name == "laplace":
            new_indiv.vector = laplace(new_indiv.vector, self.params["F"])
        elif self.name == "cauchy":
            new_indiv.vector = cauchy(new_indiv.vector, self.params["F"])
        elif self.name == "uniform":
            new_indiv.vector = uniform(new_indiv.vector, self.params["Low"], self.params["Up"])
        elif self.name == "mutrand":
            new_indiv.vector = mutateRand(new_indiv.vector, others, self.params)
        elif self.name == "randnoise":
            new_indiv.vector = randNoise(new_indiv.vector, self.params)
        elif self.name == "randsample":
            new_indiv.vector = randSample(new_indiv.vector, others, self.params)
        elif self.name == "mutsample":
            new_indiv.vector = mutateSample(new_indiv.vector, others, self.params)
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
            new_indiv.vector = harmonySearch(new_indiv.vector, others, self.params["F"], self.params["Cr"], self.params["Par"])
        elif self.name == "pso":
            new_indiv = pso_operator(indiv, others, global_best, self.params["w"], self.params["c1"], self.params["c2"])
        elif self.name == "firefly":
            new_indiv.vector = firefly(indiv, others, objfunc, self.params["a"], self.params["b"], self.params["d"], self.params["g"])
        elif self.name == "random":
            new_indiv.vector = objfunc.random_solution()
        elif self.name == "randommask":
            n = round(self.params["N"])
            mask_pos = np.hstack([np.ones(n), np.zeros(new_indiv.vector.size - n)]).astype(bool)
            np.random.shuffle(mask_pos)

            new_indiv.vector[mask_pos] = objfunc.random_solution()[mask_pos]
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
            
        return new_indiv