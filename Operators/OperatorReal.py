from .Operator import Operator
from .operatorFunctions import *


class OperatorReal(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name, params = None):
        """
        Constructor for the OperatorReal class
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
        elif self.name == "weightedAvg":
            result = weightedAverage(solution.vector.copy(), solution2.vector.copy(), self.params["F"])
        elif self.name == "blxalpha":
            result = blxalpha(solution.vector.copy(), solution2.vector.copy(), self.params["Cr"])
        elif self.name == "sbx":
            result = sbx(solution.vector.copy(), solution2.vector.copy(), self.params["Cr"])
        elif self.name == "multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "crossinteravg":
            result = crossInterAvg(solution.vector.copy(), others, self.params["N"])
        elif self.name == "mutate1sigma":
            result = mutate_1_sigma(solution.vector.copy()[0], self.params["epsilon"], self.params["tau"])
        elif self.name == "mutatensigmas":
            result = mutate_n_sigmas(solution.vector.copy(), self.params["epsilon"], self.params["tau"], self.params["tau_multiple"])
        elif self.name == "samplesigma":
            result = sample_1_sigma(solution.vector.copy(), self.params["N"], self.params["epsilon"], self.params["tau"])
        elif self.name == "perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "gauss":
            result = gaussian(solution.vector.copy(), self.params["F"])
        elif self.name == "laplace":
            result = laplace(solution.vector.copy(), self.params["F"])
        elif self.name == "cauchy":
            result = cauchy(solution.vector.copy(), self.params["F"])
        elif self.name == "uniform":
            result = uniform(solution.vector.copy(), self.params["Low"], self.params["Up"])
        elif self.name == "mutrand":
            result = mutateRand(solution.vector.copy(), population, self.params)
        elif self.name == "randnoise":
            result = randNoise(solution.vector.copy(), self.params)
        elif self.name == "randsample":
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "mutsample":
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "de/rand/1":
            result = DERand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/best/1":
            result = DEBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/rand/2":
            result = DERand2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/best/2":
            result = DEBest2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-rand/1":
            result = DECurrentToRand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-best/1":
            result = DECurrentToBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "de/current-to-pbest/1":
            result = DECurrentToPBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"], self.params["P"])
        elif self.name == "lshade":
            self.params["Cr"] = np.random.normal(self.params["Cr"], 0.1)
            self.params["F"] = np.random.normal(self.params["F"], 0.1)

            self.params["Cr"] = np.clip(self.params["Cr"], 0, 1)
            self.params["F"] = np.clip(self.params["F"], 0, 1)

            result = DECurrentToPBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])            
        elif self.name == "sa":
            result = simAnnealing(solution, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.name == "hs":
            result = harmonySearch(solution.vector.copy(), population, self.params["F"], self.params["Cr"], self.params["Par"])
        elif self.name == "pso":
            result = pso_operator(solution, population, objfunc, self.params["w"], self.params["c1"], self.params["c2"])
        elif self.name == "firefly":
            result = firefly(solution, population, objfunc, self.params["a"], self.params["b"], self.params["d"], self.params["g"])
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
            
        return result