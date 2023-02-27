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
        elif self.name == "WeightedAvg":
            result = weightedAverage(solution.vector.copy(), solution2.vector.copy(), self.params["F"])
        elif self.name == "BLXalpha":
            result = blxalpha(solution.vector.copy(), solution2.vector.copy(), self.params["Cr"])
        elif self.name == "SBX":
            result = sbx(solution.vector.copy(), solution2.vector.copy(), self.params["Cr"])
        elif self.name == "Multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "CrossInterAvg":
            result = crossInterAvg(solution.vector.copy(), others, self.params["N"])
        elif self.name == "Mutate1Sigma":
            result = mutate_1_sigma(solution.vector.copy()[0], self.params["epsilon"], self.params["tau"])
        elif self.name == "MutateNSigmas":
            result = mutate_n_sigmas(solution.vector.copy(), self.params["epsilon"], self.params["tau"], self.params["tau_multiple"])
        elif self.name == "SampleSigma":
            result = sample_1_sigma(solution.vector.copy(), self.params["N"], self.params["epsilon"], self.params["tau"])
        elif self.name == "Perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "Gauss":
            result = gaussian(solution.vector.copy(), self.params["F"])
        elif self.name == "Laplace":
            result = laplace(solution.vector.copy(), self.params["F"])
        elif self.name == "Cauchy":
            result = cauchy(solution.vector.copy(), self.params["F"])
        elif self.name == "Uniform":
            result = uniform(solution.vector.copy(), self.params["Low"], self.params["Up"])
        elif self.name == "MutRand":
            result = mutateRand(solution.vector.copy(), population, self.params)
        elif self.name == "RandNoise":
            result = randNoise(solution.vector.copy(), self.params)
        elif self.name == "RandSample":
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "MutSample":
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "DE/rand/1":
            result = DERand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/best/1":
            result = DEBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/rand/2":
            result = DERand2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/best/2":
            result = DEBest2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-rand/1":
            result = DECurrentToRand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-best/1":
            result = DECurrentToBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-pbest/1":
            result = DECurrentToPBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"], self.params["P"])
        elif self.name == "LSHADE":
            self.params["Cr"] = np.random.normal(self.params["Cr"], 0.1)
            self.params["F"] = np.random.normal(self.params["F"], 0.1)

            self.params["Cr"] = np.clip(self.params["Cr"], 0, 1)
            self.params["F"] = np.clip(self.params["F"], 0, 1)

            result = DECurrentToPBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])            
        elif self.name == "SA":
            result = simAnnealing(solution, self.params["F"], objfunc, self.params["temp_ch"], self.params["iter"])
        elif self.name == "HS":
            result = harmonySearch(solution.vector.copy(), population, self.params["F"], self.params["Cr"], self.params["Par"])
        elif self.name == "Firefly":
            result = firefly(solution, population, objfunc, self.params["a"], self.params["b"], self.params["d"], self.params["g"])
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
            
        return result