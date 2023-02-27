import sys
sys.path.append("../..")

from PyEvolAlg import *
from PyEvolAlg.benchmarks.benchmarkFuncs import *
from PyEvolAlg.Metaheuristics.Individual import Indiv 

import unittest


repetitions = 100
popul_size = 100
vecsize = 100
objfunc = Sphere(vecsize)
random_population = [Indiv(objfunc, 100*np.random.random(size=vecsize)) for i in range(popul_size)]

class OperatorRealTests(unittest.TestCase):
    def test_1pcross(self):
        op = OperatorReal("1point")
        result = op(random.choice(random_population), random_population, objfunc)
        self.assertIsInstance(result, np.ndarray)
    
    def test_2pcross(self):
        op = OperatorReal("2point")
        result = op(random.choice(random_population), random_population, objfunc)
        self.assertIsInstance(result, np.ndarray)
    
    def test_mpcross(self):
        op = OperatorReal("Multipoint")
        result = op(random.choice(random_population), random_population, objfunc)
        self.assertIsInstance(result, np.ndarray)
    
    def test_waverage(self):
        op = OperatorInt("WeightedAvg", {"F": 0.75})
        for p in np.linspace(0, 1, 100):
            op.params["Cr"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_blxcross(self):
        op = OperatorReal("BLXalpha", {"Cr": 0.75})
        for p in np.linspace(0, 1, 100):
            op.params["Cr"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_sbxcross(self):
        op = OperatorReal("SBX", {"Cr": 0.5})
        for p in np.linspace(0, 1, 100):
            op.params["Cr"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_multicross(self):
        op = OperatorReal("Multicross", {"N": 2})
        for n in range(2, popul_size):
            op.params["N"] = n
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_cinteravg(self):
        op = OperatorReal("CrossInterAvg", {"N": 2})
        for n in range(2, popul_size):
            op.params["N"] = n
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_perm(self):
        op = OperatorReal("Perm", {"N": 2})
        for n in range(2, vecsize):
            op.params["N"] = n
            chosen = random.choice(random_population)
            result = op.evolve(chosen, random_population, objfunc)
            self.assertTrue((np.unique(chosen.vector) == np.unique(result)).all())
            
    def test_mutrand(self):
        op = OperatorReal("MutRand", {"method": "Gauss", "F":0.1, "N":2})
        for n in range(2, vecsize):
            op.params["N"] = n
            chosen = random.choice(random_population)
            result = op.evolve(chosen, random_population, objfunc)
            self.assertTrue(np.count_nonzero(chosen.vector == result) >= vecsize-n)
        
        op.params["N"] = 2
        for m in ["Gauss", "Cauchy", "Laplace", "Uniform"]:
            op.params["method"] = m
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_randsample(self):
        op = OperatorReal("RandSample", {"method": "Gauss", "F":1})
        for m in ["Gauss", "Cauchy", "Laplace", "Uniform"]:
            for f in np.linspace(0, 10, 300):
                op.params["method"] = m
                op.params["F"] = f
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
    
    def test_randnoise(self):
        op = OperatorReal("RandNoise", {"method": "Gauss", "F":1})
        for m in ["Gauss", "Cauchy", "Laplace", "Uniform"]:
            for f in np.linspace(0, 10, 300):
                op.params["method"] = m
                op.params["F"] = f
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
    
    def test_mutsample(self):
        op = OperatorReal("MutSample", {"method": "Gauss", "F":1, "N":2})
        for m in ["Gauss", "Cauchy", "Laplace", "Uniform"]:
            op.params["method"] = m
            for n in range(2, vecsize):
                op.params["N"] = n
                chosen = random.choice(random_population)
                result = op(chosen, random_population, objfunc)
                self.assertTrue(np.count_nonzero(chosen.vector == result) >= vecsize-n)
    
    def test_gauss(self):
        op = OperatorReal("Gauss", {"F": 0.1})
        for p in np.linspace(0, 1, 100):
            op.params["F"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_cauchy(self):
        op = OperatorReal("Cauchy", {"F": 0.1})
        for p in np.linspace(0, 1, 100):
            op.params["F"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_laplace(self):
        op = OperatorReal("Laplace", {"F": 0.1})
        for p in np.linspace(0, 1, 100):
            op.params["F"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)

    def test_uniform(self):
        op = OperatorReal("Uniform", {"Low": -10, "Up": 10})
        for p in np.linspace(-10, 10, 100):
            op.params["Up"] = p
            result = op(random.choice(random_population), random_population, objfunc)
            self.assertIsInstance(result, np.ndarray)
    
    def test_debest1(self):
        op = OperatorReal("DE/best/1", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)

    def test_derand1(self):
        op = OperatorReal("DE/rand/1", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
    
    def test_debest2(self):
        op = OperatorReal("DE/best/2", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)

    def test_derand2(self):
        op = OperatorReal("DE/rand/2", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
    
    def test_dectrand1(self):
        op = OperatorReal("DE/current-to-rand/1", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
        
    def test_dectbest1(self):
        op = OperatorReal("DE/current-to-best/1", {"F":0.7, "Cr":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["Cr"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertIsInstance(result, np.ndarray)
    
    def test_dectpbest1(self):
        op = OperatorReal("DE/current-to-pbest/1", {"F":0.7, "Cr":0.8, "P":0.11})
        for f in np.linspace(0, 1, 20):
            op.params["F"] = f
            for cr in np.linspace(0, 1, 20):
                op.params["Cr"] = cr
                for p in np.linspace(0, 1, 20):
                    op.params["P"] = p
                    result = op(random.choice(random_population), random_population, objfunc)
                    self.assertIsInstance(result, np.ndarray)
    
    def test_mutate1sigma(self):
        op = OperatorReal("Mutate1Sigma", {"epsilon":0.7, "tau":0.8})
        for f in np.linspace(0, 1, 30):
            op.params["epsilon"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["tau"] = cr
                result = op(random.choice(random_population), random_population, objfunc)
                self.assertTrue(np.all(result >= op.params["epsilon"]))
    
    def test_mutateNsigma(self):
        op = OperatorReal("Mutate1Sigma", {"epsilon":0.7, "tau":0.8, "tau_multiple":0.11})
        for f in np.linspace(0, 1, 30):
            op.params["epsilon"] = f
            for cr in np.linspace(0, 1, 30):
                op.params["tau"] = cr
                for p in np.linspace(0, 1, 30):
                    op.params["tau_multiple"] = p
                    result = op(random.choice(random_population), random_population, objfunc)
                    self.assertTrue(np.all(result >= op.params["epsilon"]))
    
    def test_sampsigma(self):
        op = OperatorReal("SampleSigma", {"N": 2, "epsilon":0.7, "tau":0.8})
        for n in range(2, vecsize):
            op.params["N"] = n
            for f in np.linspace(0, 1, 30):
                op.params["epsilon"] = f
                for cr in np.linspace(0, 1, 30):
                    op.params["tau"] = cr
                    result = op(random.choice(random_population), random_population, objfunc)
                    self.assertTrue(np.count_nonzero(result >= op.params["epsilon"]) >= op.params["N"])
    
    def test_custom(self):
        def gaussian_custom(indiv, population, objfunc, params):
            solution = indiv.vector.copy()
            return solution + np.random.normal(0, params["F"], size=solution.shape)
        
        op = OperatorReal("Custom", {"function": gaussian_custom, "F":5})
        result = op(random.choice(random_population), random_population, objfunc)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()