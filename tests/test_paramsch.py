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
ordered_population = [Indiv(objfunc, i*np.ones(vecsize)) for i in np.linspace(0, 40, popul_size)]
#random_population = [Indiv(objfunc, 100*np.random.random(size=vecsize)) for i in range(popul_size)]

class ParamSchedulerTests(unittest.TestCase):
    def test1(self):
        pass

if __name__ == "__main__":
    unittest.main()