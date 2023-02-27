import sys
sys.path.append("../..")

from PyEvolAlg import *
from PyEvolAlg.benchmarks.benchmarkFuncs import *
from PyEvolAlg.Metaheuristics.Individual import Indiv 

import unittest

vecsize = 100
objfunc = Sphere(vecsize)

mutation_op = OperatorReal("Gauss", {"F":0.01})
cross_op = OperatorReal("Multipoint")
diffev_op = OperatorReal("DE/best/1", {"F":0.7, "Cr":0.8})
parent_select_op = ParentSelection("Best", {"amount": 75})
replace_op = SurvivorSelection("(m+n)")

mutation_ch_op = OperatorReal("Gauss", ParamScheduler("Linear", {"F":[1, 0.01]}))
cross_ch_op = OperatorReal("BLXalpha", ParamScheduler("Linear", {"Cr":[0.8, 0.5]}))
diffev_ch_op = OperatorReal("DE/best/1", ParamScheduler("Linear", {"F":[0.8, 0.5], "Cr":[0.8, 0.6]}))
parent_select_ch_op = ParentSelection("Best", ParamScheduler("Linear", {"amount": [100, 50]}))
replace_ch_op = SurvivorSelection("Elitism", ParamScheduler("Linear", {"amount": [2, 20]}))


base_params = {
    "popSize": 100,

    "stop_cond": "ngen",
    "time_limit": 3.0,
    "Ngen": 50,
    "Neval": 1000,
    "fit_target": 1,

    "verbose": False,
    "v_timer": -1
}


class OperatorIntTests(unittest.TestCase):
    def test_ES(self):
        params = base_params.copy()
        params.update({
            "offspringSize": 500
        })

        c = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_ES_change(self):
        params = base_params.copy()
        params.update({
            "offspringSize": [500, 100]
        })

        params_ch = ParamScheduler("Linear", params)

        c = ES(objfunc, mutation_ch_op, cross_ch_op, parent_select_ch_op, replace_ch_op, params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_GA(self):
        params = base_params.copy()
        params.update({
            "pmut": 0.2,
            "pcross":0.9
        })

        c = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_GA_change(self):
        params = base_params.copy()
        params.update({
            "pmut": [0.2, 0.1],
            "pcross": [0.8, 1]
        })

        params_ch = ParamScheduler("Linear", params)

        c = Genetic(objfunc, mutation_ch_op, cross_ch_op, parent_select_ch_op, replace_ch_op, params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_PSO(self):
        params = base_params.copy()
        params.update({
            "w": 0.2,
            "c1": 0.9,
            "c2": 0.9
        })

        c = PSO(objfunc, params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_PSO_chage(self):
        params = base_params.copy()
        params.update({
            "w": [0.2, 0.3],
            "c1": [0.8, 1],
            "c2": [0.8, 1]
        })

        params_ch = ParamScheduler("Linear", params)

        c = PSO(objfunc, params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_DE(self):
        params = base_params.copy()
        params.update({})

        c = DE(objfunc, diffev_op, SurvivorSelection("One-to-one"), params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_DE_chage(self):
        params = base_params.copy()

        params_ch = ParamScheduler("Linear", params)

        c = DE(objfunc, diffev_ch_op, SurvivorSelection("One-to-one"), params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_HS(self):
        params = base_params.copy()
        params.update({
            "HMCR": 0.9,
            "PAR" : 0.3,
            "BN" : 1
        })

        c = HS(objfunc, mutation_op, mutation_op, params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_HS_change(self):
        params = base_params.copy()
        params.update({
            "HMCR": [0.7, 0.9],
            "PAR" : [0.4, 0.2],
            "BN" : [0.5, 1]
        })

        params_ch = ParamScheduler("Linear", params)

        c = HS(objfunc, mutation_ch_op, mutation_ch_op, params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_SA(self):
        params = base_params.copy()
        params.update({
            "iter": 100,
            "temp_init": 100,
            "temp_ch": 0.9,
            "alpha" : 0.3
        })

        c = SimAnn(objfunc, mutation_op, params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_SA_change(self):
        params = base_params.copy()
        params.update({
            "iter": [50, 130],
            "temp_init": 100,
            "temp_ch": 0.9,
            "alpha" : 0.3
        })

        params_ch = ParamScheduler("Linear", params)

        c = SimAnn(objfunc, mutation_ch_op, params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])

    def test_CROSL(self):
        params = base_params.copy()
        params.update({
            "rho": 0.6,
            "Fb": 0.98,
            "Fd": 1,
            "Pd": 0.1,
            "k": 3,
            "K": 20,

            "group_subs": False,

            "dynamic": True,
            "dyn_method": "fitness",
            "dyn_metric": "best",
            "dyn_steps": 100,
            "prob_amp": 0.1
        })

        c = CRO_SL(objfunc, [mutation_op, cross_op], params)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])
    
    def test_CROSL_change(self):
        params = base_params.copy()
        params.update({
            "rho": [0.6, 0.8],
            "Fb": [0.9, 0.99],
            "Fd": [0.1, 0.2],
            "Pd": [0.75, 1],
            "k": [5, 3],
            "K": [20, 10],

            "group_subs": False,

            "dynamic": True,
            "dyn_method": "fitness",
            "dyn_metric": "best",
            "dyn_steps": 100,
            "prob_amp": [0.1, 0.05]
        })

        params_ch = ParamScheduler("Linear", params)

        c = CRO_SL(objfunc, [mutation_ch_op, cross_ch_op], params_ch)
        c.optimize()
        self.assertTrue(c.history[0] > c.history[-1])

if __name__ == "__main__":
    unittest.main()