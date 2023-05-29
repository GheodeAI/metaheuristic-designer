import pytest

import numpy as np

from pyevolcomp import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from pyevolcomp.SearchMethods import GeneralSearch
from pyevolcomp.Operators import OperatorReal
from pyevolcomp.Initializers import UniformVectorInitializer
from pyevolcomp.Algorithms import *

from pyevolcomp.benchmarks import Sphere


test_params = {
    "stop_cond": "neval or cpu_time_limit",
    "cpu_time_limit": 3.0,
    "neval": 10000,

    "verbose": False,
    # "v_timer": -1
}

mut_params = ParamScheduler("Linear", {"method":"Gauss", "F": [0.001, 0.00001]})
mutation_op = OperatorReal("RandNoise", mut_params)

cross_op = OperatorReal("Multipoint")

de_params = ParamScheduler("Linear", {"F":[0.8, 0.9], "Cr":[0.8, 0.5]})
de_op = OperatorReal("DE/best/1", de_params)

parent_params = ParamScheduler("Linear", {"amount": [30, 15]})
parent_sel_op = ParentSelection("Best", parent_params)

selection_op = SurvivorSelection("(m+n)")

objfunc = Sphere(10, "min")

pop_init_single = UniformVectorInitializer(10, objfunc.low_lim, objfunc.up_lim, pop_size=1)
pop_init = UniformVectorInitializer(10, objfunc.low_lim, objfunc.up_lim, pop_size=100)

def test_hillclimb():
    search_strat = HillClimb(pop_init_single, mutation_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_localseach():
    search_strat = LocalSearch(pop_init_single, mutation_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_sa():
    search_strat = SA(pop_init_single, mutation_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_random():
    search_strat = RandomSearch(pop_init)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_nosearch():
    search_strat = NoSearch(pop_init)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()

def test_staticpop():
    search_strat = StaticPopulation(pop_init, mutation_op, parent_sel_op, selection_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_varpop():
    search_strat = VariablePopulation(pop_init, mutation_op, parent_sel_op, selection_op, {"offspringSize": 200})
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_es():
    search_strat = ES(pop_init, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": 200})
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_ga():
    search_strat = GA(pop_init, mutation_op, cross_op, parent_sel_op, selection_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_hs():
    search_strat = HS(pop_init, {"HMCR":0.8, "BW":0.5, "PAR":0.2})
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_de():
    search_strat = DE(pop_init, de_op)
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_pso():
    search_strat = PSO(pop_init, {"w":0.7, "c1":1.5, "c2":1.5})
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]

def test_cro():
    search_strat = CRO(pop_init, mutation_op, cross_op, {"rho":0.5, "Fb":0.75, "Fd":0.2, "Pd":0.7, "attempts":4})
    alg = GeneralSearch(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]