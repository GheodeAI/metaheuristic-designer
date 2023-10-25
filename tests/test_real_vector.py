import pytest

import os

import numpy as np

from metaheuristic_designer import ObjectiveFunc, ParamScheduler
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import OperatorReal
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import Sphere

import metaheuristic_designer as mhd

mhd.reset_seed(0)

test_params = {
    "stop_cond": "neval or cpu_time_limit",
    "cpu_time_limit": 2.0,
    "neval": 10000,
    "verbose": False,
    "v_timer": -1,
}

mut_params = ParamScheduler("Linear", {"method": "Gauss", "F": [0.001, 0.00001]})
mutation_op = OperatorReal("RandNoise", mut_params)

cross_op = OperatorReal("Multipoint")

de_params = ParamScheduler("Linear", {"F": [0.8, 0.9], "Cr": [0.8, 0.5]})
de_op = OperatorReal("DE/best/1", de_params)

parent_params = ParamScheduler("Linear", {"amount": [30, 15]})
parent_sel_op = ParentSelection("Best", parent_params)

selection_op = SurvivorSelection("(m+n)")

objfunc = Sphere(10, "min")

pop_init_single = UniformVectorInitializer(
    10, objfunc.low_lim, objfunc.up_lim, pop_size=1
)
pop_init = UniformVectorInitializer(10, objfunc.low_lim, objfunc.up_lim, pop_size=100)


def test_hillclimb_empty():
    search_strat = HillClimb(pop_init_single)


def test_hillclimb():
    search_strat = HillClimb(pop_init_single, mutation_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == 1


def test_localseach_empty():
    search_strat = LocalSearch(pop_init_single)


def test_localseach():
    search_strat = LocalSearch(pop_init_single, mutation_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == 1


def test_sa():
    search_strat = SA(pop_init_single, mutation_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == 1


def test_random():
    search_strat = RandomSearch(pop_init)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_nosearch():
    search_strat = NoSearch(pop_init)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert search_strat.pop_size == pop_init.pop_size


def test_staticpop():
    search_strat = StaticPopulation(pop_init, mutation_op, parent_sel_op, selection_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_varpop():
    search_strat = VariablePopulation(
        pop_init, mutation_op, parent_sel_op, selection_op, 200
    )
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_es():
    search_strat = ES(
        pop_init,
        mutation_op,
        cross_op,
        parent_sel_op,
        selection_op,
        {"offspringSize": 200},
    )
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_ga():
    search_strat = GA(pop_init, mutation_op, cross_op, parent_sel_op, selection_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_hs():
    search_strat = HS(pop_init, {"HMCR": 0.8, "BW": 0.5, "PAR": 0.2})
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_de():
    search_strat = DE(pop_init, de_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_pso():
    search_strat = PSO(pop_init, {"w": 0.7, "c1": 1.5, "c2": 1.5})
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size == pop_init.pop_size


def test_cro():
    search_strat = CRO(
        pop_init,
        mutation_op,
        cross_op,
        {"rho": 0.5, "Fb": 0.75, "Fd": 0.2, "Pd": 0.7, "attempts": 4},
    )
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size <= pop_init.pop_size


def test_cro_sl():
    search_strat = CRO_SL(
        pop_init,
        [mutation_op, cross_op],
        {"rho": 0.5, "Fb": 0.75, "Fd": 0.2, "Pd": 0.7, "attempts": 4},
    )
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size <= pop_init.pop_size


def test_pcro_sl():
    search_strat = PCRO_SL(
        pop_init,
        [mutation_op, cross_op],
        {"rho": 0.5, "Fb": 0.75, "Fd": 0.2, "Pd": 0.7, "attempts": 4},
    )
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size <= pop_init.pop_size


@pytest.mark.parametrize("dyn_method", ["success", "fitness", "diff"])
@pytest.mark.parametrize("dyn_metric", ["best", "avg", "med", "worse"])
def test_dpcro_sl(dyn_method, dyn_metric):
    search_strat_params = {
        "rho": 0.6,
        "Fb": 0.95,
        "Fd": 0.1,
        "Pd": 0.9,
        "attempts": 3,
        "group_subs": True,
        "dyn_method": dyn_method,
        "dyn_metric": dyn_metric,
        "dyn_steps": 75,
        "prob_amp": 0.1,
    }
    search_strat = DPCRO_SL(pop_init, [mutation_op, cross_op], search_strat_params)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size <= pop_init.pop_size


def test_memetic():
    search_strat = GA(pop_init, mutation_op, cross_op, parent_sel_op, selection_op)
    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_init, neihbourhood_op, params={"iters": 10})
    alg = MemeticAlgorithm(
        objfunc, search_strat, local_search, mem_select, params=test_params
    )
    ind, fit = alg.optimize()
    assert alg.fit_history[0] > alg.fit_history[-1]
    assert search_strat.pop_size <= pop_init.pop_size


def test_reporting():
    test_params["verbose"] = True
    search_strat = GA(pop_init, mutation_op, cross_op, parent_sel_op, selection_op)
    alg = GeneralAlgorithm(objfunc, search_strat, params=test_params)
    ind, fit = alg.optimize()
    test_params["verbose"] = False

    alg.store_state("temp_pytest.json", True, True, True, True, True, True)
    os.remove("temp_pytest.json")


def test_reporting_memetic():
    test_params["verbose"] = True
    search_strat = GA(pop_init, mutation_op, cross_op, parent_sel_op, selection_op)
    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_init, neihbourhood_op, params={"iters": 10})
    alg = MemeticAlgorithm(
        objfunc, search_strat, local_search, mem_select, params=test_params
    )
    ind, fit = alg.optimize()
    test_params["verbose"] = False

    alg.store_state("temp_pytest.json", True, True, True, True, True, True)
    os.remove("temp_pytest.json")
