import numpy as np

from metaheuristic_designer import ObjectiveFromLambda, ParamScheduler
from metaheuristic_designer.searchMethods import GeneralSearch
from metaheuristic_designer.operators import OperatorReal
from metaheuristic_designer.algorithms import *


def func(vector):
    return np.sum(np.abs(vector))


def run_algorithm():
    params = {
        "stop_cond": "neval",
        "time_limit": 10.0,
        "cpu_time_limit": 10.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-30,
        "verbose": True,
        "v_timer": 0.5,
    }

    objfunc = ObjectiveFromLambda(func, 10, "min")

    mutation_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0001})

    search_strat = HillClimb(mutation_op)

    alg = GeneralSearch(objfunc, search_strat, params=params)

    ind, fit = alg.optimize()
    print(ind)
    alg.display_report()


def main():
    run_algorithm()


if __name__ == "__main__":
    main()
