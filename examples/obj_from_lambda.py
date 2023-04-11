import numpy as np

from pyevolcomp import ObjectiveFromLambda, ParamScheduler
from pyevolcomp.SearchMethods import GeneralSearch
from pyevolcomp.Operators import OperatorReal
from pyevolcomp.Algorithms import *


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
        "v_timer": 0.5
    }

    objfunc = ObjectiveFromLambda(func, 10, "min")

    mutation_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.0001})

    search_strat = HillClimb(mutation_op)
    
    alg = GeneralSearch(search_strat, params)
    
    ind, fit = alg.optimize(objfunc)
    print(ind)
    alg.display_report(objfunc)


def main():
    run_algorithm()

if __name__ == "__main__":
    main()