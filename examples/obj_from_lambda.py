import numpy as np

from metaheuristic_designer import ObjectiveFromLambda, ParamScheduler
from metaheuristic_designer.algorithms import GeneralAlgorithm
from metaheuristic_designer.operators import OperatorVector
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.strategies import *


def func(vector):
    return np.sum(np.abs(vector))


def run_algorithm():
    params = {
        "stop_cond": "time_limit",
        "time_limit": 10.0,
        "cpu_time_limit": 10.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-30,
        "verbose": True,
        "v_timer": 0.5,
    }

    objfunc = ObjectiveFromLambda(func, 10, "min")

    pop_init = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1)

    mutation_op = OperatorVector("RandNoise", {"distrib": "Cauchy", "F": 0.0001})

    search_strat = HillClimb(pop_init, mutation_op)

    alg = GeneralAlgorithm(objfunc, search_strat, params=params)

    population = alg.optimize()
    print(population.best_solution())
    alg.display_report()


if __name__ == "__main__":
    run_algorithm()
