# from metaheuristic_designer import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
# from metaheuristic_designer.SearchMethods import GeneralSearch, MemeticSearch
# from metaheuristic_designer.Operators import OperatorReal, OperatorInt, OperatorBinary
# from metaheuristic_designer.Initializers import UniformVectorInitializer
# from metaheuristic_designer.Algorithms import *

import metaheuristic_designer as mhd
from metaheuristic_designer.simple import hill_climb
from metaheuristic_designer.simple import genetic_algorithm
from metaheuristic_designer.simple import evolution_strategy
from metaheuristic_designer.simple import particle_swarm
from metaheuristic_designer.simple import differential_evolution
from metaheuristic_designer.simple import random_search
from metaheuristic_designer.simple import simulated_annealing
from metaheuristic_designer.benchmarks import *

import argparse

from copy import copy
import scipy as sp
import numpy as np


def run_algorithm(alg_name, memetic, save_state):
    params = {
        # General algorithm params
        # "stop_cond": "convergence or time_limit or fit_target",
        "stop_cond": "time_limit or fit_target",
        "progress_metric": "time_limit",
        "time_limit": 100.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 200,
        "verbose": True,
        "v_timer": 0.5,
        # General
        "encoding": "real",
        "mutstr": 1e-3,
        # Population based algorithms
        "pop_size": 100,
        # GA
        "pcross": 0.8,
        "pmut": 0.1,
        "cross": "multipoint",
        # ES
        "offspring_size": 150,
        # PSO
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.5,
        # DE
        "F": 0.7,
        "Cr": 0.9,
        # SA
        "iter": 100,
        "temp_init": 1,
        "alpha": 0.997,
    }

    objfunc = Sphere(30, "min")
    # objfunc = Rosenbrock(30, "min")
    # objfunc = Rastrigin(10, "min")
    # objfunc = Weierstrass(30, "min")

    if alg_name == "HillClimb":
        alg = hill_climb(params=params, objfunc=objfunc)
    elif alg_name == "SA":
        alg = simulated_annealing(params=params, objfunc=objfunc)
    elif alg_name == "ES":
        alg = evolution_strategy(params=params, objfunc=objfunc)
    elif alg_name == "GA":
        alg = genetic_algorithm(params=params, objfunc=objfunc)
    elif alg_name == "DE":
        alg = differential_evolution(params=params, objfunc=objfunc)
    elif alg_name == "PSO":
        alg = particle_swarm(params=params, objfunc=objfunc)
    elif alg_name == "RandomSearch":
        alg = random_search(params=params, objfunc=objfunc)
    else:
        raise ValueError(f'Error: Algorithm "{alg_name}" doesn\'t exist.')

    ind, fit = alg.optimize()
    print(ind)
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_pop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm", help="Specify an algorithm", default="ES")
    parser.add_argument(
        "-m",
        "--memetic",
        dest="mem",
        action="store_true",
        help="Does local search after mutation",
    )
    parser.add_argument(
        "-s",
        "--save-state",
        dest="save_state",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    args = parser.parse_args()

    run_algorithm(alg_name=args.algorithm, memetic=args.mem, save_state=args.save_state)


if __name__ == "__main__":
    main()
