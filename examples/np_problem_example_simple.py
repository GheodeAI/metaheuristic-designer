# from metaheuristic_designer import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
# from metaheuristic_designer.SearchMethods import GeneralSearch, MemeticSearch
# from metaheuristic_designer.Operators import OperatorReal, OperatorInt, OperatorBinary
# from metaheuristic_designer.Initializers import UniformVectorInitializer
# from metaheuristic_designer.Algorithms import *

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
        # "stop_cond": "time_limit or fit_target",
        "stop_cond": "time_limit",
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
        "encoding": "bin",
        "mutstr": 1,
        # "encoding": "int",
        # "mutstr": 1,
        # Population based algorithms
        "pop_size": 100,
        # GA
        "pcross": 0.8,
        "pmut": 1,
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

    values = [
        360,
        83,
        59,
        130,
        431,
        67,
        230,
        52,
        93,
        125,
        670,
        892,
        600,
        38,
        48,
        147,
        78,
        256,
        63,
        17,
        120,
        164,
        432,
        35,
        92,
        110,
        22,
        42,
        50,
        323,
        514,
        28,
        87,
        73,
        78,
        15,
        26,
        78,
        210,
        36,
        85,
        189,
        274,
        43,
        33,
        10,
        19,
        389,
        276,
        312,
    ]

    weights = [
        7,
        0,
        30,
        22,
        80,
        94,
        11,
        81,
        70,
        64,
        59,
        18,
        0,
        36,
        3,
        8,
        15,
        42,
        9,
        0,
        42,
        47,
        52,
        32,
        26,
        48,
        55,
        6,
        29,
        84,
        2,
        4,
        18,
        56,
        7,
        29,
        93,
        44,
        71,
        3,
        86,
        66,
        31,
        65,
        0,
        79,
        20,
        65,
        52,
        13,
    ]

    capacity = 850

    objfunc = Bin_Knapsack_problem(weights, values, capacity)

    if alg_name == "HillClimb":
        alg = hill_climb(objfunc, params=params)
    elif alg_name == "SA":
        alg = simulated_annealing(objfunc, params=params)
    elif alg_name == "ES":
        alg = evolution_strategy(objfunc, params=params)
    elif alg_name == "GA":
        alg = genetic_algorithm(objfunc, params=params)
    elif alg_name == "DE":
        alg = differential_evolution(objfunc, params=params)
    elif alg_name == "PSO":
        alg = particle_swarm(objfunc, params=params)
    elif alg_name == "RandomSearch":
        alg = random_search(objfunc, params=params)
    else:
        raise ValueError(f'Error: Algorithm "{alg_name}" doesn\'t exist.')

    ind, fit = alg.optimize()
    print(ind)
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_pop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="alg", help="Specify an algorithm")
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

    algorithm_name = "ES"
    mem = False
    save_state = False

    if args.alg:
        algorithm_name = args.alg

    if args.mem:
        mem = True

    if args.save_state:
        save_state = True

    run_algorithm(alg_name=algorithm_name, memetic=mem, save_state=save_state)


if __name__ == "__main__":
    main()
