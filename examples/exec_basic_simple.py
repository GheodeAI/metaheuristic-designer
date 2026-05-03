import metaheuristic_designer as mhd
from metaheuristic_designer.simple import hill_climb
from metaheuristic_designer.simple import genetic_algorithm
from metaheuristic_designer.simple import evolution_strategy
from metaheuristic_designer.simple import particle_swarm
from metaheuristic_designer.simple import differential_evolution
from metaheuristic_designer.simple import random_search
from metaheuristic_designer.simple import simulated_annealing
from metaheuristic_designer.simple import bayesian_optimization
from metaheuristic_designer.benchmarks import *

import argparse

from copy import copy
import scipy as sp
import numpy as np


def run_algorithm(alg_name, memetic, save_state, objective, dim):
    match objective:
        case "Sphere":
            objfunc = Sphere(dim, "min")
        case "Rastrigin":
            objfunc = Rastrigin(dim, "min")
        case "Rosenbrock":
            objfunc = Rosenbrock(dim, "min")
        case "Weierstrass":
            objfunc = Weierstrass(dim, "min")
        case _:
            raise Exception(f'Objective function "{objective}" doesn\'t exist.')

    params = {
        # General algorithm params
        "stop_cond": "convergence or time_limit or fit_target",
        # "stop_cond": "time_limit or fit_target",
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
        "min": objfunc.low_lim,
        "max": objfunc.up_lim,
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
        # BO
        "batch_size": 50,
        "max_samples": 100,
    }

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
    elif alg_name == "BO":
        alg = bayesian_optimization(params=params, objfunc=objfunc)
    elif alg_name == "RandomSearch":
        alg = random_search(params=params, objfunc=objfunc)
    else:
        raise ValueError(f'Error: Algorithm "{alg_name}" doesn\'t exist.')

    population = alg.optimize()
    print(population.best_solution()[0])
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
    parser.add_argument("-o", "--objective", dest="objective", help="Name of the objective function.", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    args = parser.parse_args()

    run_algorithm(alg_name=args.algorithm.upper(), memetic=args.mem, save_state=args.save_state, objective=args.objective, dim=args.dim)


if __name__ == "__main__":
    main()
