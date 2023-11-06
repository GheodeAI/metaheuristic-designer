from metaheuristic_designer import ObjectiveFunc, ParamScheduler, simple
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm, AlgorithmSelection
from metaheuristic_designer.operators import OperatorReal, OperatorInt, OperatorBinary
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection
from metaheuristic_designer.strategies import *

from metaheuristic_designer.benchmarks import *

import argparse

from copy import copy
import scipy as sp
import numpy as np


def run_algorithm(save_report):
    params = {
        "stop_cond": "neval",
        # "neval": 1e4,
        "neval": 10,
        "encoding": "real",
        "verbose": False,
    }

    objfunc = Rastrigin(3, "min")

    # Define algorithms to be tested
    algorithms = [
        simple.hill_climb(objfunc, params),
        simple.simulated_annealing(objfunc, params),
        simple.evolution_strategy(objfunc, params),
        simple.differential_evolution(objfunc, params),
        simple.genetic_algorithm(objfunc, params),
        simple.particle_swarm(objfunc, params),
        simple.random_search(objfunc, params)
    ]

    algorithm_search = AlgorithmSelection(algorithms)

    solution, report = algorithm_search.optimize()
    print(f"solution: {solution}")
    print(report)
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save-report",
        dest="save_report",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    args = parser.parse_args()

    save_report = False

    if args.save_report:
        save_report = True

    run_algorithm(save_report=save_report)


if __name__ == "__main__":
    main()
