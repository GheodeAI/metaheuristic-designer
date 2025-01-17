from metaheuristic_designer import ObjectiveFunc, ParamScheduler, simple
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm, AlgorithmSelection
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
        "neval": 1e4,
        "encoding": "real",
        "verbose": False,
    }

    objfunc = HappyCat(3, "min")

    # Define algorithms to be tested
    algorithms = [
        simple.hill_climb(params, objfunc),
        simple.simulated_annealing(params, objfunc),
        simple.evolution_strategy(params, objfunc),
        simple.differential_evolution(params, objfunc),
        simple.genetic_algorithm(params, objfunc),
        simple.particle_swarm(params, objfunc),
        simple.random_search(params, objfunc),
    ]

    algorithm_search = AlgorithmSelection(algorithms)

    solution, best_fitness, report = algorithm_search.optimize()
    print(f"solution: {solution}")
    print(f"with fitness: {best_fitness}")
    print(report)
    if save_report:
        report.to_csv("./examples/results/algorithm_selection_report.csv")


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
