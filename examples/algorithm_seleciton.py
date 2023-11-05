from metaheuristic_designer import ObjectiveFunc, ParamScheduler
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
        "progress_metric": "neval",
        "neval": 3e5,
        "verbose": False,
    }

    objfunc = Sphere(3, "min")

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)
    single_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1)

    mutation_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0001})
    cross_op = OperatorReal("Multipoint")

    parent_sel_op = ParentSelection("Best", {"amount": 50})
    selection_op = SurvivorSelection("(m+n)")

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 10})

    # Define algorithms to be tested
    algorithms = [
        GeneralAlgorithm(objfunc, HillClimb(single_initializer, mutation_op), params=params),
        GeneralAlgorithm(objfunc, LocalSearch(single_initializer, mutation_op), params=params),
        GeneralAlgorithm(objfunc, SA(pop_initializer, mutation_op, {"iter": 100, "temp_init": 1, "alpha": 0.997}), params=params),
        GeneralAlgorithm(
            objfunc,
            ES(
                pop_initializer,
                mutation_op,
                OperatorReal("Nothing"),
                parent_sel_op,
                selection_op,
                {"offspringSize": 150},
            ),
            params=params,
        ),
        GeneralAlgorithm(
            objfunc,
            GA(
                pop_initializer,
                mutation_op,
                cross_op,
                parent_sel_op,
                selection_op,
                {"pcross": 0.8, "pmut": 0.2},
            ),
            params=params,
        ),
        MemeticAlgorithm(
            objfunc,
            GA(
                pop_initializer,
                mutation_op,
                cross_op,
                parent_sel_op,
                selection_op,
                {"pcross": 0.8, "pmut": 0.2},
            ),
            local_search,
            mem_select,
            params=params,
        ),
        GeneralAlgorithm(
            objfunc,
            GA(
                pop_initializer,
                mutation_op,
                cross_op,
                parent_sel_op,
                selection_op,
                {"pcross": 0.8, "pmut": 0.1},
            ),
            params=params,
        ),
        MemeticAlgorithm(
            objfunc,
            GA(
                pop_initializer,
                mutation_op,
                cross_op,
                parent_sel_op,
                selection_op,
                {"pcross": 0.8, "pmut": 0.1},
            ),
            local_search,
            mem_select,
            params=params,
        ),
        GeneralAlgorithm(objfunc, PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5}), params=params),
        MemeticAlgorithm(objfunc, PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5}), local_search, mem_select, params=params),
        GeneralAlgorithm(objfunc, RandomSearch(pop_initializer), params=params),
    ]

    alg = AlgorithmSelection(algorithms)

    report = alg.optimize()
    print(report)
    # alg.display_report(show_plots=True)


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
