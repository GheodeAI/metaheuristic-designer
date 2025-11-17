from metaheuristic_designer import (
    ObjectiveFunc,
    ParamScheduler,
    InitializerFromLambda,
    ExtendedEncoding,
    ExtendedInitializer,
    NullOperator
)
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import VectorOperator, AdaptativeOperator
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.selection_methods import ParentSelection, SurvivorSelection
from metaheuristic_designer.strategies import *

from metaheuristic_designer.benchmarks import *

import argparse

from copy import copy
import scipy as sp
import numpy as np

def run_algorithm(alg_name, memetic, save_state):
    params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 100.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 10,
        "verbose": True,
        "v_timer": 0.5,
    }

    # objfunc = Sphere(30, "min")
    objfunc = Rastrigin(2, "min")
    # objfunc = Rosenbrock(2, "min")
    # objfunc = Ackley(30, "min")
    # objfunc = Weierstrass(30, "min")
    # objfunc = HappyCat(3, "min")

    adaption_encoding = ExtendedEncoding(objfunc.vecsize, param_sizes=(("F", 1),))

    pop_initializer = ExtendedInitializer(
        solution_init=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=adaption_encoding),
        param_init_dict={"F": InitializerFromLambda(lambda: sp.stats.expon(scale=0.01).rvs(size=1), pop_size=5, encoding=adaption_encoding)},
        encoding=adaption_encoding
    )

    ada_mutation_op = AdaptativeOperator(
        base_operator=VectorOperator("MutNoise", {"distrib": "Gauss", "N": 1}),
        param_operators={"F": VectorOperator("Mutate1Sigma", {"tau": 1 / np.sqrt(objfunc.vecsize), "epsilon": 1e-7})},
        encoding=adaption_encoding
    )

    cross_op = VectorOperator("Multipoint")

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(pop_initializer, ada_mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": 250}, name="Adaptative-ES")

    alg = GeneralAlgorithm(objfunc, search_strat, params=params)

    result = alg.optimize()
    ind, best_fitness = result.best_solution(decoded=True)
    print(f"solution vector: {ind}")
    print(f"params: {adaption_encoding.decode_params(result.genotype_matrix)}")
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_population=True)


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
