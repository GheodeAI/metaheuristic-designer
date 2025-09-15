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
import networkx as nx


def run_algorithm(alg_name, problem_name, memetic, save_state):
    params = {
        # "stop_cond": "neval or time_limit or fit_target",
        # "stop_cond": "neval or time_limit",
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
    }

    if problem_name == "Knapsack":
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

        objfunc = BinKnapsack(weights, values, capacity)
        params["encoding"] = "bin"
    elif problem_name == "3SAT":
        objfunc = ThreeSAT.from_cnf_file("./data/sat_examples/uf50-03.cnf")
        params["encoding"] = "bin"
    elif problem_name == "MaxClique":
        params["encoding"] = "perm"
        if alg_name in ["DE", "PSO"]:
            raise ValueError(f"{alg_name} algorithm not supported for permutation-encoded problems.")
        g = nx.gnp_random_graph(100, 0.8)
        adj_mat = nx.adjacency_matrix(g).todense()
        objfunc = MaxClique(adj_mat)
    else:
        raise ValueError(f"The problem '{problem_name}' does not exist.")

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

    result = alg.optimize()
    ind, _ = result.best_solution()
    print(ind)
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_population=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="alg", help="Specify an algorithm", default="GA")
    parser.add_argument("-p", "--problem", dest="prob", help="Specify an problem to solve", default="Knapsack")
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

    run_algorithm(
        alg_name=args.alg,
        problem_name=args.prob,
        memetic=args.mem,
        save_state=args.save_state,
    )


if __name__ == "__main__":
    main()
