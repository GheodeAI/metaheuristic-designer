from metaheuristic_designer import ObjectiveFunc, ParamScheduler
from metaheuristic_designer.algorithms import StandardAlgorithm, MemeticAlgorithm
from metaheuristic_designer.encodings import TypeCastEncoding
from metaheuristic_designer.operators import (
    VectorOperator,
    PermOperator,
)
from metaheuristic_designer.initializers import (
    UniformInitializer,
    PermInitializer,
)
from metaheuristic_designer.strategies import *
from metaheuristic_designer.selection_methods import *
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

        encoding = TypeCastEncoding(int, bool)
        pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=100, dtype=int, encoding=encoding)

        parent_params = ParamScheduler("Linear", {"amount": 20})
        # select_params = ParamScheduler("Linear")

        mut_params = ParamScheduler("Linear", {"N": [10, 1]})
        mutation_op = VectorOperator("Flip", mut_params)

        cross_op = VectorOperator("Multipoint")

        op_list = [
            VectorOperator("Flip", {"N": 2}),
            VectorOperator("1point"),
            VectorOperator("Multipoint"),
            VectorOperator("MutSample", {"distrib": "Bernoulli", "p": 0.2, "N": 2}),
        ]

        neighborhood_structures = [VectorOperator("Flip", {"N": n}, name=f"Flip(n={n})") for n in range(objfunc.vecsize)]

    elif problem_name == "3SAT":
        objfunc = ThreeSAT.from_cnf_file("./data/sat_examples/uf50-03.cnf")

        encoding = TypeCastEncoding(int, bool)
        pop_initializer = UniformInitializer(objfunc.vecsize, 0, 1, pop_size=100, dtype=int, encoding=encoding)

        parent_params = ParamScheduler("Linear", {"amount": 20})
        # select_params = ParamScheduler("Linear")

        mut_params = ParamScheduler("Linear", {"N": [10, 1]})
        mutation_op = VectorOperator("Flip", mut_params)

        cross_op = VectorOperator("Multipoint")

        op_list = [
            VectorOperator("Flip", {"N": 2}),
            VectorOperator("1point"),
            VectorOperator("Multipoint"),
            VectorOperator("MutSample", {"distrib": "Bernoulli", "p": 0.2, "N": 2}),
        ]

        neighborhood_structures = [VectorOperator("Flip", {"N": n}, name=f"Flip(n={n})") for n in range(objfunc.vecsize)]
    elif problem_name == "MaxClique":
        g = nx.gnp_random_graph(100, 0.8)
        adj_mat = nx.adjacency_matrix(g).todense()
        objfunc = MaxClique(adj_mat)

        print("Adjacenty matrix:")
        print(adj_mat)
        print()

        pop_initializer = PermInitializer(objfunc.vecsize, pop_size=100)

        parent_params = ParamScheduler("Linear", {"amount": 20})

        mut_params = ParamScheduler("Linear", {"N": [10, 1]})
        mutation_op = PermOperator("Perm", mut_params)

        cross_op = PermOperator("OrderCross")

        op_list = [
            PermOperator("Swap", mut_params),
            PermOperator("Insert", mut_params),
            PermOperator("Invert", mut_params),
            PermOperator("PMX"),
        ]

        neighborhood_structures = [PermOperator("Swap", {"N": n}, name=f"Swap(n={n})") for n in range(objfunc.vecsize)]
    else:
        raise ValueError(f"The problem '{problem_name}' does not exist.")

    parent_sel_op = ParentSelection("Best", parent_params)
    selection_op = SurvivorSelection("(m+n)")

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = VectorOperator("RandNoise", {"distrib": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 10})

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, params={"iters": 20})
    elif alg_name == "SA":
        pop_initializer.pop_size = 1
        search_strat = SA(
            pop_initializer,
            mutation_op,
            params={"iter": 100, "temp_init": 1, "alpha": 0.99},
        )
    elif alg_name == "ES":
        search_strat = ES(
            initializer=pop_initializer,
            mutation_op=mutation_op,
            cross_op=cross_op,
            parent_sel=parent_sel_op,
            survivor_sel=selection_op,
            params={"offspringSize": 150},
        )
    elif alg_name == "GA":
        search_strat = GA(
            initializer=pop_initializer,
            mutation_op=mutation_op,
            cross_op=cross_op,
            parent_sel=parent_sel_op,
            survivor_sel=selection_op,
            params={"pcross": 0.8, "pmut": 0.2},
        )
    elif alg_name == "DE":
        search_strat = DE(pop_initializer, OperatorVector("DE/best/1", params={"F": 0.8, "Cr": 0.8}))
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, params={"w": 0.7, "c1": 1.5, "c2": 1.5})
    elif alg_name == "BernoulliUMDA":
        search_strat = BernoulliUMDA(pop_initializer, parent_sel_op, selection_op, params={"p": 0.1, "noise": 5e-3})
    elif alg_name == "BernoulliPBIL":
        search_strat = BernoulliPBIL(pop_initializer, parent_sel_op, selection_op, params={"p": 0.1, "lr": 0.25, "noise": 5e-3})
    elif alg_name == "RVNS":
        search_strat = RVNS(pop_initializer, neighborhood_structures)
    elif alg_name == "VND":
        search_strat = VND(pop_initializer, neighborhood_structures)
    elif alg_name == "VNS":
        local_search = LocalSearch(pop_initializer, params={"iters": 100})
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 100, "nchange": "seq"},
        )
        # local_search = HillClimb(pop_initializer)
        # search_strat = VNS(pop_initializer, neighborhood_structures, local_search, params={"iters": 500})
    elif alg_name == "GVNS":
        local_search = VND(pop_initializer, neighborhood_structures, params={"nchange": "cyclic"})
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 100, "nchange": "seq"},
        )
    elif alg_name == "RandomSearch":
        search_strat = RandomSearch(pop_initializer)
    elif alg_name == "NoSearch":
        search_strat = NoSearch(pop_initializer)
    else:
        print(f'Error: Algorithm "{alg_name}" doesn\'t exist.')
        exit()

    if memetic:
        alg = MemeticAlgorithm(objfunc, search_strat, local_search, mem_select, params=params)
    else:
        alg = StandardAlgorithm(objfunc, search_strat, params=params)

    result = alg.optimize()
    ind, _ = result.best_solution(decoded=True)
    print(ind)
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_population=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="alg", help="Specify an algorithm", default="GA")
    parser.add_argument("-o", "--objective", dest="objective", help="Specify an problem to solve", default="Knapsack")
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
        problem_name=args.objective,
        memetic=args.mem,
        save_state=args.save_state,
    )


if __name__ == "__main__":
    main()
