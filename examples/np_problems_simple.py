import argparse
import numpy as np
import networkx as nx
import metaheuristic_designer as mhd
from metaheuristic_designer.benchmarks import BinKnapsack, ThreeSAT, MaxClique, TSP
from metaheuristic_designer import simple


def run_algorithm(alg_name, problem, ngen, seed):
    rng = mhd.check_random_state(seed)

    # ---- objective ----
    if problem == "knapsack":
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
        objfunc = BinKnapsack(weights, values, 850)
        encoding = "bin"
    elif problem == "3sat":
        objfunc = ThreeSAT.from_cnf_file("data/sat_examples/uf50-03.cnf")
        encoding = "bin"
    elif problem == "maxclique":
        g = nx.gnp_random_graph(100, 0.8)
        adj = nx.adjacency_matrix(g).todense()
        objfunc = MaxClique(adj)
        encoding = "perm"
    elif problem == "tsp":
        objfunc = TSP.from_csv("data/tsp_examples/r50_01.csv")
        encoding = "perm"
    else:
        raise ValueError(f"Unknown problem: {problem}")

    # Shared algorithm parameters
    algo_params = {
        "stop_cond": "max_iterations",
        "max_iterations": ngen,
        "reporter": "tqdm",
        "random_state": rng,
    }

    # Choose the wrapper family based on encoding type
    if encoding == "bin":
        alg_map = {
            "hillclimb": simple.hill_climb_binary(objfunc, **algo_params),
            "sa": simple.simulated_annealing_binary(objfunc, **algo_params),
            "es": simple.evolution_strategy_binary(objfunc, **algo_params),
            "ga": simple.genetic_algorithm_binary(objfunc, **algo_params),
            "de": simple.differential_evolution_binary(objfunc, **algo_params),
            "randomsearch": simple.random_search_binary(objfunc, **algo_params),
        }
    else:  # permutation
        alg_map = {
            "hillclimb": simple.hill_climb_permutation(objfunc, **algo_params),
            "sa": simple.simulated_annealing_permutation(objfunc, **algo_params),
            "es": simple.evolution_strategy_permutation(objfunc, **algo_params),
            "ga": simple.genetic_algorithm_permutation(objfunc, **algo_params),
            "randomsearch": simple.random_search_permutation(objfunc, **algo_params),
        }
    if alg_name not in alg_map:
        raise ValueError(f"Algorithm {alg_name} not available for {problem}")
    algo = alg_map[alg_name]

    # Optimise
    population = algo.optimize()
    solution, fitness = population.best_solution()
    print(f"\nBest fitness: {fitness}")
    print(f"Best solution (decoded): {solution}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="GA", type=str.lower, choices=["HillClimb", "SA", "ES", "GA", "DE", "RandomSearch"])
    parser.add_argument("-o", "--objective", default="knapsack", type=str.lower, choices=["knapsack", "3sat", "maxclique", "tsp"])
    parser.add_argument("--ngen", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_algorithm(args.algorithm, args.objective, args.ngen, args.seed)


if __name__ == "__main__":
    main()
