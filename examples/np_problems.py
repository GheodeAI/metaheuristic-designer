import argparse
import logging

import numpy as np
import networkx as nx

from metaheuristic_designer.algorithms import Algorithm
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import UniformInitializer, PermInitializer
from metaheuristic_designer.encodings import TypeCastEncoding
from metaheuristic_designer.strategies import *
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.benchmarks import *
from metaheuristic_designer.utils import check_random_state


def run_algorithm(alg_name, problem_name, memetic, save_state, reporter, random_state):
    # Common algorithm parameters
    algorithm_params = {
        "stop_condition_str": "real_time_limit",
        "real_time_limit": 100.0,
        "verbose_timer": 0.5,
        "reporter": reporter,
    }

    if problem_name == "knapsack":
        # fmt: off
        values = [360,83,59,130,431,67,230,52,93,125,670,892,600,38,48,147,78,256,63,17,120,164,432,35,92,110,22,42,50,323,514,28,87,73,78,15,26,78,210,36,85,189,274,43,33,10,19,389,276,312]
        weights = [7,0,30,22,80,94,11,81,70,64,59,18,0,36,3,8,15,42,9,0,42,47,52,32,26,48,55,6,29,84,2,4,18,56,7,29,93,44,71,3,86,66,31,65,0,79,20,65,52,13]
        # fmt: on
        capacity = 850
        objfunc = BinKnapsack(weights, values, capacity)
        encoding = TypeCastEncoding(int, bool)
        pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=100, dtype=int, encoding=encoding, random_state=random_state)

    elif problem_name == "3sat":
        objfunc = ThreeSAT.from_cnf_file("./data/sat_examples/uf50-03.cnf")
        encoding = TypeCastEncoding(int, bool)
        pop_initializer = UniformInitializer(objfunc.dimension, 0, 1, population_size=100, dtype=int, encoding=encoding, random_state=random_state)

    elif problem_name == "maxclique":
        g = nx.gnp_random_graph(100, 0.8)
        adj_mat = nx.adjacency_matrix(g).todense()
        objfunc = MaxClique(adj_mat)
        pop_initializer = PermInitializer(objfunc.dimension, population_size=100, random_state=random_state)
        encoding = TypeCastEncoding(float, int)  # not used for permutations, but kept for consistency
    elif problem_name == "tsp":
        objfunc = TSP.from_csv("data/tsp_examples/r20_02.csv")
        pop_initializer = PermInitializer(objfunc.dimension, population_size=100, random_state=random_state)
        encoding = TypeCastEncoding(float, int)  # not used for permutations, but kept for consistency
    else:
        raise ValueError(f"The problem '{problem_name}' does not exist.")

    if problem_name in ("knapsack", "3sat"):
        mutation_op = create_operator("mutation.bitflip", N=2, random_state=random_state)
        cross_op = create_operator("crossover.1point", random_state=random_state)
    elif problem_name in ("maxclique", "tsp"):
        mutation_op = create_operator("permutation.swap", N=2, random_state=random_state)
        cross_op = create_operator("permutation.pmx", random_state=random_state)

    search_strategy_map = {
        "hillclimb": HillClimb(
            initializer=pop_initializer,
            operator=mutation_op,
        ),
        "localsearch": LocalSearch(
            initializer=pop_initializer,
            operator=mutation_op,
            iterations=20,
        ),
        "sa": SA(
            initializer=pop_initializer,
            operator=mutation_op,
            iterations=100,
            temperature_init=1,
            alpha=0.99,
        ),
        "es": ES(
            initializer=pop_initializer,
            mutation_op=mutation_op,
            crossover_op=cross_op,
            parent_sel=create_parent_selection("best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
        ),
        "ga": GA(
            initializer=pop_initializer,
            mutation_op=mutation_op,
            crossover_op=cross_op,
            parent_sel=create_parent_selection("best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            mutation_prob=0.2,
            crossover_prob=0.8,
            random_state=random_state,
        ),
        "de": DE(
            de_operator_name="DE/best/1",
            initializer=pop_initializer,
            F=0.8,
            Cr=0.8,
        ),
        "bernoulliumda": BernoulliUMDA(
            initializer=pop_initializer,
            parent_sel=create_parent_selection("best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            p=np.full(objfunc.dimension, 0.1),
            noise=5e-3,
            random_state=random_state,
        ),
        "bernoullipbil": BernoulliPBIL(
            initializer=pop_initializer,
            parent_sel=create_parent_selection("best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            p=np.full(objfunc.dimension, 0.1),
            lr=0.1,
            noise=0.2,
            random_state=random_state,
        ),
        "randomsearch": RandomSearch(pop_initializer),
        "nosearch": NoSearch(pop_initializer),
    }

    if alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    search_strategy = search_strategy_map[alg_name]

    # ---- Memetic branch ----
    if memetic:
        local_search = LocalSearch(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=search_strategy.initializer.population_size
            ),
            operator=create_operator("mutation.gaussian_noise", F=1e-2, random_state=random_state),
            params={"iters": 20},
        )
        search_strategy = MemeticStrategy(
            main_strategy=search_strategy,
            local_search_heuristic=local_search,
            local_search_depth=10,
            local_search_frequency=5,
            improvement_selection=create_parent_selection("best", amount=10),
            keep_improved_solutions=True,
            random_state=random_state,
        )

    alg = Algorithm(objfunc, search_strategy, **algorithm_params)

    population = alg.optimize()
    best_solution, best_objective = population.best_solution()
    print("Best solution:", best_solution.astype(int))
    print("Best objective:", best_objective)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="GA", help="Algorithm name.")
    parser.add_argument("-o", "--objective", default="Knapsack", help="Problem name (Knapsack, 3SAT, MaxClique).")
    parser.add_argument("-m", "--memetic", action="store_true", help="Apply memetic wrapper.")
    parser.add_argument("-s", "--save-state", dest="save_state", action="store_true", help="Save algorithm state.")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log", default="WARNING", help="Log level.")
    parser.add_argument("-v", "--reporter", default="tqdm", help="Reporter to use for progress tracking.")
    args = parser.parse_args()

    # Set up logging and random state
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())
    rng = check_random_state(args.seed)

    run_algorithm(
        alg_name=args.algorithm.lower(),
        problem_name=args.objective.lower(),
        memetic=args.memetic,
        save_state=args.save_state,
        reporter=args.reporter,
        random_state=rng,
    )


if __name__ == "__main__":
    main()
