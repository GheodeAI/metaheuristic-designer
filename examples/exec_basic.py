import argparse
import logging
from pathlib import Path

<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np

=======
>>>>>>> devel
from metaheuristic_designer.algorithms import Algorithm, MemeticAlgorithm
=======
from metaheuristic_designer.algorithms import Algorithm
>>>>>>> bc7b8c7 (moved algorithm steps inside search strategy)
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.strategies.classic import CMA_ES
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.strategies import (
    HillClimb,
    LocalSearch,
    SA,
    ES,
    GA,
    DE,
    PSO,
    GaussianUMDA,
    GaussianPBIL,
    CrossEntropyMethod,
    BayesianOptimization,
    RandomSearch,
    NoSearch,
)
from metaheuristic_designer.benchmarks import Sphere, Rastrigin, Rosenbrock, Weierstrass
from metaheuristic_designer.utils import check_random_state

available_objectives = ("sphere", "rastrigin", "rosenbrock", "weierstrass")
available_algorithms = ("hillclimb", "localsearch", "sa", "es", "ga", "de", "gaussianumda", "gaussianpbil", "crossentropy", "randomsearch")


def run_algorithm(alg_name, memetic, save_state, show_plots, objective, dim, reporter, random_state):
    algorithm_params = {
        "stop_cond": "convergence or real_time_limit",
        "progress_metric": "real_time_limit",
        "real_time_limit": 2.0,
        "max_patience": 500,
    }

    functions_map = {
        "sphere": Sphere(dim, mode="min"),
        "rastrigin": Rastrigin(dim, mode="min"),
        "rosenbrock": Rosenbrock(dim, mode="min"),
        "weierstrass": Weierstrass(dim, mode="min"),
    }
    if objective not in functions_map:
        raise Exception(f'Objective function "{objective}" doesn\'t exist.')
    objfunc = functions_map[objective]

    search_strategy_map = {
        "hillclimb": HillClimb(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-2, N=1, random_state=random_state),
            random_state=random_state
        ),
        "localsearch": LocalSearch(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            iterations=20,
            random_state=random_state
        ),
        "sa": SA(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
            random_state=random_state
        ),
        "es": ES(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            crossover_op=create_operator("crossover.uniform", random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
            random_state=random_state
        ),
        "ga": GA(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            crossover_op=create_operator("crossover.uniform", random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=50, random_state=random_state),
            survivor_sel=create_survivor_selection("Elitism", amount=20, random_state=random_state),
            mutation_prob=0.2,
            crossover_prob=0.8,
            random_state=random_state,
        ),
        "de": DE(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            de_operator_name="DE/best/1",
            F=0.8,
            Cr=0.8,
            random_state=random_state,
        ),
        "pso": PSO(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            w=0.7,
            c1=1.5,
            c2=1.5,
            random_state=random_state,
        ),
        "cmaes": CMA_ES(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            offspring_size=200,
            random_state=random_state,
        ),
        "gaussianumda": GaussianUMDA(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            scale=0.1,
            noise=1e-3,
            random_state=random_state,
        ),
        "gaussianpbil": GaussianPBIL(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            scale=0.1,
            lr=0.3,
            noise=1e-3,
            random_state=random_state,
        ),
        "crossentropy": CrossEntropyMethod(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, random_state=random_state),
            random_state=random_state,
        ),
        "bayesianoptimizaiton": BayesianOptimization(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            batch_size=50,
            max_samples=100,
            random_state=random_state,
        ),
        "randomsearch": RandomSearch(UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state), random_state=random_state),
        "nosearch": NoSearch(UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state), random_state=random_state),
    }
    if alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    else:
        search_strategy = search_strategy_map[alg_name]

    if memetic:
        local_search = LocalSearch(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=search_strategy.initializer.pop_size),
            operator=create_operator("mutation.gaussian_noise", F=2e-4),
            params={"iters": 20},
        )
        # alg = MemeticStrategy(
        #     search_strategy,
        #     local_search,
        #     lamarckian=True
        #     **algorithm_params,
        # )

    alg = Algorithm(objfunc, search_strategy, reporter=reporter, **algorithm_params)

    population = alg.optimize()
    best_solution, best_objective = population.best_solution()
    print()
    print(f"Solution: {[float(i) for i in best_solution]}")
    print(f"Objective value: {best_objective}")

    if save_state:
        script_dir = Path(__file__).parent.absolute()
        result_dir = script_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        alg.store_state(result_dir / "test.json", readable=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--algorithm", dest="algorithm", help=f"Specify an algorithm. Available options are {available_algorithms}.", default="ES"
    )
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
    parser.add_argument(
        "-o", "--objective", dest="objective", help=f"Name of the objective function. Available options are {available_objectives}", default="Sphere"
    )
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=None, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("-v", "--reporter", default="tqdm", help="Reporter to use for progress tracking. Avaliable options are")
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    args = parser.parse_args()

    rng = check_random_state(args.seed)
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.lower(),
        memetic=args.mem,
        save_state=args.save_state,
        show_plots=args.plot,
        objective=args.objective.lower(),
        dim=args.dim,
        reporter=args.reporter,
        random_state=rng,
    )


if __name__ == "__main__":
    main()
