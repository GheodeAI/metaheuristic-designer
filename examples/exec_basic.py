import argparse
import logging

import numpy as np

# from metaheuristic_designer import *
from metaheuristic_designer.algorithms import StandardAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import create_operator
from metaheuristic_designer.initializers import UniformInitializer, ExtendedInitializer
from metaheuristic_designer.parent_selection import create_parent_selection
from metaheuristic_designer.survivor_selection import create_survivor_selection
from metaheuristic_designer.encodings import PSOEncoding
from metaheuristic_designer.constraint_handlers import BounceBoundConstraint, ExtendedConstraintHandler
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
    NoSearch
)
from metaheuristic_designer.benchmarks import (
    Sphere,
    Rastrigin,
    Rosenbrock,
    Weierstrass
)
from metaheuristic_designer.utils import check_random_state

available_objectives = {"sphere", "rastrigin", "rosenbrock", "weierstrass"}
available_algorithms = {"hillclimb", "localsearch", "sa", "es", "ga", "de", "gaussianumda", "gaussianpbil", "crossentropy", "randomsearch"}

def run_algorithm(alg_name, memetic, save_state, show_plots, objective, dim, random_state):
    algorithm_params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 120.0,
        "cpu_time_limit": 100.0,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 500,
        "verbose_timer": 0.5,
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
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-2, N=1, random_state=random_state),
        ),
        "localsearch": LocalSearch(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", f=1e-3, N=1, random_state=random_state),
            iterations=20,
        ),
        "sa": SA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", f=1e-3, N=1, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
        ),
        "es": ES(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", f=1e-3, N=1, random_state=random_state),
            cross_op=create_operator("crossover.uniform", random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
        ),
        "ga": GA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            mutation_op=create_operator("mutation.gaussian_mutation", f=1e-3, N=1, random_state=random_state),
            cross_op=create_operator("crossover.uniform", random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=50, random_state=random_state),
            survivor_sel=create_survivor_selection("Elitism", amount=20, random_state=random_state),
            mutation_prob=0.2,
            crossover_prob=0.8,
            random_state=random_state,
        ),
        "de": DE(
            de_operator_name="DE/best/1",
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            F=0.8,
            Cr=0.8,
        ),
        "gaussianumda": GaussianUMDA(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=20),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=0.1,
            noise=1e-3,
            random_state=random_state,
        ),
        "gaussianpbil": GaussianPBIL(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=20),
            survivor_sel=create_survivor_selection("(m+n)"),
            scale=0.1,
            lr=0.3,
            noise=1e-3,
            random_state=random_state,
        ),
        "crossentropy": CrossEntropyMethod(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1000, random_state=random_state),
            random_state=random_state,
        ),
        "bayesianoptimizaiton": BayesianOptimization(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state),
            batch_size=50,
            max_samples=100,
            random_state=random_state,
        ),
        "randomsearch": RandomSearch(UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state)),
        "nosearch": NoSearch(UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, random_state=random_state)),
    }
    if alg_name == "pso":
        pop_size = 100
        encoding = PSOEncoding(objfunc.vecsize)
        base_constraint_handler = objfunc.constraint_handler
        objfunc.constraint_handler = ExtendedConstraintHandler(
            solution_handler=base_constraint_handler,
            param_handler_dict={"speed": BounceBoundConstraint(objfunc.vecsize)},
            encoding=encoding
        )
        abs_up_lim = np.maximum(np.abs(objfunc.low_lim), np.abs(objfunc.up_lim))
        initializer = ExtendedInitializer(
            solution_init=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size, random_state=random_state),
            param_init_dict={"speed": UniformInitializer(objfunc.vecsize, -abs_up_lim, abs_up_lim)},
            encoding=encoding,
        )
        search_strategy = PSO(
            initializer=initializer,
            encoding=encoding,
            w=0.7,
            c1=1.5,
            c2=1.5
        )
    elif alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    else:
        search_strategy = search_strategy_map[alg_name]

    if memetic:
        local_search = LocalSearch(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=search_strategy.initializer.pop_size),
            operator=create_operator("mutation.gaussian_noise", F=2e-4),
            params={"iters": 20},
        )
        alg = MemeticAlgorithm(
            objfunc=objfunc,
            search_strategy=search_strategy,
            local_search=local_search,
            improvement_selection=create_parent_selection("Best", amount=5),
            keep_improved_solutions=True,
            **algorithm_params
        )
    else:
        alg = StandardAlgorithm(objfunc, search_strategy, **algorithm_params)

    population = alg.optimize()
    best_solution, _ = population.best_solution()
    print()
    print(f"Solution: {[float(i) for i in best_solution]}")
    # alg.display_report(show_plots=show_plots)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_pop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm", help=f"Specify an algorithm. Available options are {available_algorithms}.", default="ES")
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
    parser.add_argument("-o", "--objective", dest="objective", help=f"Name of the objective function. Available options are {available_objectives}", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=42, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
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
        random_state=rng,
    )


if __name__ == "__main__":
    main()
