import argparse
import logging
from pathlib import Path

import numpy as np

from metaheuristic_designer.algorithms import Algorithm
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
    MemeticStrategy,
)
from metaheuristic_designer.benchmarks import BBOBObjective
from metaheuristic_designer.utils import check_random_state

available_algorithms = ("hillclimb", "localsearch", "sa", "es", "ga", "de", "gaussianumda", "gaussianpbil", "crossentropy", "randomsearch")


def run_algorithm(alg_name, memetic, save_state, fid, instance, dim, evaluations, reporter, random_state):
    algorithm_params = {
        "stop_condition_str": "convergence or max_evaluations",
        "progress_metric_str": "max_evaluations",
        "max_evaluations": evaluations,
        "max_patience": 500,
    }

    objfunc = BBOBObjective(fid=fid, dimension=dim, instance=instance, compact_name=False)

    search_strategy_map = {
        "hillclimb": HillClimb(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            random_state=random_state,
        ),
        "localsearch": LocalSearch(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            iterations=20,
            random_state=random_state,
        ),
        "sa": SA(
            initializer=UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=random_state),
            operator=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
            random_state=random_state,
        ),
        "es": ES(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            crossover_op=create_operator("crossover.uniform", random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
            random_state=random_state,
        ),
        "ga": GA(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-3, N=1, random_state=random_state),
            crossover_op=create_operator("crossover.uniform", random_state=random_state),
            parent_sel=create_parent_selection("Best", amount=50, random_state=random_state),
            survivor_sel=create_survivor_selection("Elitism", amount=20, random_state=random_state),
            mutation_prob=0.2,
            crossover_prob=0.8,
            random_state=random_state,
        ),
        "de": DE(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            de_operator_name="DE/best/1",
            F=0.8,
            Cr=0.8,
            random_state=random_state,
        ),
        "pso": PSO(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            w=0.7,
            c1=1.5,
            c2=1.5,
            random_state=random_state,
        ),
        "cmaes": CMA_ES(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            offspring_size=200,
            random_state=random_state,
        ),
        "gaussianumda": GaussianUMDA(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, random_state=random_state
            ),
            parent_sel=create_parent_selection("Best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            scale=0.1,
            noise=1e-3,
            random_state=random_state,
        ),
        "gaussianpbil": GaussianPBIL(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, random_state=random_state
            ),
            parent_sel=create_parent_selection("Best", amount=20, random_state=random_state),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            scale=0.1,
            lr=0.3,
            noise=1e-3,
            random_state=random_state,
        ),
        "crossentropy": CrossEntropyMethod(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1000, random_state=random_state
            ),
            random_state=random_state,
        ),
        "bayesianoptimization": BayesianOptimization(
            initializer=UniformInitializer(
                objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state
            ),
            batch_size=50,
            max_samples=100,
            random_state=random_state,
        ),
        "randomsearch": RandomSearch(
            UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            random_state=random_state,
        ),
        "nosearch": NoSearch(
            UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=random_state),
            random_state=random_state,
        ),
    }
    if alg_name not in search_strategy_map:
        raise ValueError(f'Algorithm "{alg_name}" not recognized.')
    else:
        search_strategy = search_strategy_map[alg_name]

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

    alg = Algorithm(objfunc, search_strategy, reporter=reporter, **algorithm_params)

    try:
        population = alg.optimize()
    except KeyboardInterrupt as e:
        population = alg.population
        print()
        print(f"Optimization manually interrupted.")

    best_solution, best_objective = population.best_solution()

    # Calculate random search baseline
    random_initializer = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, random_state=42)
    n_samples = 5000
    population = random_initializer.generate_population(objfunc, n_samples)
    population.calculate_fitness()
    min_fitness = population.objective.min()

    # computed values
    optimum_objective = objfunc.problem.optimum.y
    error = best_objective - optimum_objective
    if optimum_objective != 0:
        relative_error = error / abs(optimum_objective)
    else:
        relative_error = error
    dist_to_optimum = np.linalg.norm(best_solution - objfunc.problem.optimum.x)

    print()
    print(f"{objfunc.name} with {alg.name}")
    print(f"{len(objfunc.name)*'-'}------{len(alg.name)*'-'}")
    print(f"| Evaluations:\t\t{alg.stopping_condition.evaluations}")
    print(f"| Iterations:\t\t{alg.stopping_condition.iterations}")
    print(f"| Time:\t\t\t{alg.stopping_condition.real_time_spent:.4}s")
    print("|")
    print(f"| Baseline ({n_samples}):\t{min_fitness:.4g}")
    print(f"| Target objective:\t{objfunc.problem.optimum.y:.4g}")
    print("|")
    print(f"| Best objective:\t{best_objective:.4g}")
    print(f"| Error with optimum:\t{error:.4g}")
    print(f"| Relative error:\t{100*relative_error:.3g}%")
    print()
    print(f"Optimum Solution: {objfunc.problem.optimum.x}")
    print(f"Solution obtained: {best_solution}")
    print(f"Distance to optimum: {dist_to_optimum:.4g}")

    if save_state:
        script_dir = Path(__file__).parent.absolute()
        result_dir = script_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        alg.store_state(result_dir / f"{alg.name}-{objfunc.name}.json", readable=True)


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
    parser.add_argument("-f", "--fid", dest="fid", help=f"BBOB Function idx.", default=1, type=int)
    parser.add_argument("-i", "--instance", dest="instance", help=f"BBOB instance.", default=1, type=int)
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument("-e", "--evaluations", default=100_000, help="Maximum number of evaluations.", type=int)
    parser.add_argument("-r", "--seed", dest="seed", help="Random seed to use", default=None, type=int)
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("-v", "--reporter", default="tqdm", help="Reporter to use for progress tracking. Available options are")
    args = parser.parse_args()

    rng = check_random_state(args.seed)
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.lower(),
        memetic=args.mem,
        save_state=args.save_state,
        fid=args.fid,
        instance=args.instance,
        dim=args.dim,
        evaluations=args.evaluations,
        reporter=args.reporter,
        random_state=rng,
    )


if __name__ == "__main__":
    main()
