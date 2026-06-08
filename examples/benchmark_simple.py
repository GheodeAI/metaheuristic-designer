import argparse
import logging
from pathlib import Path
import numpy as np
import metaheuristic_designer as mhd
from metaheuristic_designer.benchmarks import BBOBObjective
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.utils import check_rng
from metaheuristic_designer import simple

available_algorithms = ("hillclimb", "localsearch", "sa", "es", "ga", "de", "pso", "randomsearch")


def run_algorithm(alg_name, fid, instance, dim, reporter, evaluations, save_state, rng):
    algorithm_params = {
        "stop_condition_str": "convergence or max_evaluations",
        "progress_metric_str": "max_evaluations",
        "max_evaluations": evaluations,
        "max_patience": 500,
        "reporter": reporter,
        "rng": rng,
    }

    objfunc = BBOBObjective(fid=fid, dimension=dim, instance=instance, compact_name=False)

    # All algorithms are built right here – simple wrappers return an Algorithm
    alg_map = {
        "hillclimb": simple.hill_climb_real(objfunc, **algorithm_params),
        "localsearch": simple.local_search_real(objfunc, **algorithm_params),
        "sa": simple.simulated_annealing_real(objfunc, **algorithm_params),
        "es": simple.evolution_strategy_real(objfunc, **algorithm_params),
        "ga": simple.genetic_algorithm_real(objfunc, **algorithm_params),
        "de": simple.differential_evolution_real(objfunc, **algorithm_params),
        "pso": simple.particle_swarm_real(objfunc, **algorithm_params),
        "randomsearch": simple.random_search_real(objfunc, **algorithm_params),
    }
    if alg_name not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    alg = alg_map[alg_name]

    try:
        population = alg.optimize()
    except KeyboardInterrupt:
        population = alg.population
        print()
        print("Optimization manually interrupted.")

    best_solution, best_objective = population.best_solution()

    # Calculate random search baseline
    random_initializer = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=1, rng=42)
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

    rng = check_rng(args.seed)
    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())

    run_algorithm(
        alg_name=args.algorithm.lower(),
        fid=args.fid,
        instance=args.instance,
        dim=args.dim,
        evaluations=args.evaluations,
        reporter=args.reporter,
        save_state=args.save_state,
        rng=rng,
    )


if __name__ == "__main__":
    main()
