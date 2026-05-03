import argparse
import numpy as np
import metaheuristic_designer as mhd
from metaheuristic_designer.benchmarks import Sphere, Rastrigin, Rosenbrock, Weierstrass
from metaheuristic_designer import simple


def run_algorithm(alg_name, objective, dim, ngen, seed):
    rng = mhd.check_random_state(seed)

    # Every algorithm uses the same set of parameters
    algo_params = {
        "stop_cond": "ngen",
        "ngen": ngen,
        "reporter": "tqdm",
        "random_state": rng,
    }

    # Objective function – pick from a dictionary
    obj_map = {
        "Sphere":      Sphere(dim, mode="min"),
        "Rastrigin":   Rastrigin(dim, mode="min"),
        "Rosenbrock":  Rosenbrock(dim, mode="min"),
        "Weierstrass": Weierstrass(dim, mode="min"),
    }
    if objective not in obj_map:
        raise ValueError(f"Unknown objective: {objective}")
    objfunc = obj_map[objective]

    # All algorithms are built right here – simple wrappers return an Algorithm
    alg_map = {
        "HillClimb":    simple.hill_climb_real(objfunc, **algo_params),
        "SA":           simple.simulated_annealing_real(objfunc, **algo_params),
        "ES":           simple.evolution_strategy_real(objfunc, **algo_params),
        "GA":           simple.genetic_algorithm_real(objfunc, **algo_params),
        "DE":           simple.differential_evolution_real(objfunc, **algo_params),
        "PSO":          simple.particle_swarm_real(objfunc, **algo_params),
        "RandomSearch": simple.random_search_real(objfunc, **algo_params),
    }
    if alg_name not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    algo = alg_map[alg_name]

    # Run the optimisation and print the result
    population = algo.optimize()
    solution, fitness = population.best_solution(problem_space=True)
    print(f"\nSolution: {solution}")
    print(f"Objective value: {fitness}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", default="GA",
                        choices=["HillClimb","SA","ES","GA","DE","PSO","RandomSearch"])
    parser.add_argument("-o", "--objective", default="Sphere",
                        choices=["Sphere","Rastrigin","Rosenbrock","Weierstrass"])
    parser.add_argument("-d", "--dim", type=int, default=5)
    parser.add_argument("--ngen", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_algorithm(args.algorithm, args.objective, args.dim, args.ngen, args.seed)


if __name__ == "__main__":
    main()