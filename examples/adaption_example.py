import argparse
import logging

import numpy as np
import scipy as sp

from metaheuristic_designer.algorithms import StandardAlgorithm
from metaheuristic_designer.operators import create_operator, AdaptiveOperator, NullOperator
from metaheuristic_designer.initializers import UniformInitializer, ExtendedInitializer, InitializerFromLambda
from metaheuristic_designer.parent_selection_methods import NullParentSelection
from metaheuristic_designer.survivor_selection_methods import create_survivor_selection
from metaheuristic_designer.encodings import ParameterExtendingEncoding
from metaheuristic_designer.strategies import ES
from metaheuristic_designer.benchmarks import *
from metaheuristic_designer.utils import check_random_state


def run_algorithm(save_state, show_plots, objective, dim, random_state):
    algorithm_params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 100.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 50,
        "verbose": True,
        "v_timer": 0.5,
    }

    # ---- Objective function ----
    objective_map = {
        "sphere": Sphere(dim, mode="min"),
        "rastrigin": Rastrigin(dim, mode="min"),
        "rosenbrock": Rosenbrock(dim, mode="min"),
        "weierstrass": Weierstrass(dim, mode="min"),
    }
    if objective.lower() not in objective_map:
        raise ValueError(f'Objective function "{objective}" does not exist.')
    objfunc = objective_map[objective.lower()]

    # ---- Self‑adaption encoding ----
    adaption_encoding = ParameterExtendingEncoding(
        objfunc.vecsize,
        param_sizes=[("F", 1)],      # each individual carries its own mutation strength
    )

    # ---- Extended initializer ----
    pop_initializer = ExtendedInitializer(
        solution_init=UniformInitializer(
            objfunc.vecsize, objfunc.low_lim, objfunc.up_lim,
            pop_size=100, encoding=adaption_encoding, random_state=random_state
        ),
        param_init_dict={
            "F": InitializerFromLambda(
                generator=lambda rng: sp.stats.expon(scale=0.02).rvs(size=1, random_state=rng),
                pop_size=100, encoding=adaption_encoding, random_state=random_state
            )
        },
        encoding=adaption_encoding,
        random_state=random_state,
    )

    # ---- Adaptive operator ----
    # The base operator applies Gaussian noise to the solution part.
    # The "F" parameter operator mutates the mutation strength itself (1/√dim rule).
    ada_mutation_op = AdaptiveOperator(
        base_operator=create_operator("mutation.gaussian_mutation", N=1, random_state=random_state),
        param_operators={
            "F": create_operator(
                "mutation.mutate_1_sigma",
                tau=1.0 / np.sqrt(2*objfunc.vecsize),
                epsilon=1e-7,
                random_state=random_state,
            )
            # "F": create_operator("dummy", f=1e-4)
        },
        encoding=adaption_encoding,
    )

    # Crossover (can be null or a real crossover)
    cross_op = create_operator("crossover.multipoint", random_state=random_state)
    # cross_op = None

    # ---- Build the ES strategy ----
    search_strategy = ES(
        pop_initializer,
        mutation_op=ada_mutation_op,
        cross_op=cross_op,
        parent_sel=NullParentSelection(),
        survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
        offspring_size=250,
        name="Adaptative-ES",
        random_state=random_state,
    )

    # ---- Wrap in algorithm and optimize ----
    alg = StandardAlgorithm(objfunc, search_strategy, **algorithm_params)

    population = alg.optimize()
    decoded_solution, best_fitness = population.best_solution(decoded=True)
    genotype, _ = population.best_solution(decoded=False)

    print(f"Decoded solution vector: {decoded_solution}")
    print(f"Genotype (includes self-adapted F): {genotype}")
    print(f"Best fitness: {best_fitness}")
    alg.display_report(show_plots=show_plots)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_population=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--objective", default="Sphere",
                        help="Objective function name (Sphere, Rastrigin, Rosenbrock, Weierstrass).")
    parser.add_argument("-d", "--dim", type=int, default=3,
                        help="Dimensionality of the problem.")
    parser.add_argument("-s", "--save-state", dest="save_state", action="store_true",
                        help="Save algorithm state to JSON.")
    parser.add_argument("-p", "--plot", dest="plot", action="store_true",
                        help="Show convergence plot.")
    parser.add_argument("-r", "--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--log", default="WARNING",
                        help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())
    rng = check_random_state(args.seed)

    run_algorithm(
        save_state=args.save_state,
        show_plots=args.plot,
        objective=args.objective,
        dim=args.dim,
        random_state=rng,
    )


if __name__ == "__main__":
    main()