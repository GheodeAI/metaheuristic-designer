import argparse
import logging

import numpy as np

from metaheuristic_designer import simple
from metaheuristic_designer.algorithms import StandardAlgorithm, StrategySelection
from metaheuristic_designer.operators import create_operator, NullOperator
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.parent_selection_methods import create_parent_selection, NullParentSelection
from metaheuristic_designer.survivor_selection_methods import create_survivor_selection
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import *
from metaheuristic_designer.utils import check_random_state


def run_algorithm(save_report, random_state):
    objfunc = HappyCat(3, "min")
    single_initializer = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, pop_size=1, random_state=random_state)
    pop_initializer = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, pop_size=100, random_state=random_state)

    # Define algorithms to be tested (all strategies instantiated upfront)
    strategies = [
        HillClimb(
            single_initializer,
            operator=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            name="HillClimb-Gauss",
        ),
        HillClimb(
            single_initializer,
            operator=create_operator("mutation.cauchy_mutation", F=1e-4, random_state=random_state),
            name="HillClimb-Cauchy",
        ),
        LocalSearch(
            single_initializer,
            operator=create_operator("mutation.cauchy_mutation", F=1e-4, random_state=random_state),
            iterations=20,
            name="LocalSearch-Cauchy",
        ),
        LocalSearch(
            single_initializer,
            operator=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            iterations=20,
            name="LocalSearch-Gauss",
        ),
        SA(
            single_initializer,
            operator=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
            name="SA-Gauss",
        ),
        SA(
            single_initializer,
            operator=create_operator("mutation.cauchy_mutation", F=1e-4, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
            name="SA-Cauchy",
        ),
        SA(
            pop_initializer,
            operator=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            iterations=100,
            temperature_init=1,
            alpha=0.997,
            name="ParallelSA-Gauss",
        ),
        ES(
            pop_initializer,
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            cross_op=NullOperator(),
            parent_sel=NullParentSelection(),
            survivor_sel=create_survivor_selection("(m+n)", random_state=random_state),
            offspring_size=150,
            name="ES-(100+150)",
        ),
        ES(
            pop_initializer,
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            cross_op=NullOperator(),
            parent_sel=NullParentSelection(),
            survivor_sel=create_survivor_selection("(m,n)", random_state=random_state),
            offspring_size=400,
            name="ES-(100,400)",
        ),
        GA(
            pop_initializer,
            mutation_op=create_operator("mutation.gaussian_mutation", F=1e-4, random_state=random_state),
            cross_op=create_operator("crossover.multipoint", random_state=random_state),
            parent_sel=create_parent_selection("tournament", amount=60, tournament_size=3, prob=1.0, random_state=random_state),
            survivor_sel=create_survivor_selection("elitism", amount=10, random_state=random_state),
            mutation_prob=0.1,
            crossover_prob=0.8,
            random_state=random_state,
            name="GA",
        ),
        DE(
            de_operator_name="DE/best/1",
            initializer=pop_initializer,
            F=0.8,
            Cr=0.8,
            name="DE/best/1",
        ),
        DE(
            de_operator_name="DE/rand/1",
            initializer=pop_initializer,
            F=0.8,
            Cr=0.8,
            name="DE/rand/1",
        ),
        DE(
            de_operator_name="DE/current-to-best/1",
            initializer=pop_initializer,
            F=0.8,
            Cr=0.8,
            name="DE/current-to-best/1",
        ),
        RandomSearch(pop_initializer),
    ]

    algorithm_search = StrategySelection(
        objfunc,
        strategies,
        algorithm_params={
            "stop_cond": "neval",
            "neval": 1e4,
            "verbose": False,
        },
        params={"verbose": True, "repetitions": 10},
    )

    solution, best_fitness, report = algorithm_search.optimize()
    print(f"solution: {solution}")
    print(f"with fitness: {best_fitness}")
    print(report)
    if save_report:
        report.to_csv("./examples/results/strategy_selection_report.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save-report", dest="save_report", action="store_true", help="Save report as CSV.")
    parser.add_argument("-r", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger("metaheuristic_designer").setLevel(args.log.upper())
    rng = check_random_state(args.seed)

    run_algorithm(save_report=args.save_report, random_state=rng)


if __name__ == "__main__":
    main()
