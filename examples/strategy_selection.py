import argparse

from metaheuristic_designer import ObjectiveFunc, ParamScheduler, simple, NullOperator
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm, StrategySelection
from metaheuristic_designer.operators import VectorOperator
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.selection_methods import ParentSelection, SurvivorSelection, NullParentSelection
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import *


def run_algorithm(save_report):
    objfunc = HappyCat(3, "min")
    single_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1)
    pop_initializer = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)

    # Define algorithms to be tested
    strategies = [
        HillClimb(single_initializer, VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}), name="HillClimb-Gauss"),
        HillClimb(single_initializer, VectorOperator("RandNoise", {"distrib": "Cauchy", "F": 1e-4}), name="HillClimb-Cauchy"),
        LocalSearch(
            single_initializer, VectorOperator("RandNoise", {"distrib": "Cauchy", "F": 1e-4}), params={"iters": 20}, name="LocalSearch-Cauchy"
        ),
        LocalSearch(single_initializer, VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}), params={"iters": 20}, name="LocalSearch-Gauss"),
        SA(
            single_initializer,
            VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}),
            params={"iter": 100, "temp_init": 1, "alpha": 0.997},
            name="SA-Gauss",
        ),
        SA(
            single_initializer,
            VectorOperator("RandNoise", {"distrib": "Cauchy", "F": 1e-4}),
            params={"iter": 100, "temp_init": 1, "alpha": 0.997},
            name="SA-Cauchy",
        ),
        SA(
            pop_initializer,
            VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}),
            params={"iter": 100, "temp_init": 1, "alpha": 0.997},
            name="ParallelSA-Gauss",
        ),
        ES(
            pop_initializer,
            VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}),
            NullOperator(),
            NullParentSelection(),
            SurvivorSelection("(m+n)"),
            params={"offspringSize": 150},
            name="ES-(100+150)",
        ),
        ES(
            pop_initializer,
            VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}),
            NullOperator(),
            NullParentSelection(),
            SurvivorSelection("(m,n)"),
            params={"offspringSize": 400},
            name="ES-(100,400)",
        ),
        GA(
            pop_initializer,
            VectorOperator("RandNoise", {"distrib": "Gauss", "F": 1e-4}),
            VectorOperator("Multipoint"),
            ParentSelection("Tournament", {"amount": 60, "p": 0.1}),
            SurvivorSelection("Elitism", {"amount": 10}),
            params={"pcross": 0.8, "pmut": 0.1},
            name="GA",
        ),
        # PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5}, name="PSO"),
        DE(pop_initializer, VectorOperator("DE/best/1", {"F": 0.8, "Cr": 0.8}), name="DE/best/1"),
        DE(pop_initializer, VectorOperator("DE/rand/1", {"F": 0.8, "Cr": 0.8}), name="DE/rand/1"),
        DE(pop_initializer, VectorOperator("DE/current-to-best/1", {"F": 0.8, "Cr": 0.8}), name="DE/current-to-best/1"),
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
    parser.add_argument(
        "-s",
        "--save-report",
        dest="save_report",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    args = parser.parse_args()

    save_report = False

    if args.save_report:
        save_report = True

    run_algorithm(save_report=save_report)


if __name__ == "__main__":
    main()
