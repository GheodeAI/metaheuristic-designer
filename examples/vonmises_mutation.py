from metaheuristic_designer import ObjectiveFunc, ParamScheduler
from metaheuristic_designer.encodings import ParameterExtendingEncoding
from metaheuristic_designer.algorithms import StandardAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import VectorOperator, AdaptativeOperator, MaskedOperator
from metaheuristic_designer.initializers import UniformInitializer, ExponentialInitializer, ExtendedInitializer
from metaheuristic_designer.selection_methods import ParentSelection, SurvivorSelection
from metaheuristic_designer.strategies import *

from metaheuristic_designer.benchmarks import *

import argparse

from copy import copy
import scipy as sp
import numpy as np


class STDAdaptEncoding(ParameterExtendingEncoding):
    def decode_param_func(self, genotype):
        param_vec = self.extract_params(genotype)
        return {
            "F": np.maximum(0, param_vec[:, -1]),
            "mu": param_vec[:, :-1],
        }


def run_algorithm(save_state, objective, dim):
    params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 100.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 10,
        "verbose": True,
        "v_timer": 0.5,
    }

    match objective:
        case "Sphere":
            objfunc = Sphere(dim, "min")
        case "Rastrigin":
            objfunc = Rastrigin(dim, "min")
        case "Rosenbrock":
            objfunc = Rosenbrock(dim, "min")
        case "Weierstrass":
            objfunc = Weierstrass(dim, "min")
        case _:
            raise Exception(f'Objective function "{objective}" doesn\'t exist.')

    adaption_encoding = STDAdaptEncoding(objfunc.vecsize, param_sizes=(("mu", objfunc.vecsize), ("F", 1)))

    ada_mutation_op = AdaptativeOperator(
        VectorOperator("RandNoise", {"distrib": "VonMises"}),
        {
            "mu": VectorOperator("MutateNSigmas", {"tau": 1 / np.sqrt(objfunc.vecsize), "tau_multiple": 0.5/np.sqrt(objfunc.vecsize), "epsilon": 1e-8}),
            "F": VectorOperator("Mutate1Sigma", {"tau": 1 / np.sqrt(objfunc.vecsize), "epsilon": 1e-8})
        },
        encoding=adaption_encoding
    )

    pop_initializer = ExtendedInitializer(
        solution_init = UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=adaption_encoding),
        param_init_dict = {
            "mu": ExponentialInitializer(objfunc.vecsize, beta=100),
            "F": ExponentialInitializer(1, beta=100),
        },
        encoding=adaption_encoding,
    )

    cross_op = VectorOperator("Multipoint")

    parent_sel_op = ParentSelection("Nothing")
    selection_op = SurvivorSelection("(m+n)")

    search_strat = ES(pop_initializer, ada_mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize": 700}, name="Tikhinov Adaptative-ES")

    alg = StandardAlgorithm(objfunc, search_strat, params=params)

    result = alg.optimize()
    ind, best_fitness = result.best_solution(decoded=True)
    ind_extended, _ = result.best_solution(decoded=False)
    print(f"solution vector: {ind}")
    print(f"params: {adaption_encoding.decode_params(ind_extended[None, :])}")
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_population=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save-state",
        dest="save_state",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    parser.add_argument("-o", "--objective", dest="objective", help="Name of the objective function.", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    args = parser.parse_args()

    run_algorithm(save_state=args.save_state, objective=args.objective, dim=args.dim)


if __name__ == "__main__":
    main()

