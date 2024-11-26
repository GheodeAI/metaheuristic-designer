import argparse

import numpy as np

from metaheuristic_designer import ParamScheduler
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import OperatorVector
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import *


def run_algorithm(alg_name, memetic, save_state):
    params = {
        # "stop_cond": "neval or time_limit or fit_target",
        # "stop_cond": "neval or time_limit",
        # "stop_cond": "convergence or time_limit",
        "stop_cond": "time_limit",
        # "stop_cond": "ngen",
        "progress_metric": "time_limit",
        "time_limit": 20.0,
        "cpu_time_limit": 100.0,
        # "ngen": 10000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 200,
        "verbose": True,
        "v_timer": 0.5,
        # "v_timer": -1,
    }

    # objfunc = Sphere(3, "min")
    # objfunc = Sphere(30, "min")
    objfunc = Rastrigin(30, "min")
    # objfunc = Rosenbrock(2, "min")
    # objfunc = Weierstrass(30, "min")

    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)

    parent_params = ParamScheduler("Linear", {"amount": 20})

    mutation_op = OperatorVector("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1})

    cross_op = OperatorVector("Multipoint")

    DEparams = {"F": 0.7, "Cr": 0.8}
    op_list = [
        OperatorVector("DE/rand/1", DEparams),
        OperatorVector("DE/best/2", DEparams),
        OperatorVector("DE/current-to-best/1", DEparams),
        OperatorVector("DE/current-to-rand/1", DEparams),
    ]

    neighborhood_structures = [OperatorVector("Gauss", {"F": f}, name=f"Gauss(s={f:0.5e})") for f in np.logspace(-6, 0, base=10, num=20)]

    parent_sel_op = ParentSelection("Best", parent_params)
    selection_op = SurvivorSelection("(m+n)")

    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorVector("RandNoise", {"distrib": "Cauchy", "F": 0.0002})
    local_search = LocalSearch(pop_initializer, neihbourhood_op, params={"iters": 20})

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, params={"iters": 20})
    elif alg_name == "SA":
        pop_initializer.pop_size = 1
        search_strat = SA(pop_initializer, mutation_op, {"iter": 100, "temp_init": 1, "alpha": 0.997})
    elif alg_name == "ES":
        search_strat = ES(
            pop_initializer,
            mutation_op,
            cross_op,
            parent_sel_op,
            selection_op,
            {"offspringSize": 150},
        )
    elif alg_name == "GA":
        search_strat = GA(
            pop_initializer,
            mutation_op,
            cross_op,
            parent_sel_op,
            selection_op,
            {"pcross": 0.8, "pmut": 0.2},
        )
    elif alg_name == "HS":
        search_strat = HS(pop_initializer, {"HMCR": 0.8, "BW": 0.5, "PAR": 0.2})
    elif alg_name == "DE":
        search_strat = DE(pop_initializer, OperatorVector("DE/best/1", {"F": 0.8, "Cr": 0.8}))
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
    elif alg_name == "GaussianUMDA":
        search_strat = GaussianUMDA(pop_initializer, parent_sel_op, selection_op, params={"scale": 0.1, "noise": 1e-3})
    elif alg_name == "GaussianPBIL":
        search_strat = GaussianPBIL(pop_initializer, parent_sel_op, selection_op, params={"scale": 0.1, "lr": 0.3, "noise": 1e-3})
    elif alg_name == "CrossEntropy":
        pop_initializer.pop_size = 1000
        # pop_initializer.pop_size = 10
        search_strat = CrossEntropyMethod(pop_initializer)
    elif alg_name == "CRO":
        search_strat = CRO(
            pop_initializer,
            mutation_op,
            cross_op,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "CRO_SL":
        search_strat = CRO_SL(
            pop_initializer,
            op_list,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "PCRO_SL":
        search_strat = PCRO_SL(
            pop_initializer,
            op_list,
            {"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
        )
    elif alg_name == "DPCRO_SL":
        search_strat_params = {
            "rho": 0.6,
            "Fb": 0.95,
            "Fd": 0.1,
            "Pd": 0.9,
            "attempts": 3,
            "group_subs": True,
            "dyn_method": "diff",
            "dyn_metric": "best",
            "dyn_steps": 75,
            "prob_amp": 0.1,
        }
        search_strat = DPCRO_SL(pop_initializer, op_list, search_strat_params)
    elif alg_name == "RVNS":
        pop_initializer.pop_size = 1
        search_strat = RVNS(pop_initializer, neighborhood_structures)
    elif alg_name == "VND":
        pop_initializer.pop_size = 1
        search_strat = VND(pop_initializer, neighborhood_structures)
    elif alg_name == "VNS":
        pop_initializer.pop_size = 1
        local_search = LocalSearch(pop_initializer, params={"iters": 20})
        search_strat = VNS(
            initializer=pop_initializer,
            op_list=neighborhood_structures,
            local_search=local_search,
            params={"iters": 100, "nchange": "seq"},
        )
    elif alg_name == "GVNS":
        pop_initializer.pop_size = 1
        local_search = VND(pop_initializer, neighborhood_structures, params={"nchange": "cyclic"})
        search_strat = VNS(
            pop_initializer,
            neighborhood_structures,
            local_search,
            params={"iters": 100, "nchange": "seq"},
        )
    elif alg_name == "RandomSearch":
        search_strat = RandomSearch(pop_initializer)
    elif alg_name == "NoSearch":
        search_strat = NoSearch(pop_initializer)
    else:
        print(f'Error: Algorithm "{alg_name}" doesn\'t exist.')
        exit()

    if memetic:
        alg = MemeticAlgorithm(objfunc, search_strat, local_search, mem_select, params=params)
    else:
        alg = GeneralAlgorithm(objfunc, search_strat, params=params)

    population = alg.optimize()
    print(population.best_solution()[0])
    alg.display_report(show_plots=True)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_pop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest="algorithm", help="Specify an algorithm", default="ES")
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
    args = parser.parse_args()

    run_algorithm(alg_name=args.algorithm, memetic=args.mem, save_state=args.save_state)


if __name__ == "__main__":
    main()
