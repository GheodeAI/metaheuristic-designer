import argparse

import numpy as np

from metaheuristic_designer import *
from metaheuristic_designer.algorithms import GeneralAlgorithm, MemeticAlgorithm
from metaheuristic_designer.operators import VectorOperator
from metaheuristic_designer.initializers import *
from metaheuristic_designer.selection_methods import *
from metaheuristic_designer.encodings import *
from metaheuristic_designer.constraint_handlers import *
from metaheuristic_designer.strategies import *
from metaheuristic_designer.benchmarks import *


def run_algorithm(alg_name, memetic, save_state, show_plots, objective, dim):
    params = {
        "stop_cond": "convergence or time_limit",
        "progress_metric": "time_limit",
        "time_limit": 120.0,
        "cpu_time_limit": 100.0,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 500,
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
        
    match alg_name:
        case "HILLCLIMB":
            search_strat = HillClimb(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1),
                operator=VectorOperator("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
            )
        case "LOCALSEARCH":
            search_strat = LocalSearch(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1),
                operator=VectorOperator("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
                params={"iters": 20},
            )
        case "SA":
            search_strat = SA(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=1),
                operator=VectorOperator("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
                params={"iter": 100, "temp_init": 1, "alpha": 0.997},
            )
        case "ES":
            pop_size = 100
            lam = 150
            search_strat = ES(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                mutation_op=VectorOperator("MutNoise", {"distrib": "Gaussian", "F": 1e-3, "N": 1}),
                cross_op=VectorOperator("Multipoint"),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"offspringSize": lam},
            )
        case "GA":
            pop_size = 100
            n_parents = 50
            n_elites = 20
            search_strat = GA(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                mutation_op=VectorOperator("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
                cross_op=VectorOperator("Multipoint"),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("Elitism", {"amount": n_elites}),
                params={"pcross": 0.8, "pmut": 0.2},
            )
        case "HS":
            pop_size = 100
            params["patience"] = 1000
            search_strat = HS(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                params={"HMCR": 0.8, "BW": 0.5, "PAR": 0.2},
            )
        case "DE":
            pop_size = 100
            search_strat = DE(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                de_operator=VectorOperator("DE/best/1", {"F": 0.8, "Cr": 0.8}),
            )
        case "PSO":
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
                solution_init=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                param_init_dict={"speed": UniformInitializer(objfunc.vecsize, -abs_up_lim, abs_up_lim)},
                encoding=encoding,
            )
            search_strat = PSO(
                initializer=initializer,
                params={"w": 0.7, "c1": 1.5, "c2": 1.5},
                encoding=encoding
            )
        case "GAUSSIANUMDA":
            pop_size = 100
            n_parents = 20
            search_strat = GaussianUMDA(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"scale": 0.1, "noise": 1e-3},
            )
        case "GAUSSIANPBIL":
            pop_size = 100
            n_parents = 20
            search_strat = GaussianPBIL(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                parent_sel=ParentSelection("Best", {"amount": n_parents}),
                survivor_sel=SurvivorSelection("(m+n)"),
                params={"scale": 0.1, "lr": 0.3, "noise": 1e-3},
            )
        case "CROSSENTROPY":
            pop_size = 1000
            search_strat = CrossEntropyMethod(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
            )
        case "BO":
            pop_size = 100
            search_strat = BayesianOptimization(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                params={"batch_size": 50, "max_samples": 100},
            )
        case "CRO":
            pop_size = 100
            search_strat = CRO(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                mutation_op=VectorOperator("MutNoise", {"distrib": "Cauchy", "F": 1e-3, "N": 1}),
                cross_op=VectorOperator("Multipoint"),
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "CRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
            search_strat = CRO_SL(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                operator_list=[
                    VectorOperator("DE/rand/1", DEparams),
                    VectorOperator("DE/best/2", DEparams),
                    VectorOperator("DE/current-to-best/1", DEparams),
                    VectorOperator("DE/current-to-rand/1", DEparams),
                ],
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "PCRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
            search_strat = PCRO_SL(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                operator_list=[
                    VectorOperator("DE/rand/1", DEparams),
                    VectorOperator("DE/best/2", DEparams),
                    VectorOperator("DE/current-to-best/1", DEparams),
                    VectorOperator("DE/current-to-rand/1", DEparams),
                ],
                params={"rho": 0.6, "Fb": 0.95, "Fd": 0.1, "Pd": 0.9, "attempts": 3},
            )
        case "DPCRO_SL":
            pop_size = 100
            DEparams = {"F": 0.7, "Cr": 0.8}
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
            search_strat = DPCRO_SL(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                operator_list=[
                    VectorOperator("DE/rand/1", DEparams),
                    VectorOperator("DE/best/2", DEparams),
                    VectorOperator("DE/current-to-best/1", DEparams),
                    VectorOperator("DE/current-to-rand/1", DEparams),
                ],
                params=search_strat_params,
            )
        case "RVNS":
            pop_size = 1
            neighborhood_structures = [VectorOperator("Gauss", {"F": f}, name=f"Gauss(s={f:0.5e})") for f in np.logspace(-6, 0, base=10, num=20)]
            search_strat = RVNS(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                op_list=neighborhood_structures,
            )
        case "VND":
            pop_size = 1
            neighborhood_structures = [VectorOperator("Gauss", {"F": f}, name=f"Gauss(s={f:0.5e})") for f in np.logspace(-6, 0, base=10, num=20)]
            search_strat = VND(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                op_list=neighborhood_structures,
            )
        case "VNS":
            pop_size = 1
            neighborhood_structures = [VectorOperator("Gauss", {"F": f}, name=f"Gauss(s={f:0.5e})") for f in np.logspace(-6, 0, base=10, num=20)]
            local_search = LocalSearch(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size), params={"iters": 20}
            )
            search_strat = VNS(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                op_list=neighborhood_structures,
                local_search=local_search,
                params={"nchange": "seq"},
                inner_loop_params={
                    "stop_cond": "convergence",
                    "patience": 3,
                    "verbose": params["verbose"],
                    "v_timer": params["v_timer"],
                },
            )
        case "GVNS":
            pop_size = 1
            neighborhood_structures = [VectorOperator("Gauss", {"F": f}, name=f"Gauss(s={f:0.5e})") for f in np.logspace(-6, 0, base=10, num=20)]
            local_search = VND(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                op_list=neighborhood_structures,
            )
            search_strat = VNS(
                initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
                op_list=neighborhood_structures,
                local_search=local_search,
                params={"nchange": "pipe"},
                inner_loop_params={
                    "stop_cond": "convergence",
                    "patience": 500,
                    "verbose": params["verbose"],
                    "v_timer": params["v_timer"],
                },
            )
        case "RANDOMSEARCH":
            pop_size = 100
            search_strat = RandomSearch(UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size))
        case "NOSEARCH":
            pop_size = 100
            search_strat = NoSearch(UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size))
        case _:
            raise Exception(f'Algorithm "{alg_name}" doesn\'t exist.')

    if memetic:
        mem_select = ParentSelection("Best", {"amount": 5})
        local_search = LocalSearch(
            initializer=UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=pop_size),
            operator=VectorOperator("RandNoise", {"distrib": "Cauchy", "F": 0.0002}),
            params={"iters": 20},
        )
        alg = MemeticAlgorithm(objfunc, search_strat, local_search, mem_select, params=params)
    else:
        alg = GeneralAlgorithm(objfunc, search_strat, params=params)

    population = alg.optimize()
    print(population.best_solution()[0])
    alg.display_report(show_plots=show_plots)

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
    parser.add_argument("-o", "--objective", dest="objective", help="Name of the objective function.", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Saves the state of the search strategy",
    )
    args = parser.parse_args()

    run_algorithm(
        alg_name=args.algorithm.upper(), memetic=args.mem, save_state=args.save_state, show_plots=args.plot, objective=args.objective, dim=args.dim
    )


if __name__ == "__main__":
    main()
