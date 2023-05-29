from pyevolcomp import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from pyevolcomp.SearchMethods import GeneralSearch, MemeticSearch
from pyevolcomp.Operators import OperatorReal, OperatorInt, OperatorBinary
from pyevolcomp.Initializers import UniformVectorInitializer
from pyevolcomp.Algorithms import *

from pyevolcomp.benchmarks import *

import argparse

from copy import copy 
import scipy as sp

def run_algorithm(alg_name, memetic, save_state):
    params = {
        "stop_cond": "neval or time_limit or fit_target",
        # "stop_cond": "neval or time_limit",
        "time_limit": 100.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,

        "verbose": True,
        "v_timer": 0.5
    }

    objfunc = Sphere(30, "min")
    # objfunc = Rosenbrock(30, "min")
    # objfunc = Weierstrass(30, "min")
    pop_initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)

    
    parent_params = ParamScheduler("Linear", {"amount": 20})
    # select_params = ParamScheduler("Linear")
    
    mut_params = ParamScheduler("Linear", {"method":"Cauchy", "F": [0.01, 0.00001]})
    mutation_op = OperatorReal("RandNoise", mut_params)

    cross_op = OperatorReal("Multipoint")

    DEparams = {"F":0.7, "Cr":0.8}
    op_list = [
        OperatorReal("DE/rand/1", DEparams),
        OperatorReal("DE/best/2", DEparams),
        OperatorReal("DE/current-to-best/1", DEparams),
        OperatorReal("DE/current-to-rand/1", DEparams)
    ]


    parent_sel_op = ParentSelection("Best", parent_params)
    selection_op = SurvivorSelection("(m+n)")
    
    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.0002})
    local_search =  LocalSearch(pop_initializer, neihbourhood_op, {"iters":10})

    

    if alg_name == "HillClimb":
        pop_initializer.pop_size = 1
        search_strat = HillClimb(pop_initializer, mutation_op)
    elif alg_name == "LocalSearch":
        pop_initializer.pop_size = 1
        search_strat = LocalSearch(pop_initializer, mutation_op, {"iters":20})
    elif alg_name == "SA":
        pop_initializer.pop_size = 1
        search_strat = SA(pop_initializer, mutation_op, {"iter":100, "temp_init":1, "alpha":0.997})
    elif alg_name == "ES":
        search_strat = ES(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"offspringSize":150})
    elif alg_name == "GA":
        search_strat = GA(pop_initializer, mutation_op, cross_op, parent_sel_op, selection_op, {"pcross":0.8, "pmut":0.2})
    elif alg_name == "HS":
        search_strat = HS(pop_initializer, {"HMCR":0.8, "BW":0.5, "PAR":0.2})
    elif alg_name == "DE":
        search_strat = DE(pop_initializer, OperatorReal("DE/best/1", {"F":0.8, "Cr":0.8}))
    elif alg_name == "PSO":
        search_strat = PSO(pop_initializer, {"w":0.7, "c1":1.5, "c2":1.5})
    elif alg_name == "CRO":
        search_strat = CRO(pop_initializer, mutation_op, cross_op, {"rho":0.6, "Fb":0.95, "Fd":0.1, "Pd":0.9, "attempts":3})
    elif alg_name == "CRO_SL":
        search_strat = CRO_SL(pop_initializer, op_list, {"rho":0.6, "Fb":0.95, "Fd":0.1, "Pd":0.9, "attempts":3})
    elif alg_name == "PCRO_SL":
        search_strat = PCRO_SL(pop_initializer, op_list, {"rho":0.6, "Fb":0.95, "Fd":0.1, "Pd":0.9, "attempts":3})
    elif alg_name == "DPCRO_SL":
        search_strat_params = {
            "rho":0.6,
            "Fb":0.95,
            "Fd":0.1,
            "Pd":0.9,
            "attempts": 3,
            "group_subs": True,
            "dyn_method": "diff",
            "dyn_metric": "best",
            "dyn_steps": 75,
            "prob_amp": 0.1
        }
        search_strat = DPCRO_SL(pop_initializer, op_list, search_strat_params)
    elif alg_name == "RandomSearch":
        search_strat = RandomSearch(pop_initializer)
    elif alg_name == "NoSearch":
        search_strat = NoSearch(pop_initializer)
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
    
    if memetic:
        alg = MemeticSearch(objfunc, search_strat, local_search, mem_select, params=params)
    else:
        alg = GeneralSearch(objfunc, search_strat, params=params)
    
    ind, fit = alg.optimize()
    print(ind)
    alg.display_report(show_plots=False)

    if save_state:
        alg.store_state("./examples/results/test.json", readable=True, show_pop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    parser.add_argument("-m", "--memetic", dest='mem', action="store_true", help='Does local search after mutation')
    parser.add_argument("-s", "--save-state", dest='save_state', action="store_true", help='Saves the state of the search strategy')
    args = parser.parse_args()

    algorithm_name = "ES"
    mem = False
    save_state = False

    if args.alg:
        algorithm_name = args.alg
    
    if args.mem:
        mem = True
    
    if args.save_state:
        save_state = True
   
    run_algorithm(alg_name = algorithm_name, memetic=mem, save_state=save_state)

if __name__ == "__main__":
    main()