import sys
sys.path.append("..")

from PyMetaheuristics import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from PyMetaheuristics.SearchMethods import GeneralSearch, MemeticSearch
from PyMetaheuristics.Operators import OperatorReal, OperatorInt, OperatorBinary
from PyMetaheuristics.Algorithms import *
from PyMetaheuristics.benchmarks import *

import argparse

def run_algorithm(alg_name, memetic):
    params = {
        # General
        "stop_cond": "neval or time_limit or fit_target",
        "time_limit": 1.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-3,

        "verbose": True,
        "v_timer": 0.5
    }

    # ["neval", "ngen", "real_time", "cpu_time", "fit_target"]

    objfunc = Sphere(10, "min")

    mutation_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.001})
    cross_op = OperatorReal("Multipoint")
    parent_sel_op = ParentSelection("Best", {"amount": 20})
    selection_op = SurvivorSelection("(m+n)")


    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.0001})
    local_search =  LocalSearch(neihbourhood_op, {"iters":10})

    if alg_name == "HillClimb":
        search_strat = HillClimb(mutation_op)
    elif alg_name == "LocalSearch":
        search_strat = LocalSearch(mutation_op, {"iters":20})
    elif alg_name == "ES":
        search_strat = ES(mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "offspringSize":500})
    elif alg_name == "HS":
        search_strat = HS({"HMS":100, "HMCR":0.8, "BW":0.5, "PAR":0.2})
    elif alg_name == "GA":
        search_strat = GA(mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "pcross":0.8, "pmut":0.2})
    elif alg_name == "SA":
        search_strat = SA(mutation_op, {"iter":100, "temp_init":1, "alpha":0.997})
    elif alg_name == "DE":
        search_strat = DE(OperatorReal("DE/best/1", {"F":0.8, "Cr":0.8}), {"popSize":100})
    elif alg_name == "PSO":
        search_strat = PSO({"popSize":100, "w":0.7, "c1":1.5, "c2":1.5})
    elif alg_name == "NoSearch":
        search_strat = NoSearch({"popSize":100})
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
    
    if memetic:
        alg = MemeticSearch(search_strat, local_search, mem_select, params)
    else:
        alg = GeneralSearch(search_strat, params)
    
    ind, fit = alg.optimize(objfunc)
    print(ind)
    alg.display_report(objfunc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    parser.add_argument("-m", "--memetic", dest='mem', action="store_true", help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "ES"
    mem = False

    if args.alg:
        algorithm_name = args.alg
    
    if args.mem:
        mem = True
   
    run_algorithm(alg_name = algorithm_name, memetic=mem)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     from PyMetaheuristics.BaseSearch import parse_stopping_cond, process_condition
#     a_str = "neval or ngen"
#     parsed = parse_stopping_cond(a_str).as_list()
#     print(process_condition(parsed, neval=True, ngen=True, real_time=True, target=True))