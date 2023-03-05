import sys
sys.path.append("..")

from PyMetaheuristics import GeneralSearch, MemeticSearch, ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from PyMetaheuristics.Operators import OperatorReal, OperatorInt, OperatorBinary
from PyMetaheuristics.Algorithms import *

from PyMetaheuristics.benchmarks.benchmarkFuncs import *

import argparse

def run_algorithm(alg_name, memetic):
    params = {
        # General
        "stop_cond": "fit_target",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 5e5,
        "fit_target": 0,

        "verbose": True,
        "v_timer": 0.5
    }

    objfunc = MaxOnes(1000, "min")

    #mutation_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.001})
    mutation_op = OperatorBinary("MutSample", {"method":"Bernloulli", "p": 0.5, "N":4})
    cross_op = OperatorReal("Multipoint")
    #cross_op = OperatorReal("PSO", {"w":1.5, "c1":0.8, "c2":0.8})
    parent_sel_op = ParentSelection("Best", {"amount": 20})
    selection_op = SurvivorSelection("(m+n)")


    mem_select = ParentSelection("Best", {"amount": 5})
    neihbourhood_op = OperatorBinary("MutSample", {"method":"Bernloulli", "p": 0.5, "N":3})
    local_search =  LocalSearch(objfunc, neihbourhood_op, {"iters":10})

    if alg_name == "HillClimb":
        search_strat = HillClimb(objfunc, mutation_op)
    elif alg_name == "LocalSearch":
        search_strat = LocalSearch(objfunc, mutation_op, {"iters":20})
    elif alg_name == "ES":
        search_strat = ES(objfunc, mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "offspringSize":500})
    elif alg_name == "GA":
        search_strat = GA(objfunc, mutation_op, cross_op, parent_sel_op, selection_op, {"popSize":100, "pcross":0.8, "pmut":0.2})
    elif alg_name == "SA":
        search_strat = SA(objfunc, mutation_op, {"iter":100, "temp_init":30, "alpha":0.99})
    elif alg_name == "DE":
        search_strat = DE(objfunc, OperatorReal("DE/best/1", {"F":0.8, "Cr":0.8}), {"popSize":100})
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
    
    if memetic:
        alg = MemeticSearch(search_strat, local_search, mem_select, params)
    else:
        alg = GeneralSearch(search_strat, params)
    
    ind, fit = alg.optimize()
    print(ind)
    alg.display_report()


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