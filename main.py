import sys
sys.path.append("..")

from PyMetaheuristics import ObjectiveFunc, ParentSelection, SurvivorSelection, ParamScheduler
from PyMetaheuristics.SearchMethods import GeneralSearch
from PyMetaheuristics.Operators import OperatorReal, OperatorInt, OperatorBinary
from PyMetaheuristics.Algorithms import *
from PyMetaheuristics.benchmarks import *

import argparse

def run_algorithm(alg_name):
    params = {
        # General
        "stop_cond": "time_limit",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-8,

        "verbose": True,
        "v_timer": 0.5
    }

    objfunc = Rosenbrock(2, "min")

    mutation_op = OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.001})
    cross_op = OperatorReal("Multipoint")
    parent_sel_op = ParentSelection("Best", {"amount": 40})
    selection_op = SurvivorSelection("(m+n)")

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
        search_strat = SA(mutation_op, {"iter":100, "temp_init":30, "alpha":0.999})
    elif alg_name == "DE":
        search_strat = DE(OperatorReal("DE/best/1", {"F":0.8, "Cr":0.8}), {"popSize":100})
    elif alg_name == "PSO":
        search_strat = PSO(params={"popSize":100})
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()

    alg = GeneralSearch(search_strat, params)
    
    ind, fit = alg.optimize(objfunc)
    print(ind)
    alg.display_report(objfunc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "ES"
    if args.alg:
        algorithm_name = args.alg
   
    run_algorithm(alg_name = algorithm_name)

if __name__ == "__main__":
    main()