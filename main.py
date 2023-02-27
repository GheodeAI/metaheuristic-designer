import sys
sys.path.append("..")

from PyMetaheuristics import *
from PyMetaheuristics.benchmarks.benchmarkFuncs import *

import argparse

def run_algorithm(alg_name):
    params = {
        # General
        "stop_cond": "neval",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 0.5
    }

    objfunc = Sphere(10, "min")

    # mutation_op = OperatorReal("Cauchy", {"F": 0.001})
    mutation_op = OperatorReal("Cauchy", ParamScheduler("Linear", {"F": [0.001, 0.00001]}))

    search_strat = HillClimb(objfunc, mutation_op, params)

    alg = GeneralSearch("HillClimb", objfunc, search_strat, params)
    
    ind, fit = alg.optimize()
    print(ind)
    alg.display_report()


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