import sys
sys.path.append("..")

from PyEvolAlg import *
from PyEvolAlg.ParamScheduler import *
from PyEvolAlg.benchmarks.benchmarkFuncs import *

import argparse

def run_algorithm(alg_name):
    params = {
        # Population-based
        "popSize": 100,

        # Coral reef optimization
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.1,
        "Pd": 1,
        "k": 3,
        "K": 1,

        ## Dynamic CRO-SL
        "group_subs": False,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "best",
        "dyn_steps": 100,
        "prob_amp": 0.1,

        # Genetic algorithm
        "pmut": 0.1,
        "pcross":0.9,

        # Evolution strategy
        "offspringSize":500,

        # Particle swarm optimization
        "w": 0.729,
        "c1": 1.49445,
        "c2": 1.49445,

        # Reinforcement learning based search
        "discount": 0.6,
        "alpha": 0.7,
        "eps": 0.1,
        "nstates": 5,

        "sel_exp": 2,

        # Harmony search
        "HMCR": 0.9,
        "PAR" : 0.3,
        "BN" : 1,

        # Hill climbing
        "p": 0,

        # Simulated annealing
        "iter": 100,
        "temp_init": 20,
        "alpha" : 0.9975,

        # General
        "stop_cond": "neval",
        "time_limit": 20.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": -1
    }

    operators = [
        OperatorReal("Multipoint"),
        OperatorReal("DE/best/1", {"F":0.7, "Cr":0.8}),
        OperatorReal("Gauss", {"F":0.001}),
        OperatorReal("Cauchy", {"F":0.005}),
    ]

    objfunc = Sphere(10, "min")

    mutation_op = OperatorReal("Cauchy", {"F": 0.001})
    # mutation_op = OperatorReal("Gauss", ParamScheduler("Lineal", {"F":[0.1, 0.001]}))
    cross_op = OperatorReal("Multipoint")
    # parent_select_op = ParentSelection("Tournament", {"amount": 3, "p":0.1})
    parent_select_op = ParentSelection("Tournament", ParamScheduler("Lineal", {"amount": [2, 7], "p":0.1}))
    replace_op = SurvivorSelection("(m+n)")

    if alg_name == "CRO_SL":
        alg = CRO_SL(objfunc, operators, params)
    elif alg_name == "GA":
        alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
    elif alg_name == "ES":
        alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
    elif alg_name == "DE":
        alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
    elif alg_name == "LSHADE":
        alg = LSHADE(objfunc, params)
    elif alg_name == "PSO":
        alg = PSO(objfunc, params)
    elif alg_name == "RLevol":
        alg = RLEvolution(objfunc, operators, params)
    elif alg_name == "HS":
        alg = HS(objfunc, OperatorReal("Gauss", {"F":params["BN"]}), OperatorReal("RandSample", {"method":"Gauss", "F":params["BN"]}), params)
    elif alg_name == "In-HS":
        operators_mut_InHS = [
            OperatorReal("Gauss", {"F":params["BN"]}),
            OperatorReal("Cauchy", {"F":params["BN"]/2}),
            OperatorReal("Laplace", {"F":params["BN"]}),
        ]
        alg = InHS(objfunc, operators_mut_InHS, OperatorReal("RandSample", {"method":"Gauss", "F":params["BN"]}), params)
    elif alg_name == "SA":
        alg = SimAnn(objfunc, mutation_op, params)
    elif alg_name == "HillClimb":
        alg = HillClimb(objfunc, mutation_op, params)
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
        

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