import sys
sys.path.append("../..")

from PyMetaheuristics import *
from .benchmarkFuncs import *

import os
import pandas as pd
import signal

def exec_runs(evalg_inst, dframe, func_name, subs_name, nruns=10):
    fit_list = []
    for i in range(nruns):
        _, fit = evalg_inst.optimize()
        evalg_inst.restart()
        fit_list.append(fit)
    fit_list = np.array(fit_list)
    print(f"\nRESULTS {func_name}-{subs_name}:")
    print(f"min: {fit_list.min():e}; mean: {fit_list.mean():e}; std: {fit_list.std():e}")
    dframe.loc[len(dframe)] = [func_name, subs_name, fit_list.min(), fit_list.mean(), fit_list.std()]

def save_dframe_error(dframe, last_func, last_subs, alg_name):
    dframe.to_csv(f"./results/{alg_name}_incomplete.csv")
    with open(f"results/log.txt", "w") as f:
        f.write("Warning: stopped before all the runs were completed")
        f.write(f"\t - last function: {last_func}, with operators: {last_subs}")

def save_dframe(dframe, alg_name):
    dframe.to_csv(f"./results/{alg_name}_results.csv")
    print("saved dataframe to disk")

def main():
    DEparams = {"F":0.7, "Pr":0.8}
    operators_real = [
        OperatorReal("DE/rand/1", DEparams),
        OperatorReal("DE/best/2", DEparams),
        OperatorReal("DE/current-to-best/1", DEparams),
        OperatorReal("DE/current-to-rand/1", DEparams)
    ]

    combination_DE = [
        [0,1,2,3],
        [0,1,2],[0,1,3],[0,2,3],[1,2,3],
        [0,1],[0,2],[0,3],[1,2],[1,3],[2,3],
        [0],[1],[2],[3]
    ]

    params = {
        # CRO_SL
        "ReefSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 1,
        "Pd": 0.1,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "dynamic": True,
        "dyn_method": "fitness",
        "dyn_metric": "avg",
        "dyn_steps": 100,
        "prob_amp": 0.1,


        # population based algorithms
        "popSize": 400,


        # PSO
        "w": 0.7,
        "c1": 1.3,
        "c2": 1.7,


        # Genetic
        "pmut": 0.3,


        # General
        "stop_cond": "neval",
        "time_limit": 4000.0,
        "ngen": 3500,
        "neval": 3e3,
        "fit_target": 1000,

        "verbose": False,
        "v_timer": 1,
    }

    funcs = [
        Sphere(30),
        HighCondElliptic(30),
        BentCigar(30),
        Discus(30),
        Rosenbrock(30),
        Ackley(30),
        Weierstrass(30),
        Griewank(30),
        Rastrigin(30),
        ModSchwefel(30),
        # Katsuura(30),
        # HappyCat(30),
        # HGBat(30),
        # ExpandedGriewankPlusRosenbrock(30),
        # ExpandedShafferF6(30)
    ]


    alg_name = "pso"
    exec_data = pd.DataFrame(columns = ["operators", "function", "best", "mean", "std"])
    last_func_name = str(type(funcs[0]))[8:-2].replace("CompareTests.", "")
    comb_name = "4_DE0123"

    try:
        if not os.path.exists('results'):
            os.makedirs('results')
        
        for f in funcs:
            last_func_name = str(type(f))[8:-2].split(".")[-1]
            print()
            print(last_func_name)

            #for comb in combination_DE:
            #    comb_name = f"{len(comb)}DE_{''.join([str(i) for i in comb])}"
            #    operators_filtered = [operators_real[i] for i in range(4) if i in comb]
            #    c = CRO_SL(f, operators_filtered, params)
            #    exec_runs(c, exec_data, last_func_name, comb_name)

            #c = Genetic(f, OperatorReal("Multipoint"), OperatorReal("Gauss", {"F":0.001}), params)
            #c = LSHADE(f, params)
            c = PSO(f, params)
            exec_runs(c, exec_data, last_func_name, "Basic")
        save_dframe(exec_data, alg_name)
    except KeyboardInterrupt as k:
        print("\nexecution stopped early")
        save_dframe_error(exec_data, last_func_name, comb_name, alg_name)
    except Exception as e:
        print(f"\nsomething went wrong:\n{e}")
        save_dframe_error(exec_data, last_func_name, comb_name, alg_name)

if __name__ == "__main__":
    main()