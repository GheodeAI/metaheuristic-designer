from __future__ import annotations
import argparse
import pathlib
import pandas as pd
import metaheuristic_designer as mhd


def run_ga(objective, dim, mutation, crossover, parentsel, survsel, nexecs, popsize, nreps, fileout):
    match objective:
        case "SPHERE":
            objfunc = mhd.benchmarks.Sphere(dim, "min")
        case "RASTRIGIN":
            objfunc = mhd.benchmarks.Rastrigin(dim, "min")
        case "ROSENBROCK":
            objfunc = mhd.benchmarks.Rosenbrock(dim, "min")
        case "WEIERSTRASS":
            objfunc = mhd.benchmarks.Weierstrass(dim, "min")
        case _:
            raise Exception(f'Objective function "{objective}" doesn\'t exist.')
        
    match mutation:
        case "GAUSSMUT":
            mutation_op=mhd.operators.OperatorVector("MutNoise", {"distrib": "Gaussian", "F": 1e-3, "N": 1})
        case "GAUSSSAMPLE":
            mutation_op=mhd.operators.OperatorVector("MutNoise", {"distrib": "Gaussian", "F": 1e-3, "N": 1})

    match crossover:
        case "ONEPOINT":
            cross_op=mhd.operators.OperatorVector("1point")

    n_parents = 50
    match parentsel:
        case "BEST":
            parent_sel=mhd.selectionMethods.ParentSelection("Best", {"amount": n_parents})

    n_elites = 20        
    match survsel:
        case "ELITISM":
            survivor_sel=mhd.selectionMethods.SurvivorSelection("Elitism", {"amount": n_elites})


    search_strat = mhd.strategies.GA(
        initializer=mhd.initializers.UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=popsize),
        mutation_op=mutation_op,
        cross_op=cross_op,
        parent_sel=parent_sel,
        survivor_sel=survivor_sel,
        params={"pcross": 0.8, "pmut": 0.2},
    )

    alg = mhd.algorithms.GeneralAlgorithm(objfunc, search_strat, params={"stop_cond": "neval", "neval": nexecs, "verbose": True})

    exec_data = pd.DataFrame(columns=["Objective", "Dims", "Mutation", "Crossover", "ParentSel", "SurvSel", "Nexecs", "PopSize", "ExecTime", "fitness"])
    for i in range(nreps):
        alg.restart()
        new_population = alg.optimize()
        _, best_fitness = new_population.best_solution()
        exec_data.loc[i] = [objective, dim, mutation.lower(), crossover.lower(), parentsel.lower(), survsel.lower(), nexecs, popsize, alg.cpu_time_spent, best_fitness]
    
    script_path = pathlib.Path(__file__).parent.resolve()
    exec_data.to_csv(f"{script_path}/../data/{fileout}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--objective", dest="objective", help="Name of the objective function.", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument("-m", "--mutation", dest="mut", help="Mutation method to use.", default="GaussMut")
    parser.add_argument("-c", "--crossover", dest="cross", help="Crossover method to use.", default="OnePoint")
    parser.add_argument("-p", "--parentsel", dest="parentsel", help="Parent Selection method to use.", default="Best")
    parser.add_argument("-s", "--survsel", dest="survsel", help="Survivor Selection method to use.", default="Elitism")
    parser.add_argument("-x", "--nexecs", dest="nexecs", help="Maximum number of evaluations of the fitness function.", default=int(3e5), type=int)
    parser.add_argument("-a", "--popsize", dest="popsize", help="Size of the population.", default=100, type=int)
    parser.add_argument("-n", "--nreps", dest="nreps", help="Number of times to run the algorithm.", default=1000, type=int)
    parser.add_argument("-f", "--fileout", dest="fileout", help="Name of the output file.", default="result_GA-mhd-0.csv")
    args = parser.parse_args()

    run_ga(args.objective.upper(), args.dim, args.mut.upper(), args.cross.upper(), args.parentsel.upper(), args.survsel.upper(), args.nexecs, args.popsize, args.nreps, args.fileout)
