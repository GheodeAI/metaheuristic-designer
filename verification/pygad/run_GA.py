from __future__ import annotations
import time
import argparse
import pathlib
import pandas as pd
import metaheuristic_designer as mhd
import pygad


def run_ga(objective, dim, mutation, crossover, parentsel, survsel, ngen, popsize, nreps, fileout):
    match objective:
        case "sphere":
            objfunc = mhd.benchmarks.Sphere(dim, "min")
        case "rastrigin":
            objfunc = mhd.benchmarks.Rastrigin(dim, "min")
        case "rosenbrock":
            objfunc = mhd.benchmarks.Rosenbrock(dim, "min")
        case "weierstrass":
            objfunc = mhd.benchmarks.Weierstrass(dim, "min")
        case _:
            raise Exception(f'Objective function "{objective}" doesn\'t exist.')

    objfunc_lambda = lambda _obj, x, _idx: objfunc.objective(x)

    interpret_mutation = {
        "uniformsample": "random"
    }
    mutate_op = interpret_mutation[mutation]

    interpret_crossover = {
        "onepoint": "single_point"
    }
    cross_op = interpret_crossover[crossover]

    interpret_partentsel = {
        "generational": "sss"
    }
    parent_sel = interpret_partentsel[parentsel]

    n_elites = 20        
    match survsel:
        case "elitism":
            survivor_sel=mhd.selectionMethods.SurvivorSelection("Elitism", {"amount": n_elites})

    alg = pygad.GA(
        fitness_func=objfunc_lambda,
        num_generations=ngen,
        num_parents_mating=popsize,
        sol_per_pop=popsize,
        num_genes=dim,
        init_range_low=objfunc.low_lim,
        init_range_high=objfunc.up_lim,
        parent_selection_type=parent_sel,
        crossover_type=cross_op,
        crossover_probability=0.8,
        mutation_type=mutate_op,
        mutation_probability=0.2,
        mutation_percent_genes=10,
        random_mutation_min_val=objfunc.low_lim,
        random_mutation_max_val=objfunc.up_lim,
        keep_elitism=20,
    )

    exec_data = pd.DataFrame(columns=["Objective", "Dims", "Mutation", "Crossover", "ParentSel", "SurvSel", "Ngen", "PopSize", "ExecTime", "fitness"])
    script_path = pathlib.Path(__file__).parent.resolve()
    for i in range(nreps):
        print(f"Repetition {i+1}/{nreps}")
        cpu_time_start = time.process_time()
        new_population = alg.run()
        cpu_time_spent = time.process_time() - cpu_time_start
        _, best_fitness, _ = alg.best_solution()
        exec_data.loc[i] = [objective.lower(), dim, mutation.lower(), crossover.lower(), parentsel.lower(), survsel.lower(), ngen, popsize, cpu_time_spent, best_fitness]
        exec_data.to_csv(f"{script_path}/../data/{fileout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--objective", dest="objective", help="Name of the objective function.", default="Sphere")
    parser.add_argument("-d", "--dim", dest="dim", help="Dimension of the vectors to optimize.", default=3, type=int)
    parser.add_argument("-m", "--mutation", dest="mut", help="Mutation method to use.", default="UniformSample")
    parser.add_argument("-c", "--crossover", dest="cross", help="Crossover method to use.", default="OnePoint")
    parser.add_argument("-p", "--parentsel", dest="parentsel", help="Parent Selection method to use.", default="Generational")
    parser.add_argument("-s", "--survsel", dest="survsel", help="Survivor Selection method to use.", default="Elitism")
    parser.add_argument("-g", "--ngen", dest="ngen", help="Maximum number of generations.", default=1000, type=int)
    parser.add_argument("-a", "--popsize", dest="popsize", help="Size of the population.", default=100, type=int)
    parser.add_argument("-n", "--nreps", dest="nreps", help="Number of times to run the algorithm.", default=1000, type=int)
    parser.add_argument("-f", "--fileout", dest="fileout", help="Name of the output file.", default=f"result_GA-pygad.csv")
    args = parser.parse_args()

    run_ga(args.objective.lower(), args.dim, args.mut.lower(), args.cross.lower(), args.parentsel.lower(), args.survsel.lower(), args.ngen, args.popsize, args.nreps, args.fileout)
