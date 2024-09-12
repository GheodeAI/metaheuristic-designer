import metaheuristic_designer as mhd
import matplotlib.pyplot as plt
import numpy as np

def run_algorithm():
    params = {
        "stop_cond": "neval",
        # "stop_cond": "time_limit",
        "progress_metric": "neval",
        # "progress_metric": "time_limit",
        "time_limit": 20.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e5,
        "fit_target": 1e-10,
        "patience": 200,
        "verbose": True,
        "v_timer": 0.5,
    }

    # objfunc = mhd.benchmarks.Shaffer1()
    # objfunc = mhd.benchmarks.Kursawe()
    objfunc = mhd.benchmarks.FonsecaFleming()
    pop_initializer = mhd.initializers.UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=500)
    mutation_op = mhd.operators.OperatorVector("RandNoise", {"distrib": "Cauchy", "F": 0.001})
    cross_op = mhd.operators.OperatorVector("Multipoint")
    # parent_sel_op = mhd.selectionMethods.ParentSelectionNull()
    # selection_op = mhd.selectionMethods.SurvivorSelectionMulti("non-dominated-sorting")

    search_strategy = mhd.strategies.NSGAII(pop_initializer, mutation_op, cross_op, params={"pmut": 0.1, "pcross": 0.9})
    algorithm = mhd.algorithms.GeneralAlgorithmMulti(objfunc, search_strategy, params=params)
    
    final_population = algorithm.optimize()
    # print([i.fitness for i in final_population])
    fitness_results = np.asarray([objfunc.fitness(i, adjusted=False) for i in final_population])
    plt.scatter(fitness_results[:, 0], fitness_results[:, 1])
    plt.show()

    # .display_report(show_plots=True)

if __name__ == "__main__":
    run_algorithm()