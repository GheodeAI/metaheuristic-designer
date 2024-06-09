import metaheuristic_designer as mhd

def run_algorithm():
    params = {
        "stop_cond": "time_limit",
        "progress_metric": "time_limit",
        "time_limit": 10.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 3e6,
        "fit_target": 1e-10,
        "patience": 200,
        "verbose": True,
        "v_timer": 0.5,
    }

    objfunc = mhd.benchmarks.FonsecaFleming()
    pop_initializer = mhd.initializers.UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)
    mutation_op = mhd.operators.OperatorReal("RandNoise", {"distrib": "Cauchy", "F": 0.0001})
    cross_op = mhd.operators.OperatorReal("2point")
    # parent_sel_op = mhd.selectionMethods.ParentSelectionNull()
    # selection_op = mhd.selectionMethods.SurvivorSelectionMulti("non-dominated-sorting")

    search_strategy = mhd.strategies.NSGAII(pop_initializer, mutation_op, cross_op, params={"pmut": 0.1, "pcross": 0.9})
    algorithm = mhd.algorithms.GeneralAlgorithmMulti(objfunc, search_strategy, params=params)
    
    final_population = algorithm.optimize()
    print([i.fitness for i in final_population])
    # .display_report(show_plots=True)

if __name__ == "__main__":
    run_algorithm()