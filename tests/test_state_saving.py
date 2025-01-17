import metaheuristic_designer as mhd

if __name__ == "__main__":
    pop_size = 5
    vecsize = 10

    objfunc = mhd.benchmarks.Sphere(vecsize)

    initializer = mhd.initializers.UniformVectorInitializer(vecsize, 0, 1, pop_size)
    population = initializer.generate_population(objfunc)
    print(population.get_state())

    operator = mhd.operators.OperatorVector("MutRand", {"distrib": "Gauss", "F": 0.1})
    print(operator.get_state())

    parent_sel = mhd.selectionMethods.ParentSelection("SUS", {"method": "linrank"})
    print(parent_sel.get_state())

    survivor_sel = mhd.selectionMethods.SurvivorSelection("Elitism", {"amount": "20"})
    print(survivor_sel.get_state())

    strategy = mhd.strategies.ES(initializer, operator, parent_sel=parent_sel, survivor_sel=survivor_sel)
    algoritm = mhd.algorithms.GeneralAlgorithm(objfunc, strategy)
    algoritm.initialize()

    print(strategy.get_state(True))
    print(algoritm.get_state(True, True, True))
