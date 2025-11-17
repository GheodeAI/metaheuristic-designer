import numpy as np
import metaheuristic_designer as mhd

pso_encoding = mhd.encodings.PSOEncoding(10)

initializer = mhd.ExtendedInitializer(
    solution_init=mhd.initializers.UniformInitializer(10, -100, 100, pop_size=100),
    param_init_dict={"speed": mhd.initializers.UniformInitializer(10, -10, 10)},
    encoding=pso_encoding,
)

constraint_handler = mhd.ExtendedConstraintHandler(
    mhd.constraint_handlers.ClipBoundConstraint(10, -100, 100),
    {"speed": mhd.constraint_handlers.BounceBoundConstraint(10, -100, 100)},
    encoding=pso_encoding,
)
# constraint_handler = None

objfunc = mhd.benchmarks.Sphere(10, constraint_handler=constraint_handler)

operator = mhd.operators.VectorOperator("PSO", {"w": 0.9, "c1":0.9, "c2":0.9})

strategy = mhd.strategies.PSO(initializer)

algorithm = mhd.algorithms.GeneralAlgorithm(
    objfunc,
    strategy,
    {"stop_cond": "time_limit", "time_limit": 3}
)

single_vector = initializer.generate_random()
# print(single_vector)

population = initializer.generate_population(objfunc)
# print(population)

population = population.calculate_fitness()
# print(population)

population_decoded = pso_encoding.decode(population.genotype_matrix)
# print(population_decoded)

params_decoded = pso_encoding.decode_params(population.genotype_matrix)
# print(params_decoded)

encoded = pso_encoding.encode(population_decoded, params_decoded)
# print(encoded)

assert np.all(population.genotype_matrix == encoded)

# print(population.calculate_fitness())

pso_mutated_population = operator.evolve(population)
# print(pso_mutated_population)

initial_population = strategy.initialize(objfunc)
# print(initial_population)

initial_population = strategy.evaluate_population(initial_population)
# print(initial_population)

selected_parents = strategy.select_parents(initial_population)
# print(selected_parents)

evolved_population = strategy.perturb(selected_parents)
# print(evolved_population)

next_population = strategy.select_individuals(initial_population, evolved_population)
# print(next_population)

algorithm.optimize()

algorithm.display_report()


