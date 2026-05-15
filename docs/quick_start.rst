.. _quick-start:

Quick Start
===========

The fastest way to use the library is through the :mod:`~metaheuristic_designer.simple` wrappers.  
Choose your algorithm and problem, set a few hyperparameters, and you’re done.

**Example – genetic algorithm on the 5‑D Sphere function**:

.. code-block:: python

   from metaheuristic_designer.benchmarks import Sphere
   from metaheuristic_designer import simple, check_random_state

   objfunc = Sphere(dimension=5, mode="min")
   rng = check_random_state(42)

   algo = simple.genetic_algorithm_real(
       objfunc,
       population_size=100,
       max_iterations=100,
       reporter="tqdm",
       random_state=rng,
   )
   population = algo.optimize()
   solution, obj = population.best_solution()
   print(f"Best objective: {obj:.6g}")

If you need more control, build the algorithm from components:

.. code-block:: python

   from metaheuristic_designer import Algorithm
   from metaheuristic_designer.strategies import GA
   from metaheuristic_designer.initializers import UniformInitializer
   from metaheuristic_designer.operators import create_operator
   from metaheuristic_designer.parent_selection import create_parent_selection
   from metaheuristic_designer.survivor_selection import create_survivor_selection
   
   objfunc = Sphere(dimension=5, mode="min")
   rng = check_random_state(42)

   strategy = GA(
       initializer = UniformInitializer(
            dimension=objfunc.dimension,
            lower_bound=objfunc.lower_bound,
            upper_bound=objfunc.upper_bound,
            population_size=100, 
            random_state=rng
       ),
       mutation_op = create_operator("mutation.gaussian_mutation", F=0.1, N=1, random_state=rng),
       crossover_op = create_operator("crossover.uniform_crossover", random_state=rng),
       parent_sel = create_parent_selection("tournament", amount=50, tournament_size=3, random_state=rng),
       survivor_sel = create_survivor_selection("elitism", amount=25, random_state=rng),
       mutation_prob = 0.3,
       crossover_prob = 0.9,
       random_state = rng,
   )
   algo = Algorithm(objfunc, strategy, stop_cond="max_iterations", max_iterations=100, reporter="tqdm")
   population = algo.optimize()
   solution, obj = population.best_solution()
   print(f"Best objective: {obj:.6g}")

All the components shown here have many more options.  For a complete
catalogue, see :doc:`Operators and selection methods <api_reference.methods>`.
To learn how to provide your own components, read
:doc:`Custom Components <api_reference.custom_components>`.