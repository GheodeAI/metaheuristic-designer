.. _simple-api:

Simple subpackage
=================

The :mod:`metaheuristic_designer.simple` module contains ready‑to‑run functions
that return a configured :class:`~metaheuristic_designer.algorithm.Algorithm`.
All functions follow a strict naming pattern so you can call any algorithm
without looking up a separate catalogue.

Naming convention
-----------------

Every function is named like this:

.. code-block:: text

    simple.<algorithm>_<encoding>(objfunc, **kwargs) # kwargs passed to Algorithm

* ``<algorithm>`` – the algorithm family (e.g. ``genetic_algorithm``).
* ``<encoding>`` – the problem representation you want to use:

  * ``real`` – continuous variables
  * ``binary`` – bit‑strings
  * ``discrete`` – integer variables
  * ``permutation`` – permutations of integers

The table below lists every algorithm family, the encodings it supports, and
the corresponding function names.  You can copy any function name directly
into your code.

Available algorithms
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Algorithm family
     - Function names ("simple." prefix omitted)
     - Short description
   * - Random Search
     - | :py:class:`~metaheuristic_designer.simple.random_search_real`,
       | :py:class:`~metaheuristic_designer.simple.random_search_binary`,
       | :py:class:`~metaheuristic_designer.simple.random_search_discrete`,
       | :py:class:`~metaheuristic_designer.simple.random_search_permutation`
     - Independent random sampling each generation.
   * - Hill Climbing
     - | :py:class:`~metaheuristic_designer.simple.hill_climb_real`,
       | :py:class:`~metaheuristic_designer.simple.hill_climb_binary`,
       | :py:class:`~metaheuristic_designer.simple.hill_climb_discrete`,
       | :py:class:`~metaheuristic_designer.simple.hill_climb_permutation`
     - Single-solution greedy local optimization.
   * - Local Search
     - | :py:class:`~metaheuristic_designer.simple.local_search_real`,
       | :py:class:`~metaheuristic_designer.simple.local_search_binary`,
       | :py:class:`~metaheuristic_designer.simple.local_search_discrete`,
       | :py:class:`~metaheuristic_designer.simple.local_search_permutation`
     - Multiple perturbations around a single solution per iteration.
   * - Simulated Annealing
     - | :py:class:`~metaheuristic_designer.simple.simulated_annealing_real`,
       | :py:class:`~metaheuristic_designer.simple.simulated_annealing_binary`,
       | :py:class:`~metaheuristic_designer.simple.simulated_annealing_discrete`,
       | :py:class:`~metaheuristic_designer.simple.simulated_annealing_permutation`
     - Accepts worse solutions with a decaying temperature.
   * - Evolution Strategy (ES)
     - | :py:class:`~metaheuristic_designer.simple.evolution_strategy_real`,
       | :py:class:`~metaheuristic_designer.simple.evolution_strategy_binary`,
       | :py:class:`~metaheuristic_designer.simple.evolution_strategy_discrete`,
       | :py:class:`~metaheuristic_designer.simple.evolution_strategy_permutation`
     - (μ+λ) / (μ,λ) strategy with mutation, optional crossover.
   * - Genetic Algorithm (GA)
     - | :py:class:`~metaheuristic_designer.simple.genetic_algorithm_real`,
       | :py:class:`~metaheuristic_designer.simple.genetic_algorithm_binary`,
       | :py:class:`~metaheuristic_designer.simple.genetic_algorithm_discrete`,
       | :py:class:`~metaheuristic_designer.simple.genetic_algorithm_permutation`
     - Tournament selection, crossover, mutation, elitism.
   * - Differential Evolution (DE)
     - | :py:class:`~metaheuristic_designer.simple.differential_evolution_real`,
       | :py:class:`~metaheuristic_designer.simple.differential_evolution_binary`,
       | :py:class:`~metaheuristic_designer.simple.differential_evolution_discrete`
     - Population-based method using difference vectors. No permutation variant.
   * - Particle Swarm (PSO)
     - | :py:class:`~metaheuristic_designer.simple.particle_swarm_real`,
       | :py:class:`~metaheuristic_designer.simple.particle_swarm_binary`,
       | :py:class:`~metaheuristic_designer.simple.particle_swarm_discrete`
     - Swarm intelligence with inertia weight and acceleration coefficients. No permutation variant.
   * - Bayesian Optimization
     - :py:class:`~metaheuristic_designer.simple.bayesian_optimization_real`
     - Gaussian-process surrogate with Expected Improvement. Only continuous.

Example
-------

The following snippet minimizes the 5-dimensional Sphere function using a
real-coded Genetic Algorithm:

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

All additional keyword arguments (such as ``max_iterations``, ``reporter``,
``real_time_limit``, …) are passed directly to the :py:class:`~metaheuristic_designer.algorithm.Algorithm`
constructor.  See the :doc:`Algorithm Configuration <api_reference.algorithm_config>`
page for a full description of what you can configure.

Each function also accepts algorithm‑specific hyper‑parameters (mutation
strength, population size, number of mutated components, …).

For a complete
list of available arguments, use Python’s built‑in help, e.g.
``help(simple.genetic_algorithm_real)``.