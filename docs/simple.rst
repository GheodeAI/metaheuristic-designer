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
     - | ``random_search_real``,
       | ``random_search_binary``,
       | ``random_search_discrete``,
       | ``random_search_permutation``
     - Independent random sampling each generation.
   * - Hill Climbing
     - | ``hill_climb_real``,
       | ``hill_climb_binary``,
       | ``hill_climb_discrete``,
       | ``hill_climb_permutation``
     - Single‑solution greedy local optimisation.
   * - Local Search
     - | ``local_search_real``,
       | ``local_search_binary``,
       | ``local_search_discrete``,
       | ``local_search_permutation``
     - Multiple perturbations around a single solution per iteration.
   * - Simulated Annealing
     - | ``simulated_annealing_real``,
       | ``simulated_annealing_binary``,
       | ``simulated_annealing_discrete``,
       | ``simulated_annealing_permutation``
     - Accepts worse solutions with a decaying temperature.
   * - Evolution Strategy (ES)
     - | ``evolution_strategy_real``,
       | ``evolution_strategy_binary``,
       | ``evolution_strategy_discrete``,
       | ``evolution_strategy_permutation``
     - (μ+λ) / (μ,λ) strategy with mutation, optional crossover.
   * - Genetic Algorithm (GA)
     - | ``genetic_algorithm_real``,
       | ``genetic_algorithm_binary``,
       | ``genetic_algorithm_discrete``,
       | ``genetic_algorithm_permutation``
     - Tournament selection, crossover, mutation, elitism.
   * - Differential Evolution (DE)
     - | ``differential_evolution_real``,
       | ``differential_evolution_binary``,
       | ``differential_evolution_discrete``
     - Population‑based method using difference vectors. No permutation variant.
   * - Particle Swarm (PSO)
     - | ``particle_swarm_real``,
       | ``particle_swarm_binary``,
       | ``particle_swarm_discrete``
     - Swarm intelligence with inertia weight and acceleration coefficients. No permutation variant.
   * - Bayesian Optimisation
     - ``bayesian_optimization_real``
     - Gaussian‑process surrogate with Expected Improvement. Only continuous.

Example
-------

The following snippet minimises the 5‑dimensional Sphere function using a
real‑coded Genetic Algorithm:

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