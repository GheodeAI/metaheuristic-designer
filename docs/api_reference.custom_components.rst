.. _custom_components:

Custom Components
=================

We saw in the :doc:`quick start <api_reference.quick_start>` how to implement
algorithms using wrappers around plain functions. In this tutorial, we are going
to see what is needed to implement each of the components as standalone classes
so we can implement more complex logic and make each component hold internal
state.

Throughout this guide the following type aliases are used for clarity:

* :py:type:`MatrixLike<metaheuristic_designer.utils.MatrixLike>` – a 2‑D NumPy array
  (``n_individuals × n_vars``).
* :py:type:`VectorLike<metaheuristic_designer.utils.VectorLike>` – a 1‑D NumPy array
  (fitness values).
* :py:type:`RNGLike<metaheuristic_designer.utils.RNGLike>` – a NumPy
  :class:`~numpy.random.Generator` or a seed.

All examples assume **maximization** (higher fitness is better); if your problem
minimizes, set ``mode="min"`` in the objective function.

Random Number Generators
------------------------

In this library, reproducibility is enforced by passing a numpy ``Generator`` (``numpy.random.Generator`` subclass)
instance to each function that needs to generate random numbers. Instead of creating a generator from scratch, we
allow the use of seeds as integers. If you want to allow this functionality in your custom implementations
we provide a ``check_rng`` utility that normalizes the random generator object.

This function accepts either a numerical seed like ``42``, a random ``Generator`` or ``None`` and returns
a properly initialized ``Generator``. Note that when ``None`` is passed, a new completely random generator
is produced that will use an arbitrary seed.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer.utils import check_rng

    # Completely random Generator
    rng = check_rng(None)

    # Generator with a seed
    rng = check_rng(42)

    # Generator from a numpy generator
    random_generator = np.random.default_rng(42)
    rng = check_rng(random_generator)


Objective Function
------------------

To implement an objective function, :py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc` we will
make a new class inheriting from the Interface. The only mandatory attribute to indicate is the ``dimension`` and
we will need to provide an implementation of the ``objective`` function. When using this implementation, you are also
allowed to specify a name by the ``name`` attribute.

The ``objective`` function will take as input a decoded solution (with the type outputted by the corresponding 
``decoder`` method in the :py:class:`~metaheuristic_designer.encoding.Encoding`) and will output a single ``float``
number. Once this method is implemented, the base class provides the ``calculate_fitness`` method which will evaluate the 
solutions present in a :py:class:`~metaheuristic_designer.population.Population` object using the specified objective.

.. note::
    By default, objective function are not ``vectorized``, which means that the objective is computed individually
    for each solution. If possible, it is recommended that the attribute ``vectorized`` is set to ``True`` and the
    objective is implemented by taking a matrix of solutions and outputting a vector of objective values.

Here is an example of a concrete :py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc`. 

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import ObjectiveFunc
    from metaheuristic_designer.utils import check_rng

    class GaussianPeak(ObjectiveFunc):
        def __init__(self, dimension: int):
            super().__init__(dimension, name="Gaussian Peak")
        
        def objective(self, solution):
            objective_value = np.exp(-np.sum(solution ** 2))
            return objective_value
    
    objfunc = GaussianPeak(2)
    population = Population(np.random.uniform(0, 1, (3, 2)))
    population = objfunc.calculate_fitness(population) # Compute fitness vector inside the population
    print(population.objective)

We should note that functions are maximized by default, if we want to minimize the function, we can specify
that the ``mode`` attribute is set to ``"min"``. We can also implement the function in vectorized form.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import ObjectiveFunc, Population
    from metaheuristic_designer.utils import check_rng

    class NoisySphereVectorized(ObjectiveFunc):
        def __init__(self, dimension, sigma=1e-5, rng=None):
            super().__init__(dimension, vectorized=True, name="Min Noisy Sphere", mode="min")
            self.rng = check_rng(rng)
            self.sigma = sigma
        
        def objective(self, solution):
            objective_vector = np.sum(solution ** 2, axis=1)
            noise_vector = self.rng.normal(0, self.sigma, size=solution.shape[0])
            return objective_vector + noise_vector

    objfunc = NoisySphereVectorized(5)
    population = Population(np.random.uniform(0, 1, (3, 5)))
    population = objfunc.calculate_fitness(population) # Compute fitness vector inside the population
    print(population.objective)


Constraint Handler
------------------

Use :py:class:`ConstraintHandlerFromLambda<metaheuristic_designer.constraint_handler.ConstraintHandlerFromLambda>`.
At least one of the two possible callables must be provided.

.. code-block:: python

   def repair_fn(solution: Any) -> Any:
       """Return a repaired copy of the solution."""
       ...

   def penalty_fn(solution: Any) -> float:
       """Return a penalty that will be subtracted from the fitness."""
       ...

.. code-block:: python

   from metaheuristic_designer import ConstraintHandlerFromLambda

   def clip_to_bounds(x, low=-5.0, high=5.0):
       return np.clip(x, low, high)

   ch = ConstraintHandlerFromLambda(repair_solution_fn=clip_to_bounds, low=-5, high=5)

Initializer
-----------

Wrap a generator function with
:py:class:`InitializerFromLambda<metaheuristic_designer.initializer.InitializerFromLambda>`.

.. code-block:: python

   def my_gen(rng: RNGLike, **kwargs) -> np.ndarray:
       """Return a single new individual (genotype vector)."""
       ...

* The function is called once per individual; it receives a random state and must
  return a 1‑D array.

.. code-block:: python

   from metaheuristic_designer import InitializerFromLambda

   def uniform_gen(rng, low=0.0, high=1.0):
       return rng.uniform(low, high, size=5)

   init = InitializerFromLambda(uniform_gen, dimension=5, pop_size=100, low=-10, high=10, size=5)

Encoding
--------

Wrap an encode/decode pair with
:py:class:`EncodingFromLambda<metaheuristic_designer.encoding.EncodingFromLambda>`.

.. code-block:: python

   def my_encode(solutions: Iterable) -> MatrixLike:
       """Encode a list of solutions into a genotype matrix (2-D array)."""
       ...

   def my_decode(population_matrix: MatrixLike) -> Iterable:
       """Decode the whole genotype matrix into a list/array of solutions."""
       ...

.. code-block:: python

   from metaheuristic_designer import EncodingFromLambda

   def to_ints(real_vec):
       return np.floor(real_vec).astype(int)

   def to_reals(int_matrix):
       return int_matrix.astype(float)

   enc = EncodingFromLambda(encode_fn=to_ints, decode_fn=to_reals)

Operators
---------

There are two ways to write a custom operator, depending on the level of control you
need.

Matrix‑level (recommended for most cases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a function that works on the raw NumPy arrays and wrap it with
:py:class:`OperatorFnDef<metaheuristic_designer.operators.operator_functions.utils.OperatorFnDef>`. The wrapper handles
population bookkeeping (extracting the genotype matrix, fitness, and updating a copy
of the population).

.. code-block:: python

   def my_op(matrix: MatrixLike, fitness: VectorLike,
             rng: RNGLike, **kwargs) -> MatrixLike:
       """Return a new genotype matrix of the same shape."""
       ...

* ``matrix`` – 2‑D array (individuals × variables).
* ``fitness`` – 1‑D array of current fitness values.
* The function must return a **new** matrix; do **not** modify the input in place.

.. code-block:: python

   from metaheuristic_designer.operators import add_operator_entry, OperatorFnDef, create_operator

   def add_gaussian_noise(matrix, fitness, rng, F=0.1):
       rng = np.random.default_rng(rng)
       noise = rng.normal(0, F, size=matrix.shape)
       return matrix + noise

   # Register the operator – you must wrap it in OperatorFnDef
   add_operator_entry(OperatorFnDef(add_gaussian_noise), "my_noise", "custom")
   op = create_operator("custom.my_noise", F=0.3)

Population‑level (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to access or modify the whole :py:class:`Population` object, provide a
function that receives and returns a :class:`Population`. Make a copy at the
beginning; never mutate the original.

.. code-block:: python

   def my_pop_op(population: Population, rng: RNGLike, **kwargs) -> Population:
       pop_copy = copy(population)   # or population.__copy__()
       # … modify pop_copy …
       return pop_copy

Register it **without** a wrapper:

.. code-block:: python

   from metaheuristic_designer.operators import add_operator_entry
    
   def duplicate_best(population, rng):
       pop_copy = copy(population)
       best_gen = pop_copy.genotype_matrix[pop_copy.best_idx]
       pop_copy.genotype_matrix[:] = best_gen   # all individuals become the best
       return pop_copy

   add_operator_entry(duplicate_best, "dup_best", "custom")
   op = create_operator("custom.dup_best")

Parent Selection
----------------

The factory :py:func:`create_parent_selection<metaheuristic_designer.parent_selection_methods.parent_selection.create_parent_selection>`
expects a function that works on fitness arrays. If you need the whole population,
instantiate :py:class:`ParentSelectionFromLambda<metaheuristic_designer.parent_selection.ParentSelectionFromLambda>` directly.

**Factory pathway (fitness‑level)**

.. code-block:: python

   def my_parent_select(fitness: VectorLike, amount: int,
                        rng: RNGLike, **kwargs) -> np.ndarray:
       """Return indices of selected individuals."""
       ...

* ``fitness`` – the current fitness values.
* ``amount`` – how many individuals to select.
* Must return a 1‑D integer array (no duplicates).

For it to be accepted into the registry, it must be passed to the :py:class:`~metaheuristic_designer.parent_selection.ParentSelectionDef` wrapper since
the :py:class:`~metaheuristic_designer.parent_selection.ParentSelection` class works directly with :py:class:`metaheuristic_designer.population.Population` objects. This can be easily done by using 
:py:class:`~metaheuristic_designer.parent_selection.ParentSelectionDef` as a decorator.

.. code-block:: python

   from metaheuristic_designer.parent_selection_methods import add_parent_selection_entry, ParentSelectionDef
   from metaheuristic_designer import create_parent_selection

   def pick_top_k(fitness, amount, rng, **kwargs):
       # Maximisation: higher fitness is better → use argpartition for top k
       top_idx = np.argpartition(-fitness, amount - 1)[:amount]
       return top_idx

   add_parent_selection_entry(ParentSelectionDef(pick_top_k), "top_k")
   sel = create_parent_selection("top_k", amount=20)

**Direct pathway (Population‑level)**

.. code-block:: python

   from metaheuristic_designer import ParentSelectionFromLambda

   def pop_level_select(population: Population, amount: int,
                        rng: RNGLike, **kwargs) -> np.ndarray:
       # Access population.genotype_matrix, population.fitness, etc.
       fitness = population.fitness
       top_idx = np.argpartition(-fitness, amount - 1)[:amount]
       return top_idx

   sel = ParentSelectionFromLambda(pop_level_select, amount=20)

Survivor Selection
------------------

Similarly, the factory :py:func:`create_survivor_selection<metaheuristic_designer.survivor_selection_methods.survivor_selection.create_survivor_selection>`
works with fitness‑level functions, while direct instantiation of
:py:class:`SurvivorSelectionFromLambda<metaheuristic_designer.survivor_selection.SurvivorSelectionFromLambda>` gives access
to the Population objects.

**Factory pathway (fitness‑level)**

.. code-block:: python

   def my_survivor_select(parent_fitness: VectorLike,
                          offspring_fitness: VectorLike,
                          rng: RNGLike, **kwargs) -> np.ndarray:
       """Return indices into the concatenated [parents, offspring]."""
       ...

* ``parent_fitness`` – fitness of the parent population.
* ``offspring_fitness`` – fitness of the offspring.
* The returned indices refer to the array obtained by joining parents and offspring.

For it to be accepted into the registry, it must be passed to the :py:class:`~metaheuristic_designer.parent_selection.SurvivorSelectionDef` wrapper since
the :py:class:`~metaheuristic_designer.parent_selection.SurvivorSelection` class works directly with :py:class:`metaheuristic_designer.population.Population` objects.
This can be easily done by using :py:class:`~metaheuristic_designer.parent_selection.SurvivorSelectionDef` as a decorator.

.. code-block:: python

   from metaheuristic_designer.survivor_selection_methods import add_survivor_selection_entry
   from metaheuristic_designer import create_survivor_selection
   
   def keep_all_offspring(parent_fit, offspring_fit, rng, **kwargs):
       n_parents = len(parent_fit)
       n_offspring = len(offspring_fit)
       return np.arange(n_parents, n_parents + n_offspring)

   add_survivor_selection_entry(SurvivorSelectionDef(keep_all_offspring), "all_offspring")
   ss = create_survivor_selection("all_offspring")

**Direct pathway (Population‑level)**

.. code-block:: python

   from metaheuristic_designer import SurvivorSelectionFromLambda
   from metaheuristic_designer.population import Population

   def pop_level_survivor(parents: Population, offspring: Population,
                          rng: RNGLike, **kwargs) -> np.ndarray:
       # Compute fitness arrays, decide survivors
       combined_fit = np.concatenate([parents.fitness, offspring.fitness])
       n = len(parents)
       return np.argpartition(-combined_fit, n - 1)[:n]

   ss = SurvivorSelectionFromLambda(pop_level_survivor)

Utility Decorators for Row‑wise Operations
------------------------------------------

When writing custom operators (or other matrix‑level functions) it is often convenient to
think in terms of “apply this function to each individual’s row”. Two decorators from
:py:mod:`metaheuristic_designer.utils` make this trivial:

* :py:func:`per_individual<metaheuristic_designer.utils.per_individual>` – wraps a function that
  operates on a **single row** (1‑D array) so that it can be called on a 2‑D matrix and
  returns a 2‑D matrix of the same shape.
* :py:func:`per_individual_list<metaheuristic_designer.utils.per_individual_list>` – same idea,
  but works on a list of objects (e.g. decoded solutions), returning a list.

.. code-block:: python

   from metaheuristic_designer.utils import per_individual

   @per_individual
   def small_noise_vector(row, scale=0.1, rng=None):
       rng = np.random.default_rng(rng)
       return row + rng.normal(0, scale, size=row.shape)

   # Now small_noise_vector can be used as a matrix‑level operator function
   from metaheuristic_designer.operators import OperatorFnDef, add_operator_entry

   add_operator_entry(OperatorFnDef(small_noise_vector), "tiny_noise", "custom")
   op = create_operator("custom.tiny_noise", scale=0.05, rng=42)
