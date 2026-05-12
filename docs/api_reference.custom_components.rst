.. _custom-components:

Extending the Framework with Custom Components
==============================================

You can provide every major component of an optimisation algorithm as a plain Python
function, wrapped in dedicated ``*FromLambda`` classes. For operators and selection
methods there are also **factories** that let you register your function and then
retrieve it by name.

Throughout this guide the following type aliases are used for clarity:

* :py:type:`MatrixLike<metaheuristic_designer.utils.MatrixLike>` – a 2‑D NumPy array
  (``n_individuals × n_vars``).
* :py:type:`VectorLike<metaheuristic_designer.utils.VectorLike>` – a 1‑D NumPy array
  (fitness values).
* :py:type:`RNGLike<metaheuristic_designer.utils.RNGLike>` – a NumPy
  :class:`~numpy.random.Generator` or a seed.

All examples assume **maximisation** (higher fitness is better); if your problem
minimises, set ``mode="min"`` in the objective function.

Objective Function
------------------

Wrap an evaluation function with
:py:class:`ObjectiveFromLambda<metaheuristic_designer.objective_function.ObjectiveFromLambda>`.

.. code-block:: python

   def my_obj(solution: Any, **kwargs) -> float | np.ndarray:
       ...
       return fitness_value

* ``solution`` – the decoded individual (any Python object).
* Must return a single numeric value.
* Any extra keyword arguments given to the constructor are forwarded as ``**kwargs``.

.. code-block:: python

   from metaheuristic_designer import ObjectiveFromLambda

   def sphere(vec, offset=0):
       return -np.sum((vec - offset) ** 2)   # maximise negative squared distance

   objfunc = ObjectiveFromLambda(sphere, dimension=3, offset=3.0, mode="max")

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

   def my_gen(random_state: RNGLike, **kwargs) -> np.ndarray:
       """Return a single new individual (genotype vector)."""
       ...

* The function is called once per individual; it receives a random state and must
  return a 1‑D array.

.. code-block:: python

   from metaheuristic_designer import InitializerFromLambda

   def uniform_gen(random_state, low=0.0, high=1.0):
       return random_state.uniform(low, high, size=5)

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
             random_state: RNGLike, **kwargs) -> MatrixLike:
       """Return a new genotype matrix of the same shape."""
       ...

* ``matrix`` – 2‑D array (individuals × variables).
* ``fitness`` – 1‑D array of current fitness values.
* The function must return a **new** matrix; do **not** modify the input in place.

.. code-block:: python

   from metaheuristic_designer.operators import add_operator_entry, OperatorFnDef, create_operator

   @OperatorFnDef
   def add_gaussian_noise(matrix, fitness, random_state, F=0.1):
       rng = np.random.default_rng(random_state)
       noise = rng.normal(0, F, size=matrix.shape)
       return matrix + noise

   # Register the operator – you must wrap it in OperatorFnDef
   add_operator_entry(add_gaussian_noise, "my_noise", "custom")
   op = create_operator("custom.my_noise", F=0.3)

Population‑level (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to access or modify the whole :py:class:`Population` object, provide a
function that receives and returns a :class:`Population`. Make a copy at the
beginning; never mutate the original.

.. code-block:: python

   def my_pop_op(population: Population, initializer: Initializer,
                 random_state: RNGLike, **kwargs) -> Population:
       pop_copy = copy(population)   # or population.__copy__()
       # … modify pop_copy …
       return pop_copy

Register it **without** a wrapper:

.. code-block:: python

   from metaheuristic_designer.operators import add_operator_entry
    
   def duplicate_best(population, initializer, random_state):
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
   @ParentSelectionDef
   def my_parent_select(fitness: VectorLike, amount: int,
                        random_state: RNGLike, **kwargs) -> np.ndarray:
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

   @ParentSelectionDef
   def pick_top_k(fitness, amount, random_state, **kwargs):
       # Maximisation: higher fitness is better → use argpartition for top k
       top_idx = np.argpartition(-fitness, amount - 1)[:amount]
       return top_idx

   add_parent_selection_entry(pick_top_k, "top_k")
   sel = create_parent_selection("top_k", amount=20)

**Direct pathway (Population‑level)**

.. code-block:: python

   from metaheuristic_designer import ParentSelectionFromLambda

   def pop_level_select(population: Population, amount: int,
                        random_state: RNGLike, **kwargs) -> np.ndarray:
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

   @SurvivorSelectionDef
   def my_survivor_select(parent_fitness: VectorLike,
                          offspring_fitness: VectorLike,
                          random_state: RNGLike, **kwargs) -> np.ndarray:
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
   
   @SurvivorSelectionDef
   def keep_all_offspring(parent_fit, offspring_fit, random_state, **kwargs):
       n_parents = len(parent_fit)
       n_offspring = len(offspring_fit)
       return np.arange(n_parents, n_parents + n_offspring)

   add_survivor_selection_entry(keep_all_offspring, "all_offspring")
   ss = create_survivor_selection("all_offspring")

**Direct pathway (Population‑level)**

.. code-block:: python

   from metaheuristic_designer import SurvivorSelectionFromLambda
   from metaheuristic_designer.population import Population

   def pop_level_survivor(parents: Population, offspring: Population,
                          random_state: RNGLike, **kwargs) -> np.ndarray:
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
   def small_noise_vector(row, scale=0.1, random_state=None):
       rng = np.random.default_rng(random_state)
       return row + rng.normal(0, scale, size=row.shape)

   # Now small_noise_vector can be used as a matrix‑level operator function
   from metaheuristic_designer.operators import OperatorFnDef, add_operator_entry

   add_operator_entry(OperatorFnDef(small_noise_vector), "tiny_noise", "custom")
   op = create_operator("custom.tiny_noise", scale=0.05, random_state=42)
