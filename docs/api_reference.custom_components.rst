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

Parameter Schedules
-------------------

Many interfaces used in this library implement the :py:class:`~metaheuristic_designer.parametrizable_mixin.ParametrizableMixin`.
in particular, these are every base class that implements this mixin:

.. list-table::
    :width: 0.2

    * - :py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc`
    * - :py:class:`~metaheuristic_designer.constraint_handler.ConstraintHandler`
    * - :py:class:`~metaheuristic_designer.encoding.Encoding`
    * - :py:class:`~metaheuristic_designer.parent_selection_base.ParentSelection`
    * - :py:class:`~metaheuristic_designer.operator.Operator`
    * - :py:class:`~metaheuristic_designer.survivor_selection_base.SurvivorSelection`
    * - :py:class:`~metaheuristic_designer.search_strategy.SearchStrategy`

Any of these classes accept keyword arguments with any value or an implementation of
:py:class:`~metaheuristic_designer.schedulable_parameter.SchedulableParameter`. To access these parameters, we can access
the ``params`` attribute that will hold the concrete value of the parameter. Let's see an example.

First, we'll create a custom :py:class:`~metaheuristic_designer.schedulable_parameter.SchedulableParameter` by sub-classing it
and implementing the ``evaluate`` method, which gets a progress value between 0 and 1, and returns the value of the parameter.

.. code-block:: python

    from metaheuristic_designer import SchedulableParameter

    class ProgressProportional(SchedulableParameter):
        def __init__(self, value):
            super().__init__()
            self.value = value
        
        def evaluate(self, progress):
            return self.value * progress

With this new parameter schedule, we can use it as a key-word argument in any of the corresponding class and it will become a parameter
that depends on the progress value of the algorithm. As an example, we can have a custom class that implement
:py:class:`~metaheuristic_designer.parametrizable_mixin.ParametrizableMixin`. And we can show how to work with its parameters.
We also note that the ``update`` function is called automatically each iteration in the inner loop of the algorithm. Parameters are
initialized to a progress value of 0.

.. code-block:: python

    from metaheuristic_designer import ParametrizableMixin

    class CustomClass(ParametrizableMixin):
        def __init__(self, a, b):
            super().__init__(a=a, b=b)
    
    b_sched = ProgressProportional(10) 
    c = CustomClass(a = 1, b = b_sched)

    print(c.params.a, c.params.b) # Prints: 1, 0
    c.update(0.5)
    print(c.params.a, c.params.b) # Prints: 1, 5


Objective Function
------------------

To implement an objective function from the abstract :py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc` 
class, we will make a new class inheriting from the Interface. The only mandatory attribute to indicate is the
``dimension`` and we will need to provide an implementation of the ``objective`` function. When using this
implementation, you are also allowed to specify a name by the ``name`` attribute.

The ``objective`` function will take as input a decoded solution (with the type outputted by the corresponding 
``decoder`` method in the :py:class:`~metaheuristic_designer.encoding.Encoding`) and will output a single ``float``
number. Once this method is implemented, the base class provides the ``calculate_fitness`` method which will evaluate the 
solutions present in a :py:class:`~metaheuristic_designer.population.Population` object using the specified objective.

.. note::
    By default, objective functions are not ``vectorized``, which means that the objective is computed individually
    for each solution. If possible, it is recommended that the attribute ``vectorized`` is set to ``True`` and the
    objective is implemented by taking a matrix of solutions and outputting a vector of objective values.

Here is an example of a concrete :py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc`. 

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import ObjectiveFunc, Population
    from metaheuristic_designer.utils import check_rng

    class GaussianPeak(ObjectiveFunc):
        def __init__(self, dimension: int):
            super().__init__(dimension=dimension, name="Gaussian Peak")
        
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
            super().__init__(dimension=dimension, vectorized=True, name="Min Noisy Sphere", mode="min")
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

Constraint Handlers are objects that are meant to be embedded into an objective function and 
work by enforcing the constraints of the problem with two different mechanisms, penalties
and solution repairing.

Both mechanisms are mutually exclusive, repairing a solution means penalties will be always 0, so
we cannot have both at the same time. For this purpose, we will have two different interfaces to
implement :py:class:`~metaheuristic_designer.constraint_handler.ConstraintHandler` objects.

If we want to create a repairing procedure, we will make use of the
:py:class:`~metaheuristic_designer.constraint_handler.RepairConstraint` abstract class. This way,
we will be implementing a ``repair_solutions`` method that gets a matrix representing the solutions
of our problem and must return an identically sized matrix of repaired solutions.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import RepairConstraint

    class NormalizeRepairing(RepairConstraint):
        def __init__(self, norm=1):
            super().__init__()
            self.norm = norm
        
        def repair_solutions(self, solutions):
            current_norm = np.linalg.norm(solutions, axis=1)
            
            return (self.norm/current_norm) * solutions

Alternatively, if we want to specify a penalty method, we will use the :py:class:`~metaheuristic_designer.constraint_handler.PenalizeConstraint`
abstract class. We will only need to implement the ``penalty`` method which gets an collection of solutions and must
return a vector containing the penalty value for each of the solutions.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import PenalizeConstraint

    class NormPenaltyConstraint(PenalizeConstraint):
        def __init__(self, max_norm=1):
            # The base class stores keyword arguments in self.params
            super().__init__(max_norm=max_norm)
        
        def penalty(self, solutions):
            penalties = np.zeros(len(solutions))

            for idx, s in enumerate(solutions):
                s_norm = np.linalg.norm(s)
                if s_norm > self.params.max_norm:
                    penalties[idx] = s_norm - 1
            
            return penalties

Encoding
--------

If we need a custom encoding, we can subclass the interface :py:class:`~metaheuristic_designer.encoding.Encoding`, this new
class should implement both an ``encode`` and ``decode`` function. The ``encode`` function will take a collection of solutions
in their natural representation, and will return a numpy matrix with size (NxM) where N is the number of solutions and 
M is the number of components of each solution. The ``decode`` function will take that matrix with the internal solution
representation and return a collection of solutions in their natural representation. Sometimes the ``decode`` function is not
completely reversible, in this case, it is completely acceptable to make the ``encode`` function return a sentinel value, since it 
will very rarely be used in the actual optimization procedure.

.. code-block:: python

    from metaheuristic_designer import Encoding

    class MulticategoricalEncoding(Encoding):
        def __init__(self, categories: list):
            self.categories = categories
            self.n_vars = len(categories)
            super().__init__()
        
        def encode(self, solutions):
            N = len(solutions)
            encoded = np.empty((N, self.n_vars), dtype=int)
            for i, sol in enumerate(solutions):
                for j, val in enumerate(sol):
                    encoded[i, j] = self.categories[j].index(val)
            return encoded
        
        def decode(self, population_matrix):
            solutions = []
            for i in range(population_matrix.shape[0]):
                sol = tuple(self.categories[j][idx] for j, idx in enumerate(population_matrix[i]))
                solutions.append(sol)
            return solutions

    enc = MultiCategoricalEncoding(
        [('red', 'blue', 'green'), ('small', 'large'), ('A', 'B')]
    )

    solutions = [('red', 'small', 'B'), ('green', 'large', 'A')]

    population_matrix = enc.encode(solutions)
    print(encoded)   # [[0 0 1]
                     #  [2 1 0]]

    decoded = enc.decode(population_matrix)
    print(decoded)   # [('red', 'small', 'A'), ('blue', 'large', 'B')]

Initializer
-----------

When creating an initializer, we will make use of the :py:class:`~metaheuristic_designer.initializer.Initializer` class.
To implement a concrete class, we will have to implement at least the `generate_random` method, that will
generate a single random vector. We should pass a ``dimension`` to the parent constructor and accept a 
``population_size`` argument as well. Initializers have an integrated ``rng`` handler, so we should pass it as an
argument. We will also handle the encoding passing through the initializer to ensure the population is generated
with the correct encoding.


.. code-block:: python

    from metaheuristic_designer import Initializer, Encoding

    class CategoricalInitializer(Initializer):
        def __init__(self, dimension: int, values: list, population_size: int, encoding: Encoding = None, rng = None):
            super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)

            self.values = values
        
        def generate_random(self):
            return self.rng.choice(self.values, self.dimension)


If we are intending to make a deterministic initializer, it is recommended to use a fallback uniform
initializer, and implement the actual logic in the ``generate_individual`` method.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import Initializer, Encoding

    class ConstantInitializer(Initializer):
        def __init__(self, dimension: int, value: float, population_size: int, fallback: Initializer, encoding: Encoding = None, rng = None):
            super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)

            self.value = value
            self.fallback = fallback
        
        def generate_random(self):
            return self.fallback.generate_random()
        
        def generate_individual(self):
            return np.full(self.dimension, self.value)

Sometimes, an initializer will need to generate the entire population in one go. For this purpose, we will
override the ``generate_population`` method, which will be given a number of individuals to generate 
``n_individuals`` that will often fall back to the specified population size.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import Initializer, Encoding, Population

    class IdentityMatrixInitializer(Initializer):
        def __init__(self, dimension: int, fallback: Initializer, encoding: Encoding = None, rng = None):
            super().__init__(dimension=dimension, population_size=dimension, encoding=encoding, rng=rng)
            self.fallback = fallback
        
        def generate_random(self):
            return self.fallback.generate_random()
        
        def generate_population(self, n_individuals = None):
            if n_individuals is None:
                n_individuals = self.population_size
            
            assert n_individuals <= self.dimension, f"Can only generate up to {self.dimension} individuals"
            
            id_matrix = np.eye(self.dimension)[:n_individuals]
            return Population(id_matrix, encoding=self.encoding)

Operators
---------


Parent Selection
----------------


Survivor Selection
------------------


Search strategy
---------------


Reporter
--------

History tracker
---------------

Stopping condition
------------------

Checkpointer
------------

