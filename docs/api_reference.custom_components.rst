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

    class ProgressProportionalSchedule(SchedulableParameter):
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
            super().__init__()
            self.store_kwargs(a=a, b=b)
    
    b_sched = ProgressProportionalSchedule(value=10) 
    c = CustomClass(a = 1, b = b_sched)

    print(c.params.a, c.params.b) # Prints: 1, 0
    c.update(progress=0.5)
    print(c.params.a, c.params.b) # Prints: 1, 5

In every class that already implements this mixin, the ``store_kwargs`` call is done with the remaining kwargs not used by the class.
The parameters can be also manually reset in runtime with the ``update_kwargs`` method.

.. code-block:: python

    c = CustomClass(a = 1, b = 4)
    print(c.params.a, c.params.b) # Prints: 1, 4

    c.update_kwargs(a = 2, b = ProgressProportionalSchedule(value=6))
    c.update(progress=1)
    print(c.params.a, c.params.b) # Prints: 2, 6

There are also some available schedules that take other schedules as input and modify them in different ways, they are listed in
the :doc:`api_reference <api_reference>` page.

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

You are allowed to chain repairing methods or add the penalty values by using the
:py:class:`~metaheuristic_designer.constraint_handlers.CompositeConstraint` class.

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

    import numpy as np
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

    enc = MulticategoricalEncoding(
        [('red', 'blue', 'green'), ('small', 'large'), ('A', 'B')]
    )

    solutions = [('red', 'small', 'B'), ('green', 'large', 'A')]

    population_matrix = enc.encode(solutions)
    print(population_matrix)   # [[0 0 1]
                               #  [2 1 0]]

    decoded = enc.decode(population_matrix)
    print(decoded)   # [('red', 'small', 'B'), ('green', 'large', 'A')]

You are allowed to chain encodings with the :py:class:`~metaheuristic_designer.encodings.CompositeEncoding`
class. It is important to note that the decoding will be done in the same order as the provided list of encodings,
and the encoding will be done in reverse order.

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
    
    initializer = CategoricalInitializer(dimension=5, values=(0, 4, 10), population_size=3)
    new_population = initializer.generate_population()
    print(new_population)

    new_population = initializer.generate_population(10)
    print(new_population)


If we are intending to make a deterministic initializer, it is recommended to use a fallback uniform
initializer, and implement the actual logic in the ``generate_individual`` method.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import Initializer, Encoding
    from metaheuristic_designer.initializers import UniformInitializer

    class ConstantInitializer(Initializer):
        def __init__(self, dimension: int, value: float, population_size: int, fallback: Initializer, encoding: Encoding = None, rng = None):
            super().__init__(dimension=dimension, population_size=population_size, encoding=encoding, rng=rng)

            self.value = value
            self.fallback = fallback
        
        def generate_random(self):
            return self.fallback.generate_random()
        
        def generate_individual(self):
            return np.full(self.dimension, self.value)

    fallback_init = UniformInitializer(dimension=5, lower_bound=0, upper_bound=10, dtype=int)
    initializer = ConstantInitializer(dimension=5, value=4, population_size=3, fallback=fallback_init)
    print(initializer.generate_random())

    new_population = initializer.generate_population()
    print(new_population)


Sometimes, an initializer will need to generate the entire population in one go. For this purpose, we will
override the ``generate_population`` method, which will be given a number of individuals to generate 
``n_individuals`` that will often fall back to the specified population size.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import Initializer, Encoding, Population
    from metaheuristic_designer.initializers import UniformInitializer

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

    fallback_init = UniformInitializer(dimension=5, lower_bound=0, upper_bound=10, dtype=int)
    initializer = IdentityMatrixInitializer(dimension=5, fallback=fallback_init)

    new_population = initializer.generate_population()
    print(new_population)

You are allowed to combine initializers with the :py:class:`~metaheuristic_designer.initializers.CompositeInitializer`
class, so that individuals will be generated at random with any of the specified initializers. Alternatively, there also is a
:py:class:`~metaheuristic_designer.initializers.FixedCompositeInitializer` that chooses individuals in a deterministic manner.


Operators
---------
If we want to create a custom operator, we can subclass the :py:class:`~metaheuristic_designer.operator.Operator`, implementing
the ``evolve`` method, which gets a :py:class:`~metaheuristic_designer.population.Population` instance, and returns a new Population
with the new solutions.

.. code-block:: python

    from metaheuristic_designer import Operator, Population

    class ModularAdditionMutation(Operator):
        def __init__(self, value: int, mod: int):
            super().__init__(name="Modular addition", preserves_order=True, value=value)
            self.mod = mod
        
        def evolve(self, population: Population):
            new_matrix = (population.genotype_matrix + self.params.value) % self.mod
            new_population = population.update_genotype(new_matrix)
            return new_population
    
    op = ModularAdditionMutation(value = 1, mod = 7)
    population = Population(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    new_population = op.evolve(population)
    print(new_population.genotype_matrix)

There are a number of different ways to combine operators, we recommend checking out the :doc:`methods <api_reference.methods>` page
to see all the available composition strategies.


Parent Selection
----------------
For creating custom Parent Selection strategies, we can subclass :py:class:`~metaheuristic_designer.parent_selection_base.ParentSelection`
and implement the ``select`` method, which takes as input a :py:class:`~metaheuristic_designer.population.Population` and returns a new 
population of selected individuals. It is highly recommended to set the ``last_selection_idx`` attribute while performing the selection,
as it can be of use for other methods, it should be a vector contain only indices of the selected individuals.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import ParentSelection, Population

    class TruncateSelection(ParentSelection):
        def __init__(self, amount: int):
            super().__init__(amount=amount)
        
        def select(self, population: Population, amount: int = None):
            if amount is None:
                amount = self.params.amount

            n_individuals = population.population_size

            self.last_selection_idx = np.arange(n_individuals)[:amount]
            return population.take_selection(self.last_selection_idx)
    
    parent_sel = TruncateSelection(amount=2)
    population = Population(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    new_population = parent_sel.select(population)
    print(new_population.genotype_matrix)



Survivor Selection
------------------
For creating custom Survivor Selection strategies, we can subclass :py:class:`~metaheuristic_designer.survivor_selection_base.SurvivorSelection`
and implement the ``select`` method, which takes as input two :py:class:`~metaheuristic_designer.population.Population` instances and
returns a new population of selected individuals. It is highly recommended to set the ``last_selection_idx`` attribute while performing
the selection as with parent selection methods, as it can be of use for other methods, it should be a vector contain only indices of the
selected individuals. If the first population has size N and the second has size M, the valid indices are those between 0 and N+M-1, 
where the first N indices indicate individuals from the first population and indices between N and N+M-1 indicate individuals from the
second.

.. code-block:: python

    import numpy as np
    from metaheuristic_designer import SurvivorSelection, Population

    class FullConcatenationSelection(SurvivorSelection):
        def __init__(self, amount: int):
            super().__init__()
        
        def select(self, parents: Population, offspring: population):
            n_parents = parents.population_size
            n_offspring = offspring.population_size

            self.last_selection_idx = np.arange(n_parents+n_offspring)
            return Population.join_populations(parents, offspring)
    
    parent_sel = FullConcatenationSelection()
    parents = Population(np.array([[1, 2, 3]]))
    offspring = Population(np.array([[4, 5, 6]]))

    new_population = parent_sel.select(parents, offspring)
    print(new_population.genotype_matrix)


Search strategy
---------------

If we want to design a new Search strategy, the recommended approach is to implement the necessary components
and pass them to an already existing search strategy blueprint. In the vast majority of cases, it is enough to 
create a :py:class:`~metaheuristic_designer.strategies.population_based_strategy.PopulationBasedStrategy` or
a :py:class:`~metaheuristic_designer.strategies.single_solution_strategy.SingleSolutionStrategy` if we work with a single
solution. Here we show an example.

.. code-block:: python

    from metaheuristic_designer.benchmarks import Sphere
    from metaheuristic_designer.initializers import UniformInitializer
    from metaheuristic_designer.parent_selection import create_parent_selection
    from metaheuristic_designer.operators import create_operator
    from metaheuristic_designer.survivor_selection import create_survivor_selection
    from metaheuristic_designer.strategies import PopulationBasedStrategy

    DIM = 5

    objfunc = Sphere(DIM)
    init = UniformInitializer(DIM, -10, 10, population_size=100, rng=42)
    parent_sel = create_parent_selection("elitist", amount=50, rng=42)
    operator = create_operator("mutation.gaussian_mutation", rng=42)
    survivor_sel = create_survivor_selection("keep_best", rng=42)

    search_strategy = PopulationBasedStrategy(
        initializer=init,
        parent_sel=parent_sel,
        operator=operator,
        survivor_sel=survivor_sel,
        rng=42
    )

If the options given are not enough, you can still subclass the :py:class:`~metaheuristic_designer.search_strategy.SearchStrategy`
class and implement the ``step`` function that takes a :py:class:`~metaheuristic_designer.population.Population` and an
:py:class:`~metaheuristic_designer.objective_function.ObjectiveFunc` instance, and returns a new population. You should be very
careful to add as little logic as possible into Search strategies since they are meant to only be an orchestrator class that passes
the population object between components. The reason behind this is to encourage the use of smaller components that can be fit into
different algorithms, which would not be possible if the logic was hard-coded into a Search Strategy class. 

Instead of creating a new search strategy, since we highly discourage creating new strategies, we show the exact implementation of the
:py:class:`~metaheuristic_designer.strategies.population_based_strategy.PopulationBasedStrategy` class so the structure is clear
in case it's needed.

.. code-block:: python

    from copy import copy

    from metaheuristic_designer.population import Population
    from metaheuristic_designer.objective_function import ObjectiveFunc
    from metaheuristic_designer.initializer import Initializer
    from metaheuristic_designer.parent_selection_base import ParentSelection
    from metaheuristic_designer.survivor_selection_base import SurvivorSelection
    from metaheuristic_designer.search_strategy import SearchStrategy
    from metaheuristic_designer.operator import Operator

    class PopulationBasedStrategy(SearchStrategy):
        def __init__(
            self,
            initializer: Initializer,
            operator: Operator = None,
            parent_sel: ParentSelection = None,
            survivor_sel: SurvivorSelection = None,
            name: str = "Static Population Evolution",
            rng = None,
            **kwargs,
        ):
            super().__init__(
                initializer=initializer,
                operator=operator,
                parent_sel=parent_sel,
                survivor_sel=survivor_sel,
                name=name,
                rng=rng,
                **kwargs
            )

        def step(self, prev_population: Population, objfunc: ObjectiveFunc) -> Population:
            population = self.parent_sel.select(prev_population)
            population = self.operator.evolve(population)
            population = objfunc.repair_population(population)
            population = objfunc.calculate_fitness(population)
            population = self.survivor_sel.select(population=prev_population, offspring=population)
            return population

Reporter
--------

To implement a custom :py:class:`~metaheuristic_designer.reporter.Reporter` class, we will subclassing and
implementing the three methods ``log_init`` that will display information at the start of the algorithm, 
``log_step`` that displays information each iteration of the algorithm and ``log_end`` that is called
at the end of the optimization. Every method gets an :py:class:`~metaheuristic_designer.algorithm.Algorithm`
instance and has no return value. 

.. code-block:: python

    from metaheuristic_designer.benchmarks import Sphere
    from metaheuristic_designer.simple import evolution_strategy_real
    from metaheuristic_designer import Reporter

    class SimpleReporter(Reporter):
        def __init__(self):
            self.iterations = 0

        def log_init(self, alg):
            print("Starting algorithm")
        
        def log_step(self, alg):
            self.iterations += 1
            print(f"Iteration {self.iterations}.")
        
        def log_end(self, alg):
            print(f"Ran for {self.iterations} iterations")

    objfunc = Sphere(5)
    reporter = SimpleReporter()
    alg = evolution_strategy_real(
        objfunc,
        max_iterations=100,
        stop_condition_str="max_iterations",
        reporter=reporter
    )
    alg.optimize() # Shows the reporting prints
    
An interesting detail is that reporters are not forced to only write on the command line and, theoretically,
it is possible to let reporters output in any format you can think of, including live-updating
plots or writing to files.
