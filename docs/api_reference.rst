.. _api-ref-intro:

Welcome to the Metaheuristic Designer API
=========================================

The library is built around a **layered architecture**.  At the bottom you find
abstract interfaces (the “Base Classes” below) that define the contracts for every
optimisation component.  On top of them sit concrete implementations:
pre-built initializers, operators, encodings, selection schemes, and full search
strategies.  An :class:`Algorithm` object glues everything together and runs the
optimisation loop.

If you want to **jump straight into code**, take a look at the
:doc:`simple prepackaged functions <api_reference.methods>` or follow the
:doc:`Algorithm Configuration <api_reference.algorithm_config>` guide - they show
how to assemble a complete optimiser in a few lines.

For plotting your results, see the :doc:`Plotting Tutorial <api_reference.plotting>`.

Base Classes
------------
These are the interfaces from which to inherit to implement a new component for any optimization algorithm.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`SchedulableParameter<metaheuristic_designer.schedulable_parameter.SchedulableParameter>`", "Prototype of a parameter that changes in value as iterations go on."
   ":py:class:`ParametrizableMixin<metaheuristic_designer.parametrizable_mixin.ParametrizableMixin>`", "Mixin used by interfaces to hold schedulable parameters."
   ":py:class:`ObjectiveFunc<metaheuristic_designer.objective_function.ObjectiveFunc>`", "Prototype of a data type agnostic objective function."
   ":py:class:`VectorObjectiveFunc<metaheuristic_designer.objective_function.VectorObjectiveFunc>`", "Prototype of an objective function with vector inputs."
   ":py:class:`ConstraintHandler<metaheuristic_designer.constraint_handler.ConstraintHandler>`", "Prototype of a constraint handler class."
   ":py:class:`Initializer<metaheuristic_designer.initializer.Initializer>`", "Prototype of a population initializer."
   ":py:class:`Encoding<metaheuristic_designer.encoding.Encoding>`", "Prototype of an encoding and decoding genotypes of the population."
   ":py:class:`ParentSelection<metaheuristic_designer.parent_selection.ParentSelection>`", "Prototype of a parent selection method."
   ":py:class:`SurvivorSelection<metaheuristic_designer.survivor_selection.SurvivorSelection>`", "Prototype of a survivor selection method."
   ":py:class:`Operator<metaheuristic_designer.operator.Operator>`", "Prototype of an operator on individuals."
   ":py:class:`SearchStrategy<metaheuristic_designer.search_strategy.SearchStrategy>`", "Prototype of a search strategy applied each generation."
   ":py:class:`Algorithm<metaheuristic_designer.algorithm.Algorithm>`", "Prototype of a full optimization algorithm."

Almost every interface works with the :py:class:`Population<metaheuristic_designer.population.Population>` class internally.

Lambda Implementations
----------------------
Classes that allow implementing optimization modules with a lambda function, avoiding the need to inherit from a base class.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`ObjectiveFromLambda<metaheuristic_designer.objective_function.ObjectiveFromLambda>`", "Objective function from a function."
   ":py:class:`ConstraintHandlerFromLambda<metaheuristic_designer.constraint_handler.ConstraintHandlerFromLambda>`", "Constraint Handler from a function."
   ":py:class:`InitializerFromLambda<metaheuristic_designer.initializer.InitializerFromLambda>`", "Population initializer from a function."
   ":py:class:`EncodingFromLambda<metaheuristic_designer.encoding.EncodingFromLambda>`", "Encoding from an encoding function and a decoding function."
   ":py:class:`ParentSelectionFromLambda<metaheuristic_designer.parent_selection.ParentSelectionFromLambda>`", "Parent selection from a function."
   ":py:class:`SurvivorSelectionFromLambda<metaheuristic_designer.survivor_selection.SurvivorSelectionFromLambda>`", "Survivor selection from a function."
   ":py:class:`OperatorFromLambda<metaheuristic_designer.operator.OperatorFromLambda>`", "Operator from a function."
   ":py:class:`SearchStrategyFromLambda<metaheuristic_designer.search_strategy.SearchStrategyFromLambda>`", "Full step of the algorithm defined componentwise by functions."

For a complete walk‑through showing how to use these classes and the registration
functions, see :doc:`Custom Components <api_reference.custom_components>`.


Extended Encoding Classes
--------------------------
When the genotype vector encodes more information than just the solution - such as a speed vector for PSO or adaptive algorithm parameters - you need these interfaces to handle the extra data.

Note that concrete implementations for specific algorithms (e.g. PSO) already exist.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`~metaheuristic_designer.encodings.parameter_extending_encoding.ParameterExtendingEncoding`", "Encoding that splits the vector into a solution and a dictionary with the necessary extra data."
   ":py:class:`~metaheuristic_designer.constraint_handlers.extended_constraint.ExtendedConstraintHandler`", "Constraint handler that treats the solution and the extra data separately."
   ":py:class:`~metaheuristic_designer.initializers.extended_initializer.ExtendedInitializer`", "Initializer that handles the solution and extra data with different distributions."
   ":py:class:`~metaheuristic_designer.operators.extended_operator.ExtendedOperator`", "Operator that applies different operations to the solution and the extra data."

Parameter Schedules
--------------------
Many numerical parameters can be made schedulable by passing a :py:class:`~metaheuristic_designer.SchedulableParameter` subclass instead of a constant. The available schedules are:

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`parameter_schedules.LinearSchedule<metaheuristic_designer.parameter_schedules.linear_schedule.LinearSchedule>`", "Changes the value linearly between two values."
   ":py:class:`parameter_schedules.LogisticSchedule<metaheuristic_designer.parameter_schedules.logistic_schedule.LogisticSchedule>`", "Sigmoid-shaped transition between two values."
   ":py:class:`parameter_schedules.ThresholdSchedule<metaheuristic_designer.parameter_schedules.threshold_schedule.ThresholdSchedule>`", "Keeps the initial value until progress crosses a threshold, then switches to the final value."
   ":py:class:`parameter_schedules.StepSchedule<metaheuristic_designer.parameter_schedules.step_schedule.StepSchedule>`", "Discrete steps defined by a list of (progress, value) pairs."
   ":py:class:`parameter_schedules.RandomSchedule<metaheuristic_designer.parameter_schedules.random_schedule.RandomSchedule>`", "Randomly picks a value between two bounds at each step."

All schedules depend on the algorithm's **progress**, a number between 0 and 1, to decide parameter values.

Initializers
------------
Implemented population initializers:

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`initializers.UniformInitializer<metaheuristic_designer.initializers.uniform_initializer.UniformInitializer>`", "Initializer that uses a uniform distribution."
   ":py:class:`initializers.GaussianInitializer<metaheuristic_designer.initializers.gaussian_initializer.GaussianInitializer>`", "Initializer that uses a Gaussian distribution."
   ":py:class:`initializers.ExponentialInitializer<metaheuristic_designer.initializers.exponential_initializer.ExponentialInitializer>`", "Initializer that uses an exponential distribution."
   ":py:class:`initializers.DirectInitializer<metaheuristic_designer.initializers.direct_initializer.DirectInitializer>`", "Initializer with a predefined population of individuals."
   ":py:class:`initializers.SeedDetermInitializer<metaheuristic_designer.initializers.seed_initializer.SeedDetermInitializer>`", "Initializer that inserts a fixed number of seeded solutions."
   ":py:class:`initializers.SeedProbInitializer<metaheuristic_designer.initializers.seed_initializer.SeedProbInitializer>`", "Initializer that randomly inserts seeded solutions with a given probability."
   ":py:class:`initializers.PermInitializer<metaheuristic_designer.initializers.perm_initializer.PermInitializer>`", "Initializer that produces random permutations of n elements."

Encodings
---------
Implemented encodings that transform between the internal genotype and the phenotype evaluated by the objective function:

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`DefaultEncoding<metaheuristic_designer.encoding.DefaultEncoding>`", "No change (identity encoding)."
   ":py:class:`encodings.TypeCastEncoding<metaheuristic_designer.encodings.type_cast_encoding.TypeCastEncoding>`", "Changes the datatype (e.g. float ↔ int ↔ boolean)."
   ":py:class:`encodings.MatrixEncoding<metaheuristic_designer.encodings.matrix_encoding.MatrixEncoding>`", "Reshapes a vector to a tensor of a different shape."
   ":py:class:`encodings.ImageEncoding<metaheuristic_designer.encodings.image_encoding.ImageEncoding>`", "Reshapes a vector to an N×M×C image representation (channels last)."
   ":py:class:`encodings.SigmoidEncoding<metaheuristic_designer.encodings.sigmoid_encoding.SigmoidEncoding>`", "Maps real numbers to probabilities via a sigmoid, enabling continuous operators on binary problems."
   ":py:class:`encodings.CompositeEncoding<metaheuristic_designer.encodings.composite_encoding.CompositeEncoding>`", "Applies a sequence of other encodings in order."
   ":py:class:`encodings.PSOEncoding<metaheuristic_designer.encodings.special.PSO_encoding.PSOEncoding>`", "Extended encoding that stores a speed vector for PSO."
   ":py:class:`encodings.SelfAdaptingESEncoding<metaheuristic_designer.encodings.special.self_adapting_ES_encoding.SelfAdaptingESEncoding>`", "Extended encoding that stores mutation strength(s) for self-adaptive evolution strategies."

Selection Methods
-----------------
Parent and survivor selection are created with dedicated **factory functions**:

* :py:func:`~metaheuristic_designer.create_parent_selection` - returns a :py:class:`~metaheuristic_designer.ParentSelection` instance.
* :py:func:`~metaheuristic_designer.create_survivor_selection` - returns a :py:class:`~metaheuristic_designer.SurvivorSelection` instance.

To skip a selection step entirely, use :py:class:`~metaheuristic_designer.NullParentSelection` / :py:class:`~metaheuristic_designer.NullSurvivorSelection`.

The complete catalogue of available methods, their parameters, and instructions for
registering custom ones are given on the :doc:`API reference - Implemented Operators & Selection <api_reference.methods>` page
(see :ref:`selection-methods`).

To learn how to create custom parent or survivor selection methods and register them,
consult the :doc:`Custom Components <api_reference.custom_components>` page.

Operators
---------
Operators modify the genotype of individuals. They are created through the **factory function**
:py:func:`~metaheuristic_designer.create_operator`, which accepts a string key
in the format ``"category.method"`` (e.g. ``"mutation.gaussian_mutation"``) and optional
keyword parameters.

.. code-block:: python

    from metaheuristic_designer.operators import create_operator
    op = create_operator("mutation.gaussian_mutation", F=0.2, random_state=42)

A few built-in operators do **not** follow the factory pattern:

.. csv-table::
   :header: "Class", "Description"

   ":py:class:`~metaheuristic_designer.operator.NullOperator`", "Identity operator (no changes)."
   ":py:class:`~metaheuristic_designer.operators.composite_operator.CompositeOperator`", "Combines multiple operators sequentially."
   ":py:class:`~metaheuristic_designer.operators.masked_operator.MaskedOperator`", "Applies different operators to different sets of components of the genotype vectors."
   ":py:class:`~metaheuristic_designer.operators.branch_operator.BranchOperator`", "Randomly selects one operator from a list each time it is applied."
   ":py:class:`~metaheuristic_designer.operators.extended_operator.ExtendedOperator`", "Base class for operators that handle extra per-individual parameters (e.g. self-adaptation)."
   ":py:class:`~metaheuristic_designer.operators.BO_operator.BOOperator`", "Gaussian process regression operator for Bayesian Optimisation."

**IMPORTANT**: For the full catalogue of factory-available operators (mutation, crossover, permutation, DE,
swarm, …) and the probability distributions they support, see the
:doc:`Operator Methods <api_reference.methods>` page.

For writing and registering your own operators, refer to the
:doc:`Custom Components <api_reference.custom_components>` guide.

Search Strategies
-----------------
A **search strategy** defines how one iteration of the optimisation loop is performed:
it chooses parents, applies operators, and selects survivors.  In practice it acts as
a container for an initializer, an operator, and optional parent/survivor selection.

You can use one of the ready‑made strategies:

.. code-block:: python

   from metaheuristic_designer.strategies import GA
   strategy = GA(
       initializer=UniformInitializer(...),
       mutation_op=create_operator("mutation.gaussian_mutation", F=0.1),
       crossover_op=create_operator("crossover.uniform"),
       parent_sel=create_parent_selection("tournament", amount=50),
       survivor_sel=create_survivor_selection("elitism", amount=25),
   )

or build your own by directly combining components with the general
:class:`SearchStrategy` class:

.. code-block:: python

   strategy = SearchStrategy(
       initializer=...,
       operator=...,
       parent_sel=...,
       survivor_sel=...,
   )

Both approaches result in an object that can be passed to :class:`Algorithm`.

The following pre‑built strategies are available:

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`strategies.NoSearch<metaheuristic_designer.strategies.no_search.NoSearch>`", "No‑op strategy (does nothing)."
   ":py:class:`strategies.RandomSearch<metaheuristic_designer.strategies.classic.random_search.RandomSearch>`", "Random search."
   ":py:class:`strategies.StaticPopulation<metaheuristic_designer.strategies.static_population.StaticPopulation>`", "Fixed‑size population based evolution."
   ":py:class:`strategies.VariablePopulation<metaheuristic_designer.strategies.variable_population.VariablePopulation>`", "Variable‑size population based evolution."
   ":py:class:`strategies.HillClimb<metaheuristic_designer.strategies.hill_climb.HillClimb>`", "Greedy hill climbing."
   ":py:class:`strategies.LocalSearch<metaheuristic_designer.strategies.local_search.LocalSearch>`", "Local search with a configurable number of iterations."
   ":py:class:`strategies.SA<metaheuristic_designer.strategies.classic.SA.SA>`", "Simulated annealing."
   ":py:class:`strategies.GA<metaheuristic_designer.strategies.classic.GA.GA>`", "Genetic Algorithm."
   ":py:class:`strategies.ES<metaheuristic_designer.strategies.classic.ES.ES>`", "Evolution Strategy."
   ":py:class:`strategies.DE<metaheuristic_designer.strategies.classic.DE.DE>`", "Differential Evolution."
   ":py:class:`strategies.PSO<metaheuristic_designer.strategies.swarm.PSO.PSO>`", "Particle Swarm Optimisation."
   ":py:class:`strategies.BernoulliPBIL<metaheuristic_designer.strategies.EDA.PBIL.BernoulliPBIL>`", "Bernoulli Population‑Based Incremental Learning (EDA)."
   ":py:class:`strategies.BernoulliUMDA<metaheuristic_designer.strategies.EDA.UMDA.BernoulliUMDA>`", "Bernoulli Univariate Marginal Distribution Algorithm (EDA)."
   ":py:class:`strategies.BinomialPBIL<metaheuristic_designer.strategies.EDA.PBIL.BinomialPBIL>`", "Binomial PBIL."
   ":py:class:`strategies.BinomialUMDA<metaheuristic_designer.strategies.EDA.UMDA.BinomialUMDA>`", "Binomial UMDA."
   ":py:class:`strategies.BayesianOptimization<metaheuristic_designer.strategies.bayesian_optimization.bayesian_optimization.BayesianOptimization>`", "Bayesian Optimisation with Gaussian processes."
   ":py:class:`strategies.CMA_ES<metaheuristic_designer.strategies.classic.CMA_ES.CMA_ES>`", "Covariance Matrix Adaptation Evolution Strategy."

Algorithms
----------
The :class:`Algorithm` class runs the optimisation loop.  You pass it an objective
function, a search strategy, and optionally a stopping condition, reporter,
history tracker and checkpointer (see :doc:`Algorithm Configuration <api_reference.algorithm_config>`).

.. code-block:: python

   algo = Algorithm(objfunc, strategy)
   population = algo.optimize()
   best_solution, best_obj = population.best_solution()

Built‑in algorithm variants:

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`Algorithm<metaheuristic_designer.algorithm.Algorithm>`", "Default algorithm with the classic parent → perturb → evaluate → survivor loop."
   ":py:class:`algorithms.MemeticAlgorithm<metaheuristic_designer.algorithms.memetic_algorithm.MemeticAlgorithm>`", "Algorithm that embeds a local search step inside the main loop."
   ":py:class:`algorithms.AlgorithmSelection<metaheuristic_designer.algorithms.algorithm_selection.AlgorithmSelection>`", "Benchmarks a set of algorithms."
   ":py:class:`algorithms.StrategySelection<metaheuristic_designer.algorithms.strategy_selection.StrategySelection>`", "Benchmarks a set of search strategies."

Stopping conditions can be defined as strings combining the following tokens with
``and``, ``or`` and parentheses.  See the :doc:`Algorithm Configuration <api_reference.algorithm_config>` page for how to set them.

.. csv-table::
   :widths: 30 70
   :header: "Token", "Description"

   ``"max_evaluations"``, "Maximum number of objective function evaluations."
   ``"max_iterations"``, "Maximum number of iterations (generations)."
   ``"real_time_limit"``, "Wall‑clock time limit in seconds."
   ``"cpu_time_limit"``, "CPU time limit in seconds."
   ``"objective_target"``, "Target value for the raw objective; stops when ``best_objective <= objective_target`` (minimisation) or ``best_objective >= objective_target`` (maximisation)."
   ``"convergence"``, "Stops after ``max_patience`` consecutive iterations without improvement."

Example: ``max_iterations or real_time_limit`` will halt when the maximum number of iterations is reached or we have exceeded the maximum time.

Prepackaged Algorithms
----------------------
For the most common optimisation scenarios, the :mod:`metaheuristic_designer.simple`
module provides ready‑to‑run functions.  Each algorithm is available in up to four
encoding variants: ``_real`` (continuous), ``_binary`` (bit‑strings), ``_discrete``
(integer) and ``_permutation`` (permutations).  The table for each algorithm lists
the concrete function names and the most important parameters.

Random Search
~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Description"

   ":py:func:`~simple.random_search_real`", "Continuous search, no parameters."
   ":py:func:`~simple.random_search_binary`", "Binary search, no parameters."
   ":py:func:`~simple.random_search_discrete`", "Discrete search, no parameters."
   ":py:func:`~simple.random_search_permutation`", "Permutation search, no parameters."

Hill Climbing
~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.hill_climb_real`", "``mutation_strength``, ``mutated_components``"
   ":py:func:`~simple.hill_climb_binary`", "``mutated_bits``"
   ":py:func:`~simple.hill_climb_discrete`", "``resampled_components``"
   ":py:func:`~simple.hill_climb_permutation`", "``swapped_positions``"

Local Search
~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.local_search_real`", "``mutation_strength``, ``mutated_components``, ``samples_per_iteration``"
   ":py:func:`~simple.local_search_binary`", "``mutated_bits``, ``samples_per_iteration``"
   ":py:func:`~simple.local_search_discrete`", "``resampled_components``, ``samples_per_iteration``"
   ":py:func:`~simple.local_search_permutation`", "``swapped_positions``, ``samples_per_iteration``"

Simulated Annealing
~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.simulated_annealing_real`", "``mutation_strength``, ``mutated_components``, ``initial_temperature``, ``alpha``, ``iterations``"
   ":py:func:`~simple.simulated_annealing_binary`", "``mutated_bits``, ``initial_temperature``, ``alpha``, ``iterations``"
   ":py:func:`~simple.simulated_annealing_discrete`", "``resampled_components``, ``initial_temperature``, ``alpha``, ``iterations``"
   ":py:func:`~simple.simulated_annealing_permutation`", "``swapped_positions``, ``initial_temperature``, ``alpha``, ``iterations``"

Evolution Strategy (ES)
~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.evolution_strategy_real`", "``mutation_strength``, ``mutated_components``, ``population_size``, ``offspring_size``, ``elitist``"
   ":py:func:`~simple.evolution_strategy_binary`", "``mutated_bits``, ``population_size``, ``offspring_size``, ``elitist``"
   ":py:func:`~simple.evolution_strategy_discrete`", "``resampled_components``, ``population_size``, ``offspring_size``, ``elitist``"
   ":py:func:`~simple.evolution_strategy_permutation`", "``swapped_positions``, ``population_size``, ``offspring_size``, ``elitist``"

Genetic Algorithm (GA)
~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.genetic_algorithm_real`", "``mutation_strength``, ``mutated_components``, ``population_size``"
   ":py:func:`~simple.genetic_algorithm_binary`", "``mutated_bits``, ``population_size``"
   ":py:func:`~simple.genetic_algorithm_discrete`", "``resampled_components``, ``population_size``"
   ":py:func:`~simple.genetic_algorithm_permutation`", "``swapped_positions``, ``population_size``"

Differential Evolution (DE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.differential_evolution_real`", "``population_size``, ``F``, ``Cr``, ``de_operator_name``"
   ":py:func:`~simple.differential_evolution_binary`", "``population_size``, ``F``, ``Cr``, ``de_operator_name``"
   ":py:func:`~simple.differential_evolution_discrete`", "``population_size``, ``F``, ``Cr``, ``de_operator_name``"

Particle Swarm Optimisation (PSO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.particle_swarm_real`", "``population_size``, ``w``, ``c1``, ``c2``"
   ":py:func:`~simple.particle_swarm_binary`", "``population_size``, ``w``, ``c1``, ``c2``"
   ":py:func:`~simple.particle_swarm_discrete`", "``population_size``, ``w``, ``c1``, ``c2``"

Bayesian Optimisation
~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Function", "Key parameters"

   ":py:func:`~simple.bayesian_optimization_real`", "``population_size``, ``acquisition_function``"