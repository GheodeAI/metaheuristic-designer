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
Pre-built strategies; all inherit from :py:class:`~metaheuristic_designer.SearchStrategy`.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`strategies.NoSearch<metaheuristic_designer.strategies.no_search.NoSearch>`", "No-op strategy (does nothing)."
   ":py:class:`strategies.RandomSearch<metaheuristic_designer.strategies.classic.random_search.RandomSearch>`", "Random search."
   ":py:class:`strategies.StaticPopulation<metaheuristic_designer.strategies.static_population.StaticPopulation>`", "Fixed-size population based evolution."
   ":py:class:`strategies.VariablePopulation<metaheuristic_designer.strategies.variable_population.VariablePopulation>`", "Variable-size population based evolution."
   ":py:class:`strategies.HillClimb<metaheuristic_designer.strategies.hill_climb.HillClimb>`", "Greedy hill climbing."
   ":py:class:`strategies.LocalSearch<metaheuristic_designer.strategies.local_search.LocalSearch>`", "Local search with a configurable number of iterations."
   ":py:class:`strategies.SA<metaheuristic_designer.strategies.classic.SA.SA>`", "Simulated annealing."
   ":py:class:`strategies.GA<metaheuristic_designer.strategies.classic.GA.GA>`", "Genetic Algorithm."
   ":py:class:`strategies.ES<metaheuristic_designer.strategies.classic.ES.ES>`", "Evolution Strategy."
   ":py:class:`strategies.DE<metaheuristic_designer.strategies.classic.DE.DE>`", "Differential Evolution."
   ":py:class:`strategies.PSO<metaheuristic_designer.strategies.swarm.PSO.PSO>`", "Particle Swarm Optimisation."
   ":py:class:`strategies.BernoulliPBIL<metaheuristic_designer.strategies.EDA.PBIL.BernoulliPBIL>`", "Bernoulli Population-Based Incremental Learning (EDA)."
   ":py:class:`strategies.BernoulliUMDA<metaheuristic_designer.strategies.EDA.UMDA.BernoulliUMDA>`", "Bernoulli Univariate Marginal Distribution Algorithm (EDA)."
   ":py:class:`strategies.BinomialPBIL<metaheuristic_designer.strategies.EDA.PBIL.BinomialPBIL>`", "Binomial PBIL."
   ":py:class:`strategies.BinomialUMDA<metaheuristic_designer.strategies.EDA.UMDA.BinomialUMDA>`", "Binomial UMDA."
   ":py:class:`strategies.BayesianOptimization<metaheuristic_designer.strategies.bayesian_optimization.bayesian_optimization.BayesianOptimization>`", "Bayesian Optimisation with Gaussian processes."
   ":py:class:`strategies.CMA_ES<metaheuristic_designer.strategies.classic.CMA_ES.CMA_ES>`", "Covariance Matrix Adaptation Evolution Strategy."

Algorithms
----------
Classes that wrap a search strategy into a full optimization loop.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`algorithms.StandardAlgorithm<metaheuristic_designer.algorithms.standard_algorithm.StandardAlgorithm>`", "Default algorithm with the classic parent → perturb → evaluate → survivor loop."
   ":py:class:`algorithms.MemeticAlgorithm<metaheuristic_designer.algorithms.memetic_algorithm.MemeticAlgorithm>`", "Algorithm that embeds a local search step inside the main loop."
   ":py:class:`algorithms.AlgorithmSelection<metaheuristic_designer.algorithms.algorithm_selection.AlgorithmSelection>`", "Benchmarks a set of algorithms."
   ":py:class:`algorithms.StrategySelection<metaheuristic_designer.algorithms.strategy_selection.StrategySelection>`", "Benchmarks a set of search strategies."

Stopping conditions can be defined as strings combining the following tokens with
`and`, `or`, `not` and parentheses:

.. csv-table::
   :header: "Token", "Description"

   ``"neval"``, "Maximum number of objective function evaluations."
   ``"ngen"``, "Maximum number of generations (iterations)."
   ``"time_limit"``, "Wall-clock time limit (seconds)."
   ``"cpu_time_limit"``, "CPU time limit (seconds)."
   ``"fit_target"``, "Target fitness value; stops when reached."
   ``"convergence"``, "Stops after a number of generations without improvement (the `patience` parameter)."

Example: ``"neval or time_limit"``.

Prepackaged Algorithms
----------------------
For the most common optimisation scenarios, the `simple` module provides ready-to-run
**functions** that build a complete `Algorithm` from a dictionary of parameters.
All of them share the same interface:

.. code-block:: python

    from metaheuristic_designer.simple import hill_climb
    alg = hill_climb(params={"vecsize": 10, "encoding": "real"}, objfunc=None)

If an objective function is not provided, the ``params`` dict must contain at least
``"vecsize"`` (and optionally ``"low"``, ``"high"``) to construct a default `Sphere` problem.
The ``"encoding"`` key accepts ``"bin"``, ``"int"`` or ``"real"``.

.. csv-table::
   :header: "Function", "Parameters", "Description"

   ":py:func:`simple.random_search<metaheuristic_designer.simple.random_search>`", "", "Pure random search."
   ":py:func:`simple.hill_climb<metaheuristic_designer.simple.hill_climb>`", "mut_str (0.1)", "Hill climbing. Mutation strength ``mut_str``: for binary it flips that many bits; for integer it samples from the range; for real it adds Gaussian noise with that standard deviation."
   ":py:func:`simple.simulated_annealing<metaheuristic_designer.simple.simulated_annealing>`", "mut_str (0.1), T0 (100), alpha (0.99)", "| Simulated annealing. Temperature decays as :math:`T_0 \\alpha^k`."
   ":py:func:`simple.evolution_strategy<metaheuristic_designer.simple.evolution_strategy>`", "pop_size (100), offspring_size (100), mut_str (0.5)", "(μ+λ)-ES with the same mutation interpretations as hill climbing."
   ":py:func:`simple.genetic_algorithm<metaheuristic_designer.simple.genetic_algorithm>`", "pop_size (100), n_parents (50), pmut (0.1), pcross (0.9), mut_str (0.5)", "Genetic algorithm with tournament selection, one-point crossover and mutation as above."
   ":py:func:`simple.differential_evolution<metaheuristic_designer.simple.differential_evolution>`", "pop_size (30), F (0.8), Cr (0.9), DE_type (``de/best/1``)", "Differential evolution. Internally works with floating-point numbers and uses an encoding to match the requested datatype."
   ":py:func:`simple.particle_swarm<metaheuristic_designer.simple.particle_swarm>`", "pop_size (30), w (0.7), c1 (1.5), c2 (1.5)", "Particle Swarm Optimisation. Works with real numbers internally and encodes to the target type."