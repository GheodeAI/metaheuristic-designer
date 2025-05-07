===============
API reference
===============

Base Classes
------------
These are the intefaces from which to inherit to implement a new component for any optimizaion algorithm.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`~metaheuristic_designer.ObjectiveFunc`", "Prototype of a data type agnostic objective function."
   ":py:class:`~metaheuristic_designer.ObjectiveFunc.ObjectiveVectorFunc`", "Prototype of an objective function with vector inputs."
   ":py:class:`~metaheuristic_designer.Initializer`", "Prototype of a population initializer."
   ":py:class:`~metaheuristic_designer.Encoding`", "Prototype of an individual solution encoding."
   ":py:class:`~metaheuristic_designer.SelectionMethod`", "Prototype of an individual selection method."
   ":py:class:`~metaheuristic_designer.Operator`", "Prototype of an operator on individuals."
   ":py:class:`~metaheuristic_designer.SearchStrategy`", "Prototype of a search strategy applied each generation."
   ":py:class:`~metaheuristic_designer.Algorithm`", "Prototype of a full optimization algorithm."


Lambda Implementations
----------------------
There are classes that allow the implementation of optimization modules with a lambda function, avoiding the need for inheriting from a base class

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`ObjectiveFromLambda<metaheuristic_designer.ObjectiveFunc.ObjectiveFromLambda>`", "Objective function from a function."
   ":py:class:`initializers.InitializerFromLambda<metaheuristic_designer.initializers.InitializerFromLambda>`", "Population initializer from a function."
   ":py:class:`encodings.EncodingFromLambda<metaheuristic_designer.encodings.EncodingFromLambda>`", "Encoding from an encoding function and a decoding function."
   ":py:class:`selectionMethods.SelectionFromLambda<metaheuristic_designer.selectionMethods.SelectionFromLambda>`", "Individual selection from a function."
   ":py:class:`operators.OperatorFromLambda<metaheuristic_designer.operators.OperatorFromLambda>`", "Operator from a function."

Initializers
------------

The implemented initializers are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`initializers.UniformVectorInitializer<metaheuristic_designer.initializers.UniformInitializer.UniformVectorInitializer>`", "Initializer that uses an uniform distribution."
   ":py:class:`initializers.GaussianVectorInitializer<metaheuristic_designer.initializers.GaussianInitializer.GaussianVectorInitializer>`", "Initializer that uses an gaussian distribution."
   ":py:class:`initializers.DirectInitializer<metaheuristic_designer.initializers.DirectInitializer>`", "Initializer with a predefined population of individuals."
   ":py:class:`initializers.SeedDetermInitializer<metaheuristic_designer.initializers.SeedInitializer.SeedDetermInitializer>`", "Initializer with a fixed number of seeded solutions."
   ":py:class:`initializers.SeedProbInitializer<metaheuristic_designer.initializers.SeedInitializer.SeedProbInitializer>`", "Initializer with randomly inserted seeded solutions."
   ":py:class:`initializers.PermInitializer<metaheuristic_designer.initializers.PermInitializer>`", "Initializer that produces random permutations of n elements as vectors."

Encodings
---------

The implemented encodings are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`encodings.DefaultEncoding<metaheuristic_designer.encodings.DefaultEncoding>`", "Encoding that makes no changes while encoding or decoding."
   ":py:class:`encodings.TypeCastEncoding<metaheuristic_designer.encodings.TypeCastEncoding>`", "Encoding that changes the datatype when encoding and decoding."
   ":py:class:`encodings.MatrixEncoding<metaheuristic_designer.encodings.MatrixEncoding>`", "Encoding that reshapes a vector to a tensor with a different size."
   ":py:class:`encodings.ImageEncoding<metaheuristic_designer.encodings.ImageEncoding>`", "Encoding that reshapes a vector to a NxMxC matrix of bytes that represents NxM images with C channels."
   ":py:class:`encodings.AdaptionEncoding<metaheuristic_designer.encodings.AdaptionEncoding>`", "Encoding that makes individuals represent the solution and parameters of the algorithm."
   ":py:class:`encodings.CompositeEncoding<metaheuristic_designer.encodings.CompositeEncoding>`", "Encoding that applies other encoders in sequence."
   ":py:class:`encodings.CMAEncoding<metaheuristic_designer.encodings.CMAEncoding>`", "Encoding used by the CMA-ES algorithm."

Individual Selection Methods
----------------------------

The implemented operators for each datatype are listed in the following section 

.. csv-table::
   :header: "Module name", "Methods", "Description"

   ":py:class:`selectionMethods.ParentSelection<metaheuristic_designer.selectionMethods.ParentSelection.ParentSelection>`", ":ref:`implemented metods<Operator Binary Methods>`","Selection methods for parent selection."
   ":py:class:`selectionMethods.SurvivorSelection<metaheuristic_designer.selectionMethods.SurvivorSelection.SurvivorSelection>`", ":ref:`implemented metods<Operator Binary Methods>`", "Selection methods for survivor selection."
   ":py:class:`selectionMethods.ParentSelectionNull<metaheuristic_designer.selectionMethods.ParentSelectionNull>`", "", "Selection methods for parent selection that returns the original population."
   ":py:class:`selectionMethods.SurvivorSelectionNull<metaheuristic_designer.selectionMethods.SurvivorSelectionNull>`", "", "Selection methods for survivor selection that returns the offspring."


Operators
----------------------------

The implemented operators for each datatype are listed in the following section 

.. toctree::
   :maxdepth: 1

   api_reference.methods

.. csv-table::
   :header: "Module name", "Methods", "Description"

   ":py:class:`operators.OperatorVector<metaheuristic_designer.operators.OperatorVector.OperatorVector>`", ":ref:`implemented metods<Operator Vector Methods>`", "Operator for vectors."
   ":py:class:`operators.OperatorPerm<metaheuristic_designer.operators.OperatorPerm.OperatorPerm>`", ":ref:`implemented metods<Operator Perm Methods>`", "Operator for permutations."
   ":py:class:`operators.OperatorAdaptative<metaheuristic_designer.operators.OperatorAdaptative.OperatorAdaptative>`", "", "Operator that uses part of the individual as parameters for the operator."
   ":py:class:`operators.OperatorMeta<metaheuristic_designer.operators.OperatorMeta.OperatorMeta>`", ":ref:`implemented metods<Operator Meta Methods>`", "Operator that combines other operators."
   ":py:class:`operators.OperatorNull<metaheuristic_designer.operators.OperatorNull.OperatorNull>`", "", "Operator that makes no changes to the individual."


Search Strategies
-----------------
The implemented search strategies are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`strategies.NoSearch<metaheuristic_designer.strategies.NoSearch>`", "Search strategy that does nothing."
   ":py:class:`strategies.RandomSearch<metaheuristic_designer.strategies.Classic.RandomSearch>`", "Random search strategy."
   ":py:class:`strategies.StaticPopulation<metaheuristic_designer.strategies.StaticPopulation>`", "Basic population based search strategy with a fixed-size population."
   ":py:class:`strategies.VariablePopulation<metaheuristic_designer.strategies.VariablePopulation>`", "Basic population based search strategy with a variable-size population."
   ":py:class:`strategies.HillClimb<metaheuristic_designer.strategies.HillClimb>`", "Greedy Hill Climbing strategy."
   ":py:class:`strategies.LocalSearch<metaheuristic_designer.strategies.LocalSearch>`", "Local search strategy."
   ":py:class:`strategies.SA<metaheuristic_designer.strategies.Classic.SA>`", "Simulated annealing strategy."
   ":py:class:`strategies.GA<metaheuristic_designer.strategies.Classic.GA>`", "Genetic Algorithm strategy."
   ":py:class:`strategies.ES<metaheuristic_designer.strategies.Classic.ES>`", "Evolution Strategy."
   ":py:class:`strategies.HS<metaheuristic_designer.strategies.Classic.HS>`", "Harmony Search strategy."
   ":py:class:`strategies.PSO<metaheuristic_designer.strategies.Classic.PSO>`", "Particle Swarm Optimization strategy."
   ":py:class:`strategies.DE<metaheuristic_designer.strategies.Classic.DE>`", "Differential Evolution strategy."
   ":py:class:`strategies.BernoulliPBIL<metaheuristic_designer.strategies.EDA.BernoulliPBIL>`", "| Bernoulli distributed Population-based Incremental Learning.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.BernoulliUMDA<metaheuristic_designer.strategies.EDA.BernoulliUMDA>`", "| Bernoulli distributed Univariate Marginal Distribution Algorithm.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.CRO<metaheuristic_designer.strategies.CRO.CRO>`", "Coral Reef Optimization strategy."
   ":py:class:`strategies.CRO_SL<metaheuristic_designer.strategies.CRO.CRO_SL>`", "Coral Reef Optimization with Subtrate Layers strategy."
   ":py:class:`strategies.PCRO_SL<metaheuristic_designer.strategies.CRO.PCRO_SL>`", "Probabilistic Coral Reef Optimization with Subtrate Layers strategy."
   ":py:class:`strategies.DPCRO_SL<metaheuristic_designer.strategies.CRO.DPCRO_SL>`", "Dynamic Probabilistic Coral Reef Optimization with Subtrate Layers strategy."
   ":py:class:`strategies.VND<metaheuristic_designer.strategies.VNS.VND>`", "Variable Neighborhood Descent strategy."
   ":py:class:`strategies.RVNS<metaheuristic_designer.strategies.VNS.RVNS>`", "Restricted Varaible Neighborhood Search strategy."
   ":py:class:`strategies.VNS<metaheuristic_designer.strategies.VNS.VNS>`", "Variable Neighborhood Search strategy."
   ":py:class:`strategies.CMA_ES<metaheuristic_designer.strategies.Classic.CMA_ES>`", "Covariance Matrix Adaption-Evolution Strategy."

Algorithms
----------
These are classes that implement optimization algorithms using a specified search strategy.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`algorithms.GeneralAlgorithm<metaheuristic_designer.algorithms.GeneralAlgorithm>`", "Default algorithm implementation."
   ":py:class:`algorithms.MemeticAlgorithm<metaheuristic_designer.algorithms.MemeticAlgorithm>`", "Algorithm that combines one search strategy with a local search procedure."
   ":py:class:`algorithms.AlgorithmSelection<metaheuristic_designer.algorithms.AlgorithmSelection>`", "Algorithm that evaluates a given set of optimization algorithms."
   ":py:class:`algorithms.StrategySelection<metaheuristic_designer.algorithms.StrategySelection>`", "Algorithm that evaluates a given set of search strategies."

The stopping condition for any of the algorithms is specified as one of the following values:

.. csv-table::
   :header: "Stopping condition", "Description"

   "neval", "Number of objective function evaluations."
   "ngen", "Number of generations or iterations of the algorithm."
   "time_limit", "Maximum amount of time allowed to spent." 
   "cpu_time_limit", "Maximum amount of CPU time allowed to spent." 
   "fit_target", "Value of the objective function after which we stop optimizing." 
   "convergence", "Maximum amount of iterations we allow the algorithm to pass without any improvement to the objective function. This amount is specified as 'patience'" 

They can also be combined with logical operations, this way "ngen or time_limit" is a valid stopping condition. The "and", "or" and "not" operators are available and parenthesis are allowed.


Prepackaged algorithms
----------------------
Here we list full algorithms that should work out of the box providing an objective function and a set of parameters. It is assumed that individuals will be represented by vectors.

All parameters are optional except for the encoding which is specified as 'encoding' having as possible values 'bin', 'int' or 'real'.

In case an objective function is not specified, the 'vecsize', 'max' and 'min' arguments must be specified for the size of the vectors and their upper and lower limit respectively. In the case of binary encoding the 'max' and 'min' parameters are not necessary.

.. csv-table::
   :header: "Algorithm name", "Parameters", "Description"

   ":py:class:`simple.random_search<metaheuristic_designer.simple.random_search>`", " ", "Random search algorithm."
   ":py:class:`simple.hill_climb<metaheuristic_designer.simple.hill_climb>`", "mut_str (float | int | ndarray)", "| Hill climb algorithm. 
   | - binary mutation: Bit flip of 'mut_str' components. 
   | - integer mutation: Sample 'mut_str' component from an uniform distribution. 
   | - real mutation: Add random gaussian noise with std 'mut_str'"
   ":py:class:`simple.simulated_annealing<metaheuristic_designer.simple.simulated_annealing>`", "mut_str (float | int | ndarray)", "| Simulated annealing algorithm. 
   | - binary mutation: Bit flip of 'mut_str' components. 
   | - integer mutation: Sample 'mut_str' component from an uniform distribution. 
   | - real mutation: Add random gaussian noise with std 'mut_str'"
   ":py:class:`simple.evolution_strategy<metaheuristic_designer.simple.evolution_strategy>`", "pop_size (int), offspring_size (int), mut_str (float | int | ndarray)", "| Evolution strategy algorithm.
   | - binary mutation: Bit flip of 'mut_str' components. 
   | - integer mutation: Sample 'mut_str' component from an uniform distribution. 
   | - real mutation: Add random gaussian noise with std 'mut_str'"
   ":py:class:`simple.genetic_algorithm<metaheuristic_designer.simple.genetic_algorithm>`", "pop_size (int), n_parents (int), pmut (float), pcross (float), mut_str (float | int | ndarray)", "| Genetic algorithm.
   | - binary mutation: Bit flip of 'mut_str' components. 
   | - integer mutation: Sample 'mut_str' component from an uniform distribution. 
   | - real mutation: Add random gaussian noise with std 'mut_str'"
   ":py:class:`simple.differential_evolution<metaheuristic_designer.simple.differential_evolution>`", "pop_size (int), F (float), Cr (float), DE_type (str)", "Differential evolution algorithm. The algorithm works internaly with floating point solutions, but uses an encoding to transform to the specified datatype."
   ":py:class:`simple.particle_swarm<metaheuristic_designer.simple.particle_swarm>`", "pop_size (int), w (float), c1 (float), c2 (float)", "Particle Swarm Optimization algorithm. The algorithm works internaly with floating point solutions, but uses an encoding to transform to the specified datatype."
