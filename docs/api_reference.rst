===============
API reference
===============

Base Classes
------------
These are the intefaces from which to inherit to implement a new component for any optimizaion algorithm.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`~metaheuristic_designer.ObjectiveFunc`", "Prototype of a data type agnostic objective function."
   ":py:class:`~metaheuristic_designer.VectorObjectiveFunc`", "Prototype of an objective function with vector inputs."
   ":py:class:`~metaheuristic_designer.ConstraintHandler`", "Prototype of a constraint handler class."
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

   ":py:class:`~metaheuristic_designer.ObjectiveFromLambda`", "Objective function from a function."
   ":py:class:`~metaheuristic_designer.ConstraintHandlerFromLambda`", "Constraint Handler from a function."
   ":py:class:`~metaheuristic_designer.InitializerFromLambda`", "Population initializer from a function."
   ":py:class:`~metaheuristic_designer.EncodingFromLambda`", "Encoding from an encoding function and a decoding function."
   ":py:class:`~metaheuristic_designer.SelectionFromLambda`", "Individual selection from a function."
   ":py:class:`~metaheuristic_designer.OperatorFromLambda`", "Operator from a function."

Extended encoding classes
-------------------------
When the genotype vector encodes more information other than the soluion, like in the case of PSO with a speed vector and adaptative algorithms with 
parameters of the algorithm, you need to use the following interfaces to handle the size difference of the solution vector.

Note that there are also a number of implementations of these interfaces already available for specific algorithms like PSO.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`~metaheuristic_designer.ExtendedEncoding`", "Encoding that splits the vector into a solution and a dictionary with the necesary data."
   ":py:class:`~metaheuristic_designer.ExtendedConstraintHandler`", "Split constraint handler that treats the solution and the rest of the data separately."
   ":py:class:`~metaheuristic_designer.ExtendedInitializer`", "Split initializer that initializes the solution with a different distribution than other data."
   ":py:class:`~metaheuristic_designer.ExtendedOperator`", "Split operator that applies a different operation to the solution and the other data."


Initializers
------------

The implemented initializers are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`initializers.UniformInitializer<metaheuristic_designer.initializers.UniformInitializer>`", "Initializer that uses an uniform distribution."
   ":py:class:`initializers.GaussianInitializer<metaheuristic_designer.initializers.GaussianInitializer>`", "Initializer that uses an gaussian distribution."
   ":py:class:`initializers.DirectInitializer<metaheuristic_designer.initializers.DirectInitializer>`", "Initializer with a predefined population of individuals."
   ":py:class:`initializers.SeedDetermInitializer<metaheuristic_designer.initializers.SeedDetermInitializer>`", "Initializer with a fixed number of seeded solutions."
   ":py:class:`initializers.SeedProbInitializer<metaheuristic_designer.initializers.SeedProbInitializer>`", "Initializer with randomly inserted seeded solutions."
   ":py:class:`initializers.PermInitializer<metaheuristic_designer.initializers.PermInitializer>`", "Initializer that produces random permutations of n elements as vectors."

Encodings
---------

The implemented encodings are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`DefaultEncoding<metaheuristic_designer.DefaultEncoding>`", "Encoding that makes no changes while encoding or decoding."
   ":py:class:`encodings.TypeCastEncoding<metaheuristic_designer.encodings.TypeCastEncoding>`", "Encoding that changes the datatype when encoding and decoding."
   ":py:class:`encodings.MatrixEncoding<metaheuristic_designer.encodings.MatrixEncoding>`", "Encoding that reshapes a vector to a tensor with a different size."
   ":py:class:`encodings.ImageEncoding<metaheuristic_designer.encodings.ImageEncoding>`", "Encoding that reshapes a vector to a NxMxC matrix of bytes that represents NxM images with C channels."
   ":py:class:`encodings.SigmoidEncoding<metaheuristic_designer.encodings.SigmoidEncoding>`", "Encoding that allows algorithms for continuous problems to be applied to binary problems. Converts real numbers to probabilities."
   ":py:class:`encodings.CompositeEncoding<metaheuristic_designer.encodings.CompositeEncoding>`", "Encoding that applies other encoders in sequence."
   ":py:class:`encodings.CMAEncoding<metaheuristic_designer.encodings.CMAEncoding>`", "Encoding used by the CMA-ES algorithm."

Individual Selection Methods
----------------------------

The implemented operators for each datatype are listed in the following section 

.. csv-table::
   :header: "Module name", "Methods", "Description"

   ":py:class:`~metaheuristic_designer.selection_methods.ParentSelection`", ":ref:`implemented metods<Parent Selection>`","Selection methods for parent selection."
   ":py:class:`~metaheuristic_designer.selection_methods.NullParentSelection`", "", "Selection methods for parent selection that returns the original population."
   ":py:class:`~metaheuristic_designer.selection_methods.SurvivorSelection`", ":ref:`implemented metods<Survivor Selection>`", "Selection methods for survivor selection."
   ":py:class:`~metaheuristic_designer.selection_methods.NullSurvivorSelection`", "", "Selection methods for survivor selection that returns the offspring."


Operators
----------------------------

The implemented operators for each datatype are listed in the following section 

.. toctree::
   :maxdepth: 1

   api_reference.methods

.. csv-table::
   :header: "Module name", "Methods", "Description"

   ":py:class:`NullOperator<metaheuristic_designer.NullOperator>`", "", "Operator that makes no changes to the individual."
   ":py:class:`operators.VectorOperator<metaheuristic_designer.operators.VectorOperator>`", ":ref:`implemented metods<Operator Vector Methods>`", "Operator for vectors."
   ":py:class:`operators.PermOperator<metaheuristic_designer.operators.PermOperator>`", ":ref:`implemented metods<Operator Perm Methods>`", "Operator for permutations."
   ":py:class:`operators.MetaOperator<metaheuristic_designer.operators.MetaOperator>`", ":ref:`implemented metods<Operator Meta Methods>`", "Operator that combines other operators."
   ":py:class:`operators.AdaptativeOperator<metaheuristic_designer.operators.AdaptativeOperator>`", "", "Operator that uses part of the individual as parameters for the operator."
   ":py:class:`operators.BOOperator<metaheuristic_designer.operators.BOOperator>`", "", "Operator used in Bayesian Optimization."


Search Strategies
-----------------
The implemented search strategies are listed below.

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`strategies.NoSearch<metaheuristic_designer.strategies.NoSearch>`", "Search strategy that does nothing."
   ":py:class:`strategies.RandomSearch<metaheuristic_designer.strategies.RandomSearch>`", "Random search strategy."
   ":py:class:`strategies.StaticPopulation<metaheuristic_designer.strategies.StaticPopulation>`", "Basic population based search strategy with a fixed-size population."
   ":py:class:`strategies.VariablePopulation<metaheuristic_designer.strategies.VariablePopulation>`", "Basic population based search strategy with a variable-size population."
   ":py:class:`strategies.HillClimb<metaheuristic_designer.strategies.HillClimb>`", "Greedy Hill Climbing strategy."
   ":py:class:`strategies.LocalSearch<metaheuristic_designer.strategies.LocalSearch>`", "Local search strategy."
   ":py:class:`strategies.SA<metaheuristic_designer.strategies.SA>`", "Simulated annealing strategy."
   ":py:class:`strategies.GA<metaheuristic_designer.strategies.GA>`", "Genetic Algorithm strategy."
   ":py:class:`strategies.ES<metaheuristic_designer.strategies.ES>`", "Evolution Strategy."
   ":py:class:`strategies.HS<metaheuristic_designer.strategies.HS>`", "Harmony Search strategy."
   ":py:class:`strategies.DE<metaheuristic_designer.strategies.DE>`", "Differential Evolution strategy."
   ":py:class:`strategies.PSO<metaheuristic_designer.strategies.swarm.PSO>`", "Particle Swarm Optimization strategy."
   ":py:class:`strategies.BernoulliPBIL<metaheuristic_designer.strategies.EDA.BernoulliPBIL>`", "| Bernoulli distributed Population-based Incremental Learning.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.BernoulliUMDA<metaheuristic_designer.strategies.EDA.BernoulliUMDA>`", "| Bernoulli distributed Univariate Marginal Distribution Algorithm.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.BinomialPBIL<metaheuristic_designer.strategies.EDA.BinomialPBIL>`", "| Binomial distributed Population-based Incremental Learning.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.BinomialUMDA<metaheuristic_designer.strategies.EDA.BinomialUMDA>`", "| Binomial distributed Univariate Marginal Distribution Algorithm.
   | Estimation of Distribution Algorithm."
   ":py:class:`strategies.BayesianOptimization<metaheuristic_designer.strategies.bayesian_optimization.BayesianOptimization>`", "Bayesian Optimization"
   ":py:class:`strategies.VND<metaheuristic_designer.strategies.VNS.VND>`", "Variable Neighborhood Descent strategy."
   ":py:class:`strategies.RVNS<metaheuristic_designer.strategies.VNS.RVNS>`", "Restricted Varaible Neighborhood Search strategy."
   ":py:class:`strategies.VNS<metaheuristic_designer.strategies.VNS.VNS>`", "Variable Neighborhood Search strategy."
   ":py:class:`strategies.CMA_ES<metaheuristic_designer.strategies.CMA_ES>`", "Covariance Matrix Adaption-Evolution Strategy."
   ":py:class:`strategies.CRO<metaheuristic_designer.strategies.CRO>`", "Coral Reef Optimization strategy."
   ":py:class:`strategies.CRO_SL<metaheuristic_designer.strategies.CRO_SL>`", "Coral Reef Optimization with Subtrate Layers strategy."
   ":py:class:`strategies.PCRO_SL<metaheuristic_designer.strategies.PCRO_SL>`", "Probabilistic Coral Reef Optimization with Subtrate Layers strategy."
   ":py:class:`strategies.DPCRO_SL<metaheuristic_designer.strategies.DPCRO_SL>`", "Dynamic Probabilistic Coral Reef Optimization with Subtrate Layers strategy."

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
