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

.. toctree::
   :maxdepth: 1

   api_reference.methods

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`selectionMethods.ParentSelection<metaheuristic_designer.selectionMethods.ParentSelection.ParentSelection>`", "Selection methods for parent selection."
   ":py:class:`selectionMethods.ParentSelectionNull<metaheuristic_designer.selectionMethods.ParentSelectionNull>`", "Selection methods for parent selection that returns the original population."
   ":py:class:`selectionMethods.SurvivorSelection<metaheuristic_designer.selectionMethods.SurvivorSelection.SurvivorSelection>`", "Selection methods for survivor selection."
   ":py:class:`selectionMethods.SurvivorSelectionNull<metaheuristic_designer.selectionMethods.SurvivorSelectionNull>`", "Selection methods for survivor selection that returns the offspring."


Operators
----------------------------

The implemented operators for each datatype are listed in the following section 

.. toctree::
   :maxdepth: 1

   api_reference.methods

.. csv-table::
   :header: "Module name", "Methods" ,"Description"

   ":py:class:`operators.OperatorBinary<metaheuristic_designer.operators.OperatorBinary.OperatorBinary>`", ":ref:`implemented metods<Operator Binary Methods>`", "Operator for binary coded vectors."
   ":py:class:`operators.OperatorInt<metaheuristic_designer.operators.OperatorInt.OperatorInt>`", "", "Operator for integer coded vectors."
   ":py:class:`operators.OperatorReal<metaheuristic_designer.operators.OperatorReal.OperatorReal>`", "", "Operator for real coded vectors."
   ":py:class:`operators.OperatorPerm<metaheuristic_designer.operators.OperatorPerm.OperatorPerm>`", "", "Operator for permutations."
   ":py:class:`operators.OperatorList<metaheuristic_designer.operators.OperatorList.OperatorList>`", "", "Operator for variable-length collections."
   ":py:class:`operators.OperatorAdaptative<metaheuristic_designer.operators.OperatorAdaptative.OperatorAdaptative>`", "", "Operator that uses part of the individual as parameters for the operator."
   ":py:class:`operators.OperatorMeta<metaheuristic_designer.operators.OperatorMeta.OperatorMeta>`", "", "Operator that combines other operators."
   ":py:class:`operators.OperatorNull<metaheuristic_designer.operators.OperatorNull.OperatorNull>`", "", "Operator that makes no changes to the individual."


Search Strategies
-----------------

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

.. csv-table::
   :header: "Module name", "Description"

   ":py:class:`algorithms.GeneralAlgorithm<metaheuristic_designer.algorithms.GeneralAlgorithm>`", "Default algorithm implementation."
   ":py:class:`algorithms.MemeticAlgorithm<metaheuristic_designer.algorithms.MemeticAlgorithm>`", "Algorithm that combines one search strategy with a local search procedure."
   ":py:class:`algorithms.AlgorithmSelection<metaheuristic_designer.algorithms.AlgorithmSelection>`", "Algorithm that evaluates a given set of optimization algorithms."
   ":py:class:`algorithms.StrategySelection<metaheuristic_designer.algorithms.StrategySelection>`", "Algorithm that evaluates a given set of search strategies."


   


