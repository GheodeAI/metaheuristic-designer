.. metaheuristic-designer documentation master file, created by
   sphinx-quickstart on Wed Jul 19 18:39:56 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to metaheuristic-designer's documentation!
==================================================

Description
-----------
This is an object-oriented framework for the development, testing and analysis of metaheuristic optimization algorithms.

It defines the components of a general evolutionary algorithm and offers some implementations of algorithms along with components
that can be used directly. Those components will be explained below.

It was inspired by the article `Metaheuristics “in the large” <https://doi.org/10.1016/j.ejor.2021.05.042>`_ that 
discusses some of the issues in the research on metaheuristic optimization, suggesting the development of libraries for the standardization
of metaheuristic algorithms.

Most of the design decisions are based on the book `Introduction to evolutionary computing <https://doi.org/10.1007/978-3-662-44874-8>`_ by Eiben, Agoston E.,
and James E. Smith, which is very well explained and is highly recommended to anyone willing to 
learn about the topic.

This framework doesn't claim to have high performance, especially since the chosen language is Python and the code has not been 
designed for speed. This shouldn't really be an issue since the highest amount of time spent in these kind of algorithms
tends to be in the evaluation of the objective function. If you want to compare an algorithm made with this tool with another
one that is available by other means, it is recommended to use the number of evaluations of the objective function as a metric instead of 
execution time.   


Structure
---------

Explanation
~~~~~~~~~~~

The terminology used is inherited mainly from genetic algorithms and classical optimization algorithms.

The structure of the optimization algorithms is very much inspired by the development of the
`DPCRO_SL algorithm <https://doi.org/10.3390/math11071666>`_ of which I was a part, which gave me the realization that evolutionary
algorithms and some other optimization algorithms tend to share a common structure:

1. **Initialize** a random solution or population of solutions.

2. **Repeat** until a stopping condition is met (time passed, number of evaluations, iterations without improvement...).

   2.1. **Select parents** (trivial in algorithms with only one solution).

   2.2. **Perturb the parents** (might involve a sequence of operators like a crossover and a mutation).

   2.3. **Select individuals** for the next iteration.

3. **Return the best solution**.

Implementation
~~~~~~~~~~~~~~

This is implemented as abstract classes (interfaces) that are related to each other and which must be implemented
to construct an optimization algorithm. Most common implementations of these interfaces are already available in the package.

First, **objective functions** are implemented as instances of :class:`ObjectiveFunc <metaheuristic_designer.ObjectiveFunc>` which receive
an input in some unspecified format (an array, a tree or any other object) and output a single numerical value. Our goal is to find
an input that maximizes (or minimizes) this output value.

Objective functions often come with constraints, which can be dealt with in several ways, including specifically engineered encodings or operators 
(explained later), applying a solution fixing procedure or applying a penalty to the objective when constraints are violated. For solution repairing
and penalty we have the :class:`ConstraintHandler <metaheuristic_designer.ConstraintHandler>` class, that is hooked up to the objective function. Penalties
are added at the same time as the objective is calculated, and solution repairing is performed immediately after applying the operators to the population.

Our algorithms will work with **populations**, represented as instances of :class:`Population <metaheuristic_designer.Population>` which are a collection
of solutions to our optimization problem. These populations will hold the solutions, their value on the optimization problem and the best solution found so far.

The solutions that an individual contains are encoded in a certain way, but our objective function might need an input encoded in a different way. 
This is where **encodings** are used, represented as instances of :class:`Encoding <metaheuristic_designer.Encoding>` which isolates the 
representation of the solution in the optimization procedure from the one used in the calculation of the objective function.

For the **initialization step**, there will be an instance of :class:`Initializer <metaheuristic_designer.Initializer>` that will generate
an initial population, often completely at random, and will be used whenever a random solution needs to be generated. This class will
also indicate the size of the population (1 if the algorithm works with only one solution).

Both **parent selection** and **survivor selection** are implemented as instances of
:class:`ParentSelection <metaheuristic_designer.selection_methods.ParentSelection>` and
:class:`SurvivorSelection <metaheuristic_designer.selection_methods.SurvivorSelection>` respectively.

To perturb individuals we use operators that are instances of :class:`Operator <metaheuristic_designer.Operator>` which take a population
and return a new population of modified individuals. This could represent a crossover operation, a mutation, generating a completely random individual or
even a sequence of operators. For the full catalogue of available operators see the :doc:`Operator Methods <api_reference.methods>` page.

We define a **search strategy** as the way of combining the previous elements and specifying how each iteration or step is carried out,
implemented as an instance of :class:`SearchStrategy <metaheuristic_designer.SearchStrategy>`.

The actual optimization is carried out by repeating steps until a stopping condition is reached. This is implemented as an instance of 
:class:`Algorithm <metaheuristic_designer.Algorithm>` which also provides progress information and is the interface used for the optimization.

All of these components have working implementations in their respective subpackages except for individuals, which are already implemented since
there is not much room for customization there.

There are also some default implementations of popular optimization algorithms available in the :any:`metaheuristic_designer.simple` module, such 
as genetic algorithms, particle swarm and simulated annealing. These are provided as convenience functions that take a dictionary of parameters.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

    API reference <api_reference>
    Operators and selection methods <api_reference.methods>
    Module Details <auto/modules>