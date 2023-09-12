.. metaheuristic-designer documentation master file, created by
   sphinx-quickstart on Wed Jul 19 18:39:56 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to metaheuristic-designer's documentation!
======================================

Description
-----------
This is an object-oriented framework for the development, testing and analysis of metaheuristic optimization algorithms.

It defines the components of a general evolutionary algorithm and offers some implementations of algorithms along with components
that can be used directly. Those components will be explained below.

It was inspired by the article `Metaheuristics “in the large” <https://doi.org/10.1016/j.ejor.2021.05.042>`_ that 
discusses some of the issues in the research on metaheuristic optimization, sugesting the development of libraries for the standarization
of metaheuristic algorithms.

Most of the design decisions are based on the book `Introduction to evolutionary computing by Eiben, Agoston E.,
and James E. Smith <https://doi.org/10.1007/978-3-662-44874-8>`_ which is very well expained and is highly recomended to anyone willing to 
learn about the topic.

This framework doesn't claim to have a high performance, specially since the chosen language is Python and the code has not been 
designed for speed. This shouldn't really be an issue since the highest amount of time spent in these kind of algorithms
tends to be in the evaluation of the objective function. If you want to compare an algorithm made with this tool with another
one that is available by other means, it is recomended to use the number of evaluations of the objective function as a metric instead of 
execution time.   


Structure
---------

Explanation
~~~~~~~~~~~

The terminology used is inherited mainly from genetic algorithms and classical optimization algorithms.

The structure of the optimization algorithms are very much inspired on the development of the
`DPCRO_SL algorithm <https://doi.org/10.3390/math11071666>`_ of which I was a part of, which gave me the realization that evolutionary
algorithms and some other optimization algorithms tend share a common structure:

1. **Initialize** a random solution or population of solutions.

2. **Repeat** until a stopping condition is met (time passed, number of evaluations, iterations without improvement...).

   2.1. **Select parents** (trivial in algorithms with only one solution).

   2.2. **Perturb the parents** (might involve a sequence of operators like a cross and a mutation).

   2.3. **Select individuals** for the next iteration.

3. **Return the best solution**.

Implementation
~~~~~~~~~~~~~~

This is implemented as Interfaces (or more accurately, abstract classes) that are related to each other and which must be implemented
to constuct an optimization algorithm.

First, **objective functions** are implemented as instances of the class :class:`ObjectiveFunc <metaheuristic_designer.ObjectiveFunc>` which receive
an input in some unspecified format (an array, a tree or any other object) and output a single numerical value. Our goal is to find
an input that maximizes (or minimizes) this output value.

Each **solution** is represented as an instance of the class :class:`Individual <metaheuristic_designer.Individual>`, that is a class that holds a possible 
solution (its genotype) to our optimization problem and has a fitness which is the value of the objective function for this solution adjusted
so that we are always solving a maximization problem (flipping the sign for minimization problems).

The solutions that an individual has are encoded in a certain way, but our objective function might need an input encoded in a different way. 
This is where **encodings** are used, they are represented as instances of the class :class:`Encoding <metaheuristic_designer.Encoding>` which isolates the 
representation of the solution in the optimization prodecure and the one used in the calculation of the objective function.

For the **initialization step**, there will be an instance of the class :class:`Initializer <metaheuristic_designer.Initializer>` that will generate
an initial population, often completely at random, and will be used whenever a random solution needs to be generated. This class will
also indicate the size of the population (1 if the algorithm works with only one solution).

Both parent selection and survivor selection are implemented as instances of the class :class:`SelectionMethod <metaheuristic_designer.SelectionMethod>` 
although it is recomended to use the classes :class:`ParentSelection <metaheuristic_designer.SelectionMethods.ParentSelection>` and
:class:`SurvivorSelection <metaheuristic_designer.SelectionMethods.SurvivorSelection>` respectively.

To perturb individuals we use operators are instances of the class :class:`Operator <metaheuristic_designer.Operator>` which take an individual and the population
and returns a perturbed individual. This could represent a crossing operation, a mutation, generating a completely random individual or
even a sequence of operators.

We define an algorithm as the way of combining of the previous elements and specifies how each iteration or step is carried out and
is implemented as as instance of the class :class:`Algorithm <metaheuristic_designer.Algorithm>`.

The proper optimization is carried repeating steps until a stopping condition is reached, this is implemented as an instance of the class 
:class:`Search <metaheuristic_designer.Search>` which also provides some information about the progress of the algorithm and is the interface that is
used for the optimization.

All of these components have working implementations in their respective subpackages except for individuals, which are already implemented since
there is not much room for customization there.

There is also some default implementation of popular optimization algorithms available in the :any:`metaheuristic_designer.simple` such 
as genetic algorithms, particle swarm and simulated annealing.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
    API reference <metaheuristic_designer>