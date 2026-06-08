====================================
API reference – Implemented Methods
====================================

Discovering Available Components
--------------------------------

The library provides several **factory functions** to create operators and
selection methods by name.  You can explore all registered methods interactively
by calling the appropriate ``list_*`` functions:

.. code-block:: python

   from metaheuristic_designer.operators import list_operators
   from metaheuristic_designer.parent_selection import list_parent_selection_methods
   from metaheuristic_designer.survivor_selection import list_survivor_selection_methods
   from metaheuristic_designer.operators.operator_functions.probability_distributions_factory import list_distributions

   print(list_operators())                      # all operators (mutation, crossover, DE, …)
   print(list_parent_selection_methods())       # all parent selection methods
   print(list_survivor_selection_methods())     # all survivor selection methods
   print(list_distributions())                  # all available probability distributions

These lists grow automatically when you register custom components through
:py:func:`~metaheuristic_designer.operators.factories.generic.add_operator_entry`,
:py:func:`~metaheuristic_designer.parent_selection.parent_selection.add_parent_selection_entry`,
:py:func:`~metaheuristic_designer.survivor_selection.survivor_selection.add_survivor_selection_entry`, or
:py:func:`~metaheuristic_designer.operators.operator_functions.probability_distributions_factory.add_distribution_entry`.

.. _operator-methods:

Implemented Operators
=====================

All operators are created through the factory function
:py:func:`~metaheuristic_designer.operators.factories.generic.create_operator`
using a ``"category.method"`` string (or just ``"method"`` when unambiguous).
The available categories are:

* ``"mutation"`` – alter existing values (:ref:`mutation_operators`).
* ``"crossover"`` – recombine multiple parents (:ref:`crossover_operators`).
* ``"permutation"`` – operators for permutation‑based genotypes (:ref:`perm_operators`).
* ``"de"`` – Differential Evolution mutation variants (:ref:`de_operators`).
* ``"swarm"`` – swarm intelligence specific operators (:ref:`swarm_operators`).
* ``"random"`` – replace values with random ones using initializers (:ref:`random_operators`).
* ``"debug"`` – placeholder operators for testing (:ref:`debug_operators`)
* ``"custom"`` – user‑registered operators (:ref:`custom_operators`)

Within each category, many **aliases** are defined.  The tables below list every
built‑in operator along with its parameters.  Parameters shown with a value in
parentheses are defaults; they can be overridden by passing keyword arguments to
:func:`create_operator`.

----

.. _mutation_operators:

Mutation
--------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - mutation.gaussian_noise
     - | normal_mutation
       | gauss_noise
       | gaussian
       | normal
       | gauss       
     - Add Gaussian noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - loc (0)
       | - scale (1)
   * - mutation.gaussian_mutation
     - | normal_mutation
       | gauss_mutation
       | gaussian_mut
       | normal_mut
       | gauss_mut
     - Add Gaussian noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - loc (0)
       | - scale (1)
   
   * - mutation.uniform_noise
     - | uniform
     - Add Gaussian noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - min (-1)
       | - max (1)
   * - mutation.uniform_mutation
     - | uniform_mut
     - Add Uniform noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - min (-1)
       | - max (1)
   
   * - mutation.laplace_mutation
     - | laplace_mut
     - Add Laplace noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - loc (0)
       | - scale (1)
   * - mutation.laplace_noise
     - | laplace
     - Add Laplace noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - loc (0)
       | - scale (1)
   
   * - mutation.cauchy_mutation
     - | cauchy_mut
     - Add Cauchy noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - loc (0)
       | - scale (1)
   * - mutation.cauchy_noise
     - | cauchy
     - Add Cauchy noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - loc (0)
       | - scale (1)
   
   * - mutation.poisson_mutation
     - | poisson_mut
     - Add Poisson noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - mu
       | - scale (1)
   * - mutation.poisson_noise
     - | poisson
     - Add Poisson noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - mu
       | - scale (1)
   
   * - mutation.bernoulli_mutation
     - | bernoulli_mut
     - Add Bernoulli noise to the entire vector multiplied by a scaling factor `F`.
     - | - F
       | - p
       | - loc (0)
   * - mutation.bernoulli_noise
     - | bernoulli
     - Add Bernoulli noise to `N` randomly selected components multiplied by a scaling factor `F`.
     - | - F
       | - N
       | - p
       | - loc (0)
   
   * - mutation.bernoulli_sample
     - | coinflip
     - Replaces the vector randomly with 0 or 1.
     - | - p
   * - mutation.bernoulli_reset
     - | coinflip_reset
     - Replaces `N` randomly selected components with 0 or 1.
     - | - N
       | - p


   * - mutation.additive_noise_mutation
     - | noise_mutation
       | mutnoise
     - | Add randomly distributed noise to `N` randomly selected components multiplied by a scaling factor `F`.
       | (distributions available in :ref:`probability-distributions`)
     - | - distribution
       | - F
       | - N
       | - `parameters of the distribution`
   * - mutation.sampling_mutation
     - | replacement_mutation
       | mutsample
     - | Replaces `N` randomly selected components with random samples from a distribution.
       | (distributions available in :ref:`probability-distributions`)
     - | - distribution
       | - N
       | - `parameters of the distribution`
   * - mutation.full_additive_noise
     - | additive_noise
       | full_mutation
       | random_noise
       | randnoise
     - | Add randomly distributed noise to the entire vector multiplied by a scaling factor `F`.
       | (distributions available in :ref:`probability-distributions`)
     - | - distribution
       | - F
       | - `parameters of the distribution`
   * - mutation.full_random_sampling
     - | full_resampling
       | random_sampling
       | randsample
       | regenerate
     - | Replace the entire genotype with new samples from a distribution. 
       | (distributions available in :ref:`probability-distributions`)
     - | - distribution
       | - `parameters of the distribution`
   
   * - mutation.xor
     - | byte_xor
       | int_xor
       | bit_xor
       | bitflip
     - | Applies an XOR operator between `N` components of the vector and a random mask.
     - | - mode ("bit", "byte", "int")
       | - N 

   * - mutation.mutate_1_sigma
     - mutate1sigma
     - Self-adaptation: mutate a single sigma stored in the genotype.
     - | - epsilon (1e-10)
       | - tau (1.0) 
   * - mutation.mutate_n_sigmas
     - mutatensigmas
     - Self-adaptation: mutate per-variable sigmas.
     - | - epsilon (1e-10)
       | - tau (1.0) 
       | - tau_multiple (1.0) 
   * - mutation.sample_1_sigma
     - sample1sigma
     - Replace genotype with a sample from a Gaussian using the stored sigma.
     - | - epsilon (1e-10)
       | - tau (1.0) 
       | - n (all) 

Many mutation operators accept a ``distribution`` parameter to choose the
underlying probability distribution.  The available distribution names and their
parameters are described in the :ref:`probability-distributions` section below.

----

.. _crossover_operators:

Crossover
---------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - crossover.one_point_crossover
     - | one_point
       | onepoint
       | 1point
     - Exchange segments after a single crossing point.
     - | - crossover_prob (1)
       | - pairing_method (random)
   * - crossover.two_point_crossover
     - | two_point
       | twopoint
       | 2point
     - Exchange the middle segment between two points.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
   * - crossover.k_point_crossover
     - | k-point_crossover
       | kpoint_crossover
       | k_point
       | k-point
       | kpoint
       | multipoint_crossover
       | multipoint
     - k-point crossover with a configurable number of cut points.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
       | - k
   * - crossover.uniform_crossover
     - uniform
     - Swap each component independently with probability 0.5.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
   * - crossover.average_crossover
     - | avgcross
       | averagecross
       | arithmetic_crossover
       | intermediate_crossover
     - Arithmetic mean of parents.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
       | - alpha (0.5)
   * - crossover.blend_crossover
     - | blxalpha
       | blx_alpha
       | blend_crossover
     - Blend crossover (BLX-:math:`\alpha`) for real values.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
       | - alpha (0.5)
   * - crossover.sbx_crossover
     - | sbx
       | simulated_binary
       | simulated_binary_crossover
     - Simulated binary crossover for real values.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
       | - eta (0.5)
   * - crossover.bitwise_xor_crossover
     - | xorcross
       | xor_crossover
       | bitwise_xor
       | flipcross
       | bitflip_cross
     - Bitwise XOR for binary genotypes.
     - | - crossover_prob (1.0)
       | - pairing_method ("random")
   * - crossover.multi_parent_discrete_crossover
     - | multicross
       | multi_parent
       | multi_parent_crossover
     - Perform uniform recombination between `k` parents.
     - | - crossover_prob (1.0)
       | - k (3)
   * - crossover.multiparent_intermediate_crossover
     - | crossinteravg
       | interavg
       | multiparent_avg
     - Perform arithmetic recombination between `k` parents.
     - | - crossover_prob (1.0)
       | - k (3)

All dual‑parent crossovers accept a ``pairing_method`` parameter (``"random"``
or ``"stable"``) that controls how parents are paired.  The ``crossover_prob``
parameter (default 1) is applied per pair.

----

.. _perm_operators:

Permutation operators
---------------------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - permutation.swap
     - | swap_mutation
       | two_swap
     - Swap two random positions.
     - | - N (2)

   * - permutation.scramble
     - | scramble_mutation
       | perm
       | permutate
       | permutation_mutation
       | permute_components
     - Randomly reorder a segment.
     -

   * - permutation.invert
     - | reverse
       | inversion_mutation
     - Reverse the order of a subsequence.
     -

   * - permutation.roll
     - | roll_mutation
       | cyclic_shift
     - Cyclically shift the genotype.
     - | - N (1)

   * - permutation.pmx
     - | pmx_crossover
       | partially_mapped_crossover
     - Partially mapped crossover for permutations.
     - | - pairing_method (random)
       | - crossover_prob (1)

   * - permutation.ox
     - | order_cross
       | order_crossover
     - Order crossover for permutations.
     - | - pairing_method (random)
       | - crossover_prob (1)

   * - permutation.shift
     - | insert
       | roll1
       | block_shift
     - Remove a block and insert it elsewhere.
     -

These operators assume the genotype represents a permutation of integers.

----

.. _de_operators:

Differential Evolution
----------------------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - DE/rand/1
     - de_rand_1
     - Classic rand/1 mutation and binomial crossover.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/best/1
     - de_best_1
     - Uses the best individual as base vector.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/rand/2
     - de_rand_2
     - Two difference vectors for increased perturbation.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/best/2
     - de_best_2
     - Best individual with two difference vectors.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/current-to-rand/1
     - de_current_to_rand_1
     - Blends the current individual with a random donor.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/current-to-best/1
     - de_current_to_best_1
     - Blends the current individual towards the best.
     - | - F (0.8)
       | - Cr (0.9)

   * - DE/current-to-pbest/1
     - de_current_to_pbest_1
     - Blends the current individual towards a p‑best.
     - | - F (0.8)
       | - Cr (0.9)
       | - p (0.1)

----

.. _swarm_operators:

Swarm operators
---------------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - swarm.pso
     - pso_operator
     - Particle Swarm velocity and position update.
     - | - w (0.7)
       | - c1 (1.5)
       | - c2 (1.5)

These operators require a :class:`~metaheuristic_designer.encodings.ParameterExtendingEncoding`
(e.g., :class:`~metaheuristic_designer.encodings.special.PSO_encoding.PSOEncoding`) and an
:class:`~metaheuristic_designer.initializers.extended_initializer.ExtendedInitializer` to manage
the extra data (velocity, etc.).

----

.. _random_operators:

Random / reinitialization operators
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - random.reinitialize
     - | random
       | regenerate
       | full_random_reset
     - Replace the entire population with fresh random individuals.
     -

   * - random.reset
     - | reset_n
       | reset_random
       | reset_components
     - Replace a subset of components with random values.
     - | - n (1)

----

.. _debug_operators:

Debug operators
---------------

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - debug.dummy
     - | debug
       | constant
       | set_to_value
     - Set all genotype values to a constant ``f``.
     - | - f (0)

   * - debug.zeros
     -
     - Set all values to 0.
     -

   * - debug.ones
     -
     - Set all values to 1.
     - 


----

.. _custom_operators:

Custom operators
----------------

You can register your own operator functions and use them with the factory:

.. code-block:: python

    from metaheuristic_designer.operators import add_operator_entry, OperatorVectorDef
    from metaheuristic_designer.utils import MatrixLike, VectorLike, RNGLike

    @OperatorFnDef
    def my_operator(population_matrix: MatrixLike, fitness_array: VectorLike, rng: RNGLike, **kwargs) -> MatrixLike:
        ...

    add_operator_entry(my_operator, "myop", "custom")

    op = create_operator("custom.myop")

For a complete walk‑through, including the required function signatures, see
:doc:`Custom Components <api_reference.custom_components>`.

.. _probability-distributions:

Probability Distributions
-------------------------

Mutation operators that involve randomness accept a ``distribution`` parameter.
The name is a case‑insensitive string that can optionally include the registry
prefix (e.g., ``"scipy-univar.norm"``).  Distributions are organised in three
sub‑registries: ``scipy-univar`` (standard univariate distributions),
``scipy-multivar`` (multivariate distributions), and ``custom`` (user‑defined
distributions).

To list all available distributions at runtime, call
:func:`~metaheuristic_designer.operators.operator_functions.probability_distributions_factory.list_distributions`.
To register a new one, use
:func:`~metaheuristic_designer.operators.operator_functions.probability_distributions_factory.add_distribution_entry`.

.. list-table::
   :header-rows: 1

   * - Distribution name
     - Registry
     - Aliases
     - Short Description
     - Parameters

   * - ``norm``
     - scipy‑univar
     - | normal
       | gauss
       | gaussian
     - Normal (Gaussian) distribution.
     - | - loc (0)
       | - scale (1)

   * - ``uniform``
     - scipy‑univar
     -
     - Uniform distribution. Accepts ``min``/``max``, automatically converted to ``loc``/``scale``.
     - | - loc (0)
       | - scale (1)
       | - min & max (optional)

   * - ``cauchy``
     - scipy‑univar
     -
     - Cauchy distribution.
     - | - loc (0)
       | - scale (1)

   * - ``laplace``
     - scipy‑univar
     -
     - Laplace distribution.
     - | - loc (0)
       | - scale (1)

   * - ``gamma``
     - scipy‑univar
     -
     - Gamma distribution (shape ``a``).
     - | - a
       | - loc (0)
       | - scale (1)

   * - ``exponential``
     - scipy‑univar
     - | expon
       | exp
     - Exponential distribution.
     - | - loc (0)
       | - scale (1)

   * - ``levy_stable``
     - scipy‑univar
     - levy
     - Lévy‑stable distribution.
     - | - alpha (2)
       | - beta (0)
       | - loc (0)
       | - scale (1)

   * - ``poisson``
     - scipy‑univar
     -
     - Poisson distribution.
     - | - mu
       | - loc (0)

   * - ``bernoulli``
     - scipy‑univar
     -
     - Bernoulli distribution.
     - | - p
       | - loc (0)

   * - ``binomial``
     - scipy‑univar
     - binom
     - Binomial distribution.
     - | - n
       | - p
       | - loc (0)

   * - ``categorical``
     - scipy‑univar
     -
     - Categorical distribution given by a probability vector.
     - | - p

   * - ``tikhinov``
     - scipy‑univar
     - vonmises
     - Von Mises (circular) distribution. ``kappa`` (:math:`\kappa`) is converted to concentration :math:`\kappa = \frac{1}{\text{scale}}`.
     - | - kappa
       | - loc

   * - ``multivariate_normal``
     - scipy‑multivar
     - | multigauss
       | multinormal
       | mvn
     - Multivariate normal distribution.
     - | - mean
       | - cov

   * - ``dirichlet``
     - scipy‑multivar
     -
     - Dirichlet distribution.
     - | - alpha

   * - ``vonmises_fisher``
     - scipy‑multivar
     -
     - Von Mises‑Fisher distribution on the hypersphere.
     - | - mu
       | - kappa

   * - ``multicategorical``
     - custom
     -
     - Multivariate categorical distribution (per‑row probability weights).
     - | - categories
       | - weight_matrix

When a parameter can be an array, its shape must be compatible with the
population matrix (number of individuals × number of variables).  The special
self‑adaptation operators (``mutate_1_sigma``, ``mutate_n_sigmas``,
``sample_1_sigma``) use their own internal parameters and **do not** accept a
``distribution`` argument.

.. _selection-methods:

Implemented Selection Methods
-----------------------------

Parent and survivor selection are created through dedicated **factory functions**:

* :py:func:`~metaheuristic_designer.parent_selection.parent_selection.create_parent_selection` – returns a :py:class:`~metaheuristic_designer.parent_selection.ParentSelection` instance.
* :py:func:`~metaheuristic_designer.survivor_selection.survivor_selection.create_survivor_selection` – returns a :py:class:`~metaheuristic_designer.survivor_selection.SurvivorSelection` instance.

Both accept a case‑insensitive method name as the first argument, followed by any
method‑specific parameters as keyword arguments.

To **skip** a selection step entirely, use the
:py:class:`~metaheuristic_designer.parent_selection.NullParentSelection` /
:py:class:`~metaheuristic_designer.survivor_selection.NullSurvivorSelection` classes.

Parent Selection
================

.. function:: create_parent_selection(method, amount=None, ...)

   Available method keys (case‑insensitive).  The primary name is listed first;
   aliases are shown in the second column.

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - tournament
     - tournament_selection
     - Tournament selection: pick ``tournament_size`` random individuals and keep the best with probability ``prob`` (default 1.0 ensures the best always wins).
     - | - amount
       | - tournament_size (3)
       | - prob (1.0)

   * - probabilistic_tournament
     -
     - Tournament where the winner is chosen with probability ``prob`` (default 0.5).
     - | - amount
       | - tournament_size (3)
       | - prob (0.5)

   * - best
     - | truncation
       | select_best
     - Select the ``amount`` individuals with the highest fitness.
     - | - amount

   * - random
     - uniform
     - Uniformly random selection with replacement.
     - | - amount

   * - random_without_replacement
     - | shuffle
       | permute
       | random_subset
     - Random selection without replacement (shuffle).
     - | - amount

   * - roulette
     -
     - Fitness‑proportionate (roulette wheel) selection. Weighting method selected via ``method`` (see :ref:`roulette-weighting`).
     - | - amount
       | - scaling_factor
       | - method

   * - fitness_proportional
     -
     - Roulette with fitness proportional weighting.
     - | - amount
       | - scaling_factor

   * - sigma_scaling
     - std_roulette
     - Roulette with sigma scaling.
     - | - amount
       | - scaling_factor

   * - linear_rank
     - rank_roulette
     - Roulette with linear ranking.
     - | - amount
       | - scaling_factor

   * - exponential_rank
     - exp_rank_roulette
     - Roulette with exponential ranking.
     - | - amount
       | - scaling_factor

   * - sus
     - stochastic_universal_sampling
     - Stochastic universal sampling. Same weighting options as ``roulette``.
     - | - amount
       | - scaling_factor
       | - method

   * - sus_fitness_proportional
     - | sus_fit_prop
       | sus_proportional
       | sus_prop
     - SUS with fitness proportional weighting.
     - | - amount
       | - scaling_factor

   * - sus_sigma
     - sus_std
     - SUS with sigma scaling.
     - | - amount
       | - scaling_factor

   * - sus_rank
     -
     - SUS with linear ranking.
     - | - amount
       | - scaling_factor

   * - sus_exponential
     - sus_exp
     - SUS with exponential ranking.
     - | - amount
       | - scaling_factor

.. _roulette-weighting:

Roulette & SUS weighting methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``"roulette"`` or ``"sus"``, the ``method`` parameter selects how
fitness is mapped to selection probabilities.  The available values are
case‑insensitive:

.. list-table::
   :header-rows: 1

   * - method value
     - Aliases
     - Description
     - Parameters

   * - fitness_proportional
     - fitness_prop
     - Fitness proportional scaling. Minimum fitness is subtracted and an offset ``scaling_factor`` is added.
     - scaling_factor

   * - sigma_scaling
     -
     - Sigma scaling: weights based on standard deviations above the mean.
     - scaling_factor

   * - linear_rank
     - lin_rank
     - Linear ranking: weight proportional to rank.
     - scaling_factor

   * - exponential_rank
     - exp_rank
     - Exponential ranking: weight decays exponentially with rank.
     - scaling_factor

If no ``method`` is given, ``fitness_proportional`` is applied.

Survivor Selection
==================

.. function:: create_survivor_selection(method, ...)

   Available method keys (case‑insensitive).

.. list-table::
   :header-rows: 1

   * - Primary name
     - Aliases
     - Description
     - Parameters

   * - elitism
     -
     - Preserve the ``amount`` fittest parents; fill remaining slots with best offspring.
     - | - amount

   * - conditional_elitism
     - cond_elitism
     - Like ``elitism``, but only preserves elites if offspring's best fitness is worse.
     - | - amount

   * - generational
     - nothing
     - Discard parents; the offspring become the new population.
     -

   * - one_to_one
     - | hillclimb
       | hill_climb
     - Each offspring competes against its corresponding parent; the winner stays.
     -

   * - probabilistic_one_to_one
     - | prob_one_to_one
       | prob_hillclimb
       | prob_hill_climb
       | probabilistic_hillclimb
       | probabilistic_hill_climb
     - Like ``one_to_one``, but the offspring wins with probability ``p`` regardless of fitness.
     - | - p

   * - many_to_one
     - local_search
     - Each parent is compared to several offspring; the best among them survives.
     -

   * - probabilistic_many_to_one
     - | prob_many_to_one
       | prob_local_search
       | probabilistic_local_search
     - Like ``many_to_one``, but the winner is chosen randomly with probability ``p``.
     - | - p

   * - keep_best
     - | (m+n)
       | (mu+lambda)
       | mu+lambda
     - Keep the best ``pop_size`` individuals from the union of parents and offspring.
     -

   * - keep_offspring
     - | (m,n)
       | (mu,lambda)
       | mu,lambda
     - Replace the whole population with the best offspring (offspring size must be ≥ population size).
     -

registering them with the factories, refer to :doc:`Custom Components <api_reference.custom_components>`.

.. _selection-discovery-note:

Need to see what’s available right now?  The functions
:func:`~metaheuristic_designer.parent_selection.list_parent_selection_methods`
and :func:`~metaheuristic_designer.survivor_selection.list_survivor_selection_methods`
return live lists – perfect for prototyping.
