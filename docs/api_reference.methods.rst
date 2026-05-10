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
:func:`add_operator_entry`, :func:`add_parent_selection_entry`,
:func:`add_survivor_selection_entry`, or :func:`add_distribution_entry`.

.. _operator-methods:

Implemented Operators
=====================

All operators are created through the factory function
:py:func:`create_operator<metaheuristic_designer.operators.factories.generic.create_operator>`
using a ``"category.method"`` string (or just ``"method"`` when unambiguous).
The available categories are:

* ``"mutation"`` – alter existing values
* ``"crossover"`` – recombine multiple parents
* ``"permutation"`` – operators for permutation‑based genotypes
* ``"de"`` – Differential Evolution mutation variants
* ``"swarm"`` – swarm intelligence specific operators (PSO, …)
* ``"random"`` – replace values with random ones using initializers
* ``"debug"`` – placeholder operators for testing
* ``"custom"`` – user‑registered operators (via :py:func:`add_operator_entry<metaheuristic_designer.operators.factories.generic.add_operator_entry>`)

Within each category, many **aliases** are defined.  The tables below list every
built‑in operator along with its parameters.  Parameters shown with a value in
parentheses are defaults; they can be overridden by passing keyword arguments to
:func:`create_operator`.

----

Mutation
--------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "mutation.gaussian_mutation", "gauss_mut, normal_mutation", "Add Gaussian noise to existing values.", "loc (0), scale (1), N (all)"
   "mutation.uniform_mutation", "uniform_mut", "Add uniform random noise to components.", "low (-1), high (1), N (all)"
   "mutation.gaussian_noise", "gauss, normal, normal_noise", "Replace with new values from a Gaussian distribution.", "loc, scale"
   "mutation.laplace_mutation", "laplace_mut, laplace_mutation", "Add Laplace noise.", "loc (0), scale (1), N (all)"
   "mutation.cauchy_mutation", "cauchy_mut, cauchy_mutation", "Add Cauchy noise.", "loc (0), scale (1), N (all)"
   "mutation.uniform", "uniform_noise", "Replace with uniform random values (independent resampling).", "low, high"
   "mutation.poisson_mutation", "poisson_mut, poisson_mutation", "Add Poisson noise.", "lam (1), N"
   "mutation.bernoulli_mutation", "bernoulli_mut, coinflip_mut, coinflip_mutation", "Add binary values with Bernoulli probability.", "p (0.5), N"
   "mutation.coinflip", "coinflip_noise, coinflip", "Replace each component with a Bernoulli trial.", "p (0.5)"
   "mutation.additive_noise_mutation", "mutnoise, noise_mutation", "Add noise (distribution specified by `distribution`).", "loc, scale, N, distribution"
   "mutation.sampling_mutation", "mutsample, replacement_mutation", "Replace components with samples from a distribution.", "loc, scale, N, distribution"
   "mutation.full_additive_noise", "randnoise, random_noise, full_mutation", "Replace the entire genotype with noisy values.", "loc, scale, distribution"
   "mutation.full_random_sampling", "randsample, random_sampling, regenerate", "Replace the entire genotype with new samples from a distribution.", "loc, scale, distribution"
   "mutation.mutate_1_sigma", "mutate1sigma", "Self‑adaptation: mutate a single sigma stored in the genotype (requires ParameterExtendingEncoding).", "epsilon (1e-10), tau (1.0)"
   "mutation.mutate_n_sigmas", "mutatensigmas", "Self‑adaptation: mutate per‑variable sigmas.", "epsilon (1e-10), tau (1.0), tau_multiple (1.0)"
   "mutation.sample_1_sigma", "sample1sigma", "Replace genotype with a sample from a Gaussian using the stored sigma.", "epsilon, tau, n (all)"

Many mutation operators accept a ``distribution`` parameter to choose the
underlying probability distribution.  The available distribution names and their
parameters are described in the :ref:`probability-distributions` section below.

----

Crossover
---------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "crossover.one_point_crossover", "1point, onepoint, one_point", "Exchange segments after a single crossing point.", "crossover_prob (1), pairing_method (random), k (1)"
   "crossover.two_point_crossover", "2point, twopoint, two_point", "Exchange the middle segment between two points.", "crossover_prob (1), pairing_method (random), k (2)"
   "crossover.multipoint_crossover", "multipoint, kpoint, k_point", "k‑point crossover with configurable k.", "crossover_prob (1), pairing_method (random), k (2)"
   "crossover.uniform_crossover", "uniform", "Swap each component independently with probability 0.5.", "crossover_prob (1), pairing_method (random)"
   "crossover.multi_parent_discrete_crossover", "multicross, multi_parent, multi_parent_crossover", "For each position, choose a random parent's value.", "crossover_prob (1), k (3)"
   "crossover.average_crossover", "avgcross, averagecross, arithmetic_crossover, intermediate_crossover", "Arithmetic mean of parents.", "crossover_prob (1), pairing_method (random), alpha (0.5)"
   "crossover.blend_crossover", "blxalpha, blx_alpha, blend_crossover", "Blend crossover (BLX‑α) for real values.", "crossover_prob (1), pairing_method (random), alpha (0.5)"
   "crossover.sbx_crossover", "sbx, simulated_binary, simulated_binary_crossover", "Simulated binary crossover for real values.", "crossover_prob (1), pairing_method (random), eta (0.5)"
   "crossover.bitwise_xor_crossover", "xorcross, xor_crossover, bitwise_xor, flipcross, bitflip_cross", "Bitwise XOR for binary genotypes.", "crossover_prob (1), pairing_method (random)"
   "crossover.multiparent_intermediate_crossover", "crossinteravg, interavg, multiparent_avg, multiparent_intermediate_crossover", "Averaging recombination with multiple parents.", "crossover_prob (1), k (3)"

All dual‑parent crossovers accept a ``pairing_method`` parameter (``"random"``
or ``"stable"``) that controls how parents are paired.  The ``crossover_prob``
parameter (default 1) is applied per pair.

----

Permutation operators
---------------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "permutation.swap", "swap_mutation, two_swap", "Swap two random positions.", "N (2)"
   "permutation.scramble", "scramble_mutation, perm, permutate, permutation_mutation, permute_components", "Randomly reorder a segment.", ""
   "permutation.invert", "reverse, inversion_mutation", "Reverse the order of a subsequence.", ""
   "permutation.roll", "roll_mutation, cyclic_shift", "Cyclically shift the genotype.", "N (1)"
   "permutation.pmx", "pmx_crossover, partially_mapped_crossover", "Partially mapped crossover for permutations.", "pairing_method (random), crossover_prob (1)"
   "permutation.ox", "order_cross, order_crossover", "Order crossover for permutations.", "pairing_method (random), crossover_prob (1)"
   "permutation.shift", "insert, roll1, block_shift", "Remove a block and insert it elsewhere.", ""

These operators assume the genotype represents a permutation of integers.

----

Differential Evolution
----------------------

.. csv-table::
   :header: "Method", "Aliases", "Parameters"

   "DE/rand/1", "de_rand_1, de.rand.1", "F (0.8), Cr (0.9)"
   "DE/best/1", "de_best_1, de.best.1", "F, Cr"
   "DE/rand/2", "de_rand_2, de.rand.2", "F, Cr"
   "DE/best/2", "de_best_2, de.best.2", "F, Cr"
   "DE/current-to-rand/1", "de_current_to_rand_1, de.current-to-rand.1", "F, Cr"
   "DE/current-to-best/1", "de_current_to_best_1, de.current-to-best.1", "F, Cr"
   "DE/current-to-pbest/1", "de_current_to_pbest_1, de.current-to-pbest.1", "F (0.8), Cr (0.9), p (0.1)"

----

Swarm operators
---------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "swarm.pso", "pso_operator", "Particle Swarm velocity and position update.", "w (0.7), c1 (1.5), c2 (1.5)"

These operators require a :class:`~metaheuristic_designer.encodings.ParameterExtendingEncoding`
(e.g., :class:`~metaheuristic_designer.encodings.special.PSO_encoding.PSOEncoding`) and an
:class:`~metaheuristic_designer.initializers.extended_initializer.ExtendedInitializer` to manage
the extra data (velocity, etc.).

----

Random / reinitialization operators
-----------------------------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "random.reinitialize", "random, regenerate, full_random_reset", "Replace the entire population with fresh random individuals.", ""
   "random.reset", "reset_n, reset_random, reset_components", "Replace a subset of components with random values.", "n (1)"

----

Debug operators
---------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "debug.dummy", "debug, constant, set_to_value", "Set all genotype values to a constant `f`.", "f (0)"
   "debug.zeros", "", "Set all values to 0.", ""
   "debug.ones", "", "Set all values to 1.", ""

----

Custom operators
----------------

You can register your own operator functions and use them with the factory:

.. code-block:: python

    from metaheuristic_designer.operators import add_operator_entry, OperatorVectorDef
    from metaheuristic_designer.utils import MatrixLike, VectorLike, RNGLike

    def my_operator(population_matrix: MatrixLike, fitness_array: VectorLike, random_state: RNGLike, **kwargs) -> MatrixLike:
        ...

    add_operator_entry(OperatorVectorDef(my_operator), "myop", "custom")

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

.. csv-table::
   :header: "Distribution name", "Registry", "Aliases", "Parameters", "Description"

   "``norm``", "scipy-univar", "normal, gauss, gaussian", "loc, scale", "Normal (Gaussian) distribution with mean ``loc`` and standard deviation ``scale``."
   "``uniform``", "scipy-univar", "", "min, max (or loc, scale)", "Uniform distribution.  ``min``/``max`` are automatically converted to ``loc``/``scale``."
   "``cauchy``", "scipy-univar", "", "loc, scale", "Cauchy distribution."
   "``laplace``", "scipy-univar", "", "loc, scale", "Laplace distribution."
   "``gamma``", "scipy-univar", "", "a, loc, scale", "Gamma distribution (shape ``a``)."
   "``exponential``", "scipy-univar", "expon, exp", "loc, scale", "Exponential distribution."
   "``levy_stable``", "scipy-univar", "levy", "a, b, loc, scale", "Lévy‑stable distribution (stability ``a``, skewness ``b``)."
   "``poisson``", "scipy-univar", "", "mu, loc", "Poisson distribution with mean ``mu``."
   "``bernoulli``", "scipy-univar", "", "p", "Bernoulli distribution (probability of success ``p``)."
   "``binomial``", "scipy-univar", "binom", "n, p, loc", "Binomial distribution (``n`` trials, success probability ``p``)."
   "``categorical``", "scipy-univar", "", "p", "Categorical distribution given by a probability vector ``p``."
   "``tikhinov``", "scipy-univar", "vonmises", "mu, scale", "Von Mises distribution (kappa = 1/``scale``).  ``mu`` is the mean direction."
   "``multivariate_normal``", "scipy-multivar", "multigauss, multinormal, mvn", "mean, cov", "Multivariate normal distribution."
   "``dirichlet``", "scipy-multivar", "", "alpha", "Dirichlet distribution."
   "``vonmises_fisher``", "scipy-multivar", "", "mu, kappa", "Von Mises‑Fisher distribution.  ``mu`` is a direction vector."
   "``multicategorical``", "custom", "", "categories, weight_matrix", "Multivariate categorical distribution (per‑row probability weights)."

When a parameter can be an array, its shape must be compatible with the
population matrix (number of individuals × number of variables).  The special
self‑adaptation operators (``mutate_1_sigma``, ``mutate_n_sigmas``,
``sample_1_sigma``) use their own internal parameters and **do not** accept a
``distribution`` argument.

.. _selection-methods:

Implemented Selection Methods
=============================

Parent and survivor selection are created through dedicated **factory functions**:

* :py:func:`create_parent_selection<metaheuristic_designer.parent_selection.parent_selection.create_parent_selection>` – returns a :py:class:`ParentSelection<metaheuristic_designer.parent_selection.ParentSelection>` instance.
* :py:func:`create_survivor_selection<metaheuristic_designer.survivor_selection.survivor_selection.create_survivor_selection>` – returns a :py:class:`SurvivorSelection<metaheuristic_designer.survivor_selection.SurvivorSelection>` instance.

Both accept a case‑insensitive method name as the first argument, followed by any
method‑specific parameters as keyword arguments.

To **skip** a selection step entirely, use the
:py:class:`NullParentSelection<metaheuristic_designer.parent_selection.NullParentSelection>` /
:py:class:`NullSurvivorSelection<metaheuristic_designer.survivor_selection.NullSurvivorSelection>` classes.

Parent Selection
----------------

.. function:: create_parent_selection(method, amount=None, ...)

   Available method keys (case‑insensitive).  The primary name is listed first;
   aliases are shown in the second column.

   .. csv-table::
      :header: "Method (primary)", "Aliases", "Parameters", "Description"

      "``tournament``", "``tournament_selection``", "``amount`` (int), ``tournament_size`` (int, default 3), ``prob`` (float, default 1.0)", "Tournament selection: pick ``tournament_size`` random individuals and keep the best with probability ``prob`` (default 1.0 ensures the best always wins)."
      "``probabilistic_tournament``", "", "``amount`` (int), ``tournament_size`` (int, default 3), ``prob`` (float, default 0.5)", "Tournament where the winner is chosen with probability ``prob`` (default 0.5)."
      "``best``", "``truncation``, ``select_best``", "``amount`` (int)", "Select the ``amount`` individuals with the highest fitness."
      "``random``", "``uniform``", "``amount`` (int)", "Uniformly random selection with replacement."
      "``random_without_replacement``", "``shuffle``, ``permute``, ``random_subset``", "``amount`` (int)", "Random selection without replacement (shuffle)."
      "``roulette``", "", "``amount`` (int), ``scaling_factor`` (float), ``method`` (str, see :ref:`roulette-weighting`)", "Fitness‑proportionate (roulette wheel) selection.  Weighting method selected via ``method``."
      "``fitness_proportional``", "", "``amount`` (int), ``scaling_factor`` (float)", "Roulette with fitness proportional weighting."
      "``sigma_scaling``", "``std_roulette``", "``amount`` (int), ``scaling_factor`` (float)", "Roulette with sigma scaling."
      "``linear_rank``", "``rank_roulette``", "``amount`` (int), ``scaling_factor`` (float)", "Roulette with linear ranking."
      "``exponential_rank``", "``exp_rank_roulette``", "``amount`` (int), ``scaling_factor`` (float)", "Roulette with exponential ranking."
      "``sus``", "``stochastic_universal_sampling``", "``amount`` (int), ``scaling_factor`` (float), ``method`` (str, see :ref:`roulette-weighting`)", "Stochastic universal sampling.  Same weighting options as ``roulette``."
      "``sus_fitness_proportional``", "``sus_fit_prop``, ``sus_proportional``, ``sus_prop``", "``amount`` (int), ``scaling_factor`` (float)", "SUS with fitness proportional weighting."
      "``sus_sigma``", "``sus_std``", "``amount`` (int), ``scaling_factor`` (float)", "SUS with sigma scaling."
      "``sus_rank``", "", "``amount`` (int), ``scaling_factor`` (float)", "SUS with linear ranking."
      "``sus_exponential``", "``sus_exp``", "``amount`` (int), ``scaling_factor`` (float)", "SUS with exponential ranking."

.. _roulette-weighting:

Roulette & SUS weighting methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``"roulette"`` or ``"sus"``, the ``method`` parameter selects how
fitness is mapped to selection probabilities.  The available values are
case‑insensitive:

.. csv-table::
   :header: "``method`` value", "Aliases", "Description"

   "``fitness_proportional``", "``fitness_prop``", "Fitness proportional scaling.  Minimum fitness is subtracted and an offset ``scaling_factor`` is added."
   "``sigma_scaling``", "", "Sigma scaling: weights based on standard deviations above the mean."
   "``linear_rank``", "``lin_rank``", "Linear ranking: weight proportional to rank."
   "``exponential_rank``", "``exp_rank``", "Exponential ranking: weight decays exponentially with rank."

If no ``method`` is given, ``fitness_proportional`` is applied.

Survivor Selection
------------------

.. function:: create_survivor_selection(method, ...)

   Available method keys (case‑insensitive).

   .. csv-table::
      :header: "Method (primary)", "Aliases", "Parameters", "Description"

      "``elitism``", "", "``amount`` (int)", "Preserve the ``amount`` fittest parents; fill remaining slots with best offspring."
      "``conditional_elitism``", "``cond_elitism``", "``amount`` (int)", "Like ``elitism``, but only preserves elites if offspring's best fitness is worse."
      "``generational``", "``nothing``", "", "Discard parents; the offspring become the new population."
      "``one_to_one``", "``hillclimb``, ``hill_climb``", "", "Each offspring competes against its corresponding parent; the winner stays."
      "``probabilistic_one_to_one``", "``prob_one_to_one``, ``prob_hillclimb``, ``prob_hill_climb``, ``probabilistic_hillclimb``, ``probabilistic_hill_climb``", "``p`` (float)", "Like ``one_to_one``, but the offspring wins with probability ``p`` regardless of fitness."
      "``many_to_one``", "``local_search``", "", "Each parent is compared to several offspring; the best among them survives."
      "``probabilistic_many_to_one``", "``prob_many_to_one``, ``prob_local_search``, ``probabilistic_local_search``", "``p`` (float)", "Like ``many_to_one``, but the winner is chosen randomly with probability ``p``."
      "``keep_best``", "``(m+n)``, ``(mu+lambda)``, ``mu+lambda``", "", "Keep the best ``pop_size`` individuals from the union of parents and offspring."
      "``keep_offspring``", "``(m,n)``, ``(mu,lambda)``, ``mu,lambda``", "", "Replace the whole population with the best offspring (offspring size must be ≥ population size)."

For guidance on writing your own parent or survivor selection functions and
registering them with the factories, refer to :doc:`Custom Components <api_reference.custom_components>`.

.. _selection-discovery-note:

Need to see what’s available right now?  The functions
:func:`~metaheuristic_designer.parent_selection.list_parent_selection_methods`
and :func:`~metaheuristic_designer.survivor_selection.list_survivor_selection_methods`
return live lists – perfect for prototyping.