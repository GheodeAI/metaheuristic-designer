====================================
API reference, Implemented Operators
====================================

Implemented Operators
=====================

.. _operator-methods:

Operator Methods
================

All methods are accessible through :py:func:`create_operator<metaheuristic_designer.operators.any_operator.create_operator>` using the ``"category.method"`` string.

The following categories are available:

- ``"mutation"``: alter existing values
- ``"crossover"``:  recombine multiple parents
- ``"permutation"``:  operators designed for permutation-based genotypes
- ``"de"``: Differential Evolution mutation variants
- ``"swarm"``: swarm intelligence specific operators (PSO, Firefly)
- ``"random"``: replace values with random ones using initializers
- ``"debug"``: placeholder operators used to diagnose optimization logic
- ``"custom"``: user-registered operators (see :py:func:`add_operator_entry<metaheuristic_designer.operators.any_operator.add_operator_entry>`)

Within each category, many **aliases** are defined for convenience. The tables below list
the primary name and all recognized aliases, along with a description and the parameters
they accept. Parameters can be passed as keyword arguments to :py:func:`create_operator<metaheuristic_designer.operators.any_operator.create_operator>`
and will override defaults.

----

Mutation
--------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "mutation.gaussian_mutation", "gauss_mut, normal_mutation", "Add Gaussian noise to existing values.", "loc (0), scale (1), N (all)"
   "mutation.uniform_mutation", "uniform_mut", "Replace components with uniform random numbers.", "low (-1), high (1), N (all)"
   "mutation.gaussian_noise", "gauss, normal, normal_noise", "Replace with new values from a Gaussian distribution.", "loc, scale"
   "mutation.laplace_mutation", "laplace_mut, laplace_mutation", "Add Laplace noise.", "loc (0), scale (1), N (all)"
   "mutation.cauchy_mutation", "cauchy_mut, cauchy_mutation", "Add Cauchy noise.", "loc (0), scale (1), N (all)"
   "mutation.uniform", "uniform_noise", "Replace with uniform random values (independent resampling).", "low, high"
   "mutation.poisson_mutation", "poisson_mut, poisson_mutation", "Add Poisson noise.", "lam (1), N"
   "mutation.bernoulli_mutation", "bernoulli_mut, coinflip_mut, coinflip_mutation", "Adds binary values with Bernoulli probability.", "p (0.5), N"
   "mutation.coinflip", "coinflip_noise, coinflip", "Replace each component with a Bernoulli trial.", "p (0.5)"
   "mutation.additive_noise_mutation", "mutnoise, noise_mutation", "Add noise (distribution determined by extra kwargs `distrib`).", "loc, scale, N, distrib"
   "mutation.sampling_mutation", "mutsample, replacement_mutation", "Replace some components with samples from a distribution.", "loc, scale, N, distrib"
   "mutation.full_additive_noise", "randnoise, random_noise, full_mutation", "Replace the entire genotype with noisy values.", "loc, scale, distrib"
   "mutation.full_random_sampling", "randsample, random_sampling, regenerate", "Replace the entire genotype with new samples from a distribution.", "loc, scale, distrib"
   "mutation.mutate_1_sigma", "mutate1sigma", "Self-adaptation: mutate a single sigma stored in the genotype (requires ParameterExtendingEncoding).", "epsilon (1e-10), tau (1.0)"
   "mutation.mutate_n_sigmas", "mutatensigmas", "Self-adaptation: mutate per-variable sigmas.", "epsilon (1e-10), tau (1.0), tau_multiple (1.0)"
   "mutation.sample_1_sigma", "sample1sigma", "Replace genotype with a sample from a Gaussian using the stored sigma.", "epsilon, tau, n (all)"

Many operators accept a ``distrib`` parameter to choose a probability distribution.
See :ref:`Probability Distributions <probability-distributions>` for the complete list.

----

Crossover
---------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "crossover.one_point_crossover", "1point, onepoint, one_point", "Exchange segments after a single crossing point.", ""
   "crossover.two_point_crossover", "2point, twopoint, two_point", "Exchange the middle segment between two points.", ""
   "crossover.uniform_crossover", "multipoint, uniform", "Swap each component independently with probability 0.5.", "p (0.5)"
   "crossover.multi_parent_discrete_crossover", "multicross, multi_parent, multi_parent_crossover", "For each position, choose a random parent's value.", "N (3)"
   "crossover.average_crossover", "avgcross, averagecross, arithmetic_crossover, intermediate_crossover", "Arithmetic mean of parents.", "alpha (0.5)"
   "crossover.intermediate_avg", "crossinteravg, interavg, multi_parent_avg", "Weighted average of multiple parents.", "N (3), alpha"
   "crossover.blx_alpha_crossover", "blxalpha, blx_alpha", "Blend crossover (BLX-α) for real values.", "alpha (0.5), low, high"
   "crossover.sbx_crossover", "sbx, simulated_binary, simulated_binary_crossover", "Simulated binary crossover for real values.", "F (1)"
   "crossover.bitwise_xor_crossover", "xorcross, xor_crossover, bitwise_xor, flipcross, bitflip_cross", "Bitwise XOR for binary genotypes.", ""

All crossover methods work on pairs (or groups) of parents selected before the operation.

----

Permutation operators
---------------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "permutation.swap", "swap_mutation, two_swap", "Swap two random positions.", "N (2)"
   "permutation.scramble", "scramble_mutation, perm, permutate, permutation_mutation, permute_components", "Randomly reorder a segment.", ""
   "permutation.invert", "reverse, inversion_mutation", "Reverse the order of a subsequence.", ""
   "permutation.roll", "roll_mutation, cyclic_shift", "Cyclically shift the genotype.", "N (1)"
   "permutation.pmx", "pmx_crossover, partially_mapped_crossover", "Partially mapped crossover for permutations.", ""
   "permutation.ox", "order_cross, order_crossover", "Order crossover for permutations.", ""
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
   "DE/current-to-rand/1", "de_current_to_rand_1, DE.current-to-rand.1", "F, Cr"
   "DE/current-to-best/1", "de_current_to_best_1, DE.current-to-best.1", "F, Cr"
   "DE/current-to-pbest/1", "de_current_to_pbest_1, DE.current-to-pbest.1", "F (0.8), Cr (0.9), p (0.1)"

Note: the string key can be written with dots or underscores.

----

Swarm operators
---------------

.. csv-table::
   :header: "Primary name", "Aliases", "Description", "Parameters"

   "swarm.pso", "pso_operator", "Particle Swarm velocity and position update.", "w (0.7), c1 (1.5), c2 (1.5)"
   "swarm.firefly", "", "Firefly algorithm movement.", "alpha_0 (0.2), beta_0 (1.0), delta (1.0), gamma (1.0)"

These operators require a `ParameterExtendingEncoding` (e.g., `PSOEncoding`) and
an `ExtendedInitializer` to manage the extra data (velocity, etc.).

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

    from metaheuristic_designer.operators import add_operator_entry

    def my_operator(population_matrix, fitness_array, random_state, **kwargs):
        ...

    add_operator_entry(my_operator, "myop", "custom")

    op = create_operator("custom.myop")

.. _probability-distributions:

Probability Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^
Operator methods that involve randomness choose the underlying probability distribution
via the ``distrib`` parameter.  The name is a case-insensitive string.

.. csv-table::
   :header: "Distribution name", "Parameters", "Description"

   "Uniform", "min, max", "Uniform distribution between ``min`` and ``max`` (mapped internally to ``loc=min, scale=max-min``)."
   "Gauss, Gaussian, Normal", "loc, scale", "Normal (Gaussian) distribution with mean ``loc`` and standard deviation ``scale``."
   "Multivariate Normal, Multigauss, …", "mean, cov (or loc, scale)", "Multivariate normal distribution. If ``loc``/``scale`` are scalars, an isotropic covariance is built; otherwise pass ``mean`` and ``cov`` arrays."
   "Cauchy", "loc, scale", "Cauchy distribution."
   "Laplace", "loc, scale", "Laplace distribution."
   "Gamma", "a, loc, scale", "Gamma distribution (shape ``a``)."
   "Exp, Expon, Exponential", "loc, scale", "Exponential distribution."
   "LevyStable, levy_stable", "a, b, loc, scale", "Lévy-stable distribution (stability ``a``, skewness ``b``)."
   "Tikhonov, vonMises, vonMises-Fisher", "mu, scale", "von Mises-Fisher distribution (kappa = 1/``scale``). ``mu`` must be a direction vector."
   "Poisson", "mu, loc", "Poisson distribution with mean ``mu``."
   "Bernoulli", "p", "Bernoulli distribution (probability of success ``p``)."
   "Binomial", "n, p, loc", "Binomial distribution (``n`` trials, success probability ``p``)."
   "Categorical", "p", "Categorical distribution given by a probability vector ``p``."
   "Multivariate Categorical, Multicategorical", "p", "Multivariate categorical distribution; ``p`` is a 2-D array of probabilities (rows sum to 1)."
   "Custom", "distrib_class (any scipy distribution)", "Any user-provided probability distribution from ``scipy.stats`` (the class itself, not an instance)."

When a parameter can be an array, its shape must be compatible with the population matrix
(number of individuals x number of variables).

.. note::

   The special **self-adaptation operators** (``mutate_1_sigma``, ``mutate_n_sigmas``,
   ``sample_1_sigma``) use their own internal parameters (``epsilon``, ``tau``, …)
   and **do not** accept a ``distrib`` argument.

.. _selection-methods:

Implemented Selection Methods
=============================

Parent and survivor selection are created through dedicated **factory functions**:

* :py:func:`create_parent_selection<metaheuristic_designer.parent_selection_methods.parent_selection.create_parent_selection>` – returns a :py:class:`ParentSelection<metaheuristic_designer.parent_selection.ParentSelection>` instance.
* :py:func:`create_survivor_selection<metaheuristic_designer.survivor_selection_methods.survivor_selection.create_survivor_selection>` – returns a :py:class:`SurvivorSelection<metaheuristic_designer.survivor_selection.SurvivorSelection>` instance.

Both accept a case-insensitive method name as the first argument, followed by any
method-specific parameters as keyword arguments.

To **skip** a selection step entirely, use the
:py:class:`NullParentSelection<metaheuristic_designer.parent_selection.NullParentSelection>` / :py:class:`NullSurvivorSelection<metaheuristic_designer.survivor_selection.NullSurvivorSelection>` classes.

Parent Selection
----------------

.. function:: create_parent_selection(method, amount=None, ...)

   Available method keys (case-insensitive). The primary (recommended) name is listed first;
   aliases are shown in the second column. Parameters can be passed as keyword arguments.

   .. csv-table::
      :header: "Method (primary)", "Aliases", "Parameters", "Description"

      "``tournament_selection``", "``tournament``", "``amount`` (int), ``tournament_size`` (int, default 3), ``p`` (float, default 1.0)", "Tournament selection: pick ``tournament_size`` random individuals and keep the best with probability ``p`` (default 1.0 ensures the best always wins)."
      "``probabilistic_tournament``", "", "``amount`` (int), ``tournament_size`` (int, default 3), ``p`` (float, default 0.5)", "Tournament selection where the winner is chosen with probability ``p`` (default 0.5)."
      "``select_best``", "``best``, ``truncation``", "``amount`` (int)", "Select the ``amount`` individuals with the highest fitness."
      "``uniform``", "``random``", "``amount`` (int)", "Uniformly random selection without replacement."
      "``roulette``", "", "``amount`` (int), ``method`` (str, see :ref:`roulette-weighting`), ``F`` (float)", "Fitness-proportionate (roulette wheel) selection. The weighting method can be chosen via the ``method`` parameter."
      "``fitness_proportional``", "", "``amount`` (int), ``F`` (float)", "Roulette selection with **fitness proportional** weighting (shortcut for ``roulette, method='fitness_prop'``)."
      "``sigma_scaling``", "``std_roulette``", "``amount`` (int), ``F`` (float)", "Roulette selection with **sigma scaling** weighting."
      "``linear_rank``", "``rank_roulette``", "``amount`` (int), ``F`` (float)", "Roulette selection with **linear rank** weighting."
      "``exponential_rank``", "``exp_rank_roulette``", "``amount`` (int), ``F`` (float)", "Roulette selection with **exponential rank** weighting."
      "``stochastic_universal_sampling``", "``sus``", "``amount`` (int), ``method`` (str, see :ref:`roulette-weighting`), ``F`` (float)", "Stochastic universal sampling. Same weighting options as ``roulette``."
      "``sus_fitness_proportional``", "``sus_fit_prop``, ``sus_proportional``, ``sus_prop``", "``amount`` (int), ``F`` (float)", "SUS with **fitness proportional** weighting."
      "``sus_sigma``", "``sus_std``", "``amount`` (int), ``F`` (float)", "SUS with **sigma scaling** weighting."
      "``sus_rank``", "", "``amount`` (int), ``F`` (float)", "SUS with **linear rank** weighting."
      "``sus_exponential``", "``sus_exp``", "``amount`` (int), ``F`` (float)", "SUS with **exponential rank** weighting."

   .. note::

      The convenience shortcuts (``fitness_proportional``, ``sigma_scaling``, …,
      and their SUS counterparts) fix the weighting method internally and do **not**
      require a ``method`` argument. Use the base ``roulette`` / ``sus`` names when
      you need to supply the ``method`` explicitly.

.. _roulette-weighting:

Roulette & SUS weighting methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``"roulette"`` or ``"sus"``, the ``method`` parameter selects how fitness is
mapped to selection probabilities. The available values are **case-insensitive**:

.. csv-table::
   :header: "``method`` value", "Description"

   "``fitness_prop``", "Fitness proportional scaling. The minimum fitness is subtracted to avoid negative values, then an offset ``F`` is added."
   "``sigma_scaling``", "Sigma scaling: weights are based on standard deviations above the mean."
   "``lin_rank``", "Linear ranking: the weight of the i-th best individual is proportional to its rank."
   "``exp_rank``", "Exponential ranking: weights are exponentially decayed with rank."

If no ``method`` is given, plain fitness proportional scaling is applied.

Survivor Selection
------------------

.. function:: create_survivor_selection(method, ...)

   Available method keys (case-insensitive), with the recommended explicit name shown first.

   .. csv-table::
      :header: "Method (primary)", "Aliases", "Parameters", "Description"

      "``elitism``", "", "``amount`` (int)", "Preserve the ``amount`` fittest parents and fill the remaining slots with the best offspring."
      "``conditional_elitism``", "``cond_elitism``", "``amount`` (int)", "Like ``elitism``, but only preserves elites if the offspring's best fitness is worse than the parent's best."
      "``generational``", "``nothing``", "", "Discard parents; the offspring becomes the new population."
      "``one_to_one``", "``hillclimb``, ``hill_climb``", "", "Each offspring competes against its corresponding parent; the winner stays."
      "``probabilistic_one_to_one``", "``prob_one_to_one``, ``prob_hillclimb``, ``prob_hill_climb``, ``probabilistic_hillclimb``, ``probabilistic_hill_climb``", "``p`` (float)", "Like ``one_to_one``, but the offspring wins with probability ``p`` regardless of fitness."
      "``many_to_one``", "``local_search``", "", "Each parent is compared to the corresponding offspring (element-wise) and the better one is kept."
      "``probabilistic_many_to_one``", "``prob_many_to_one``, ``prob_local_search``, ``probabilistic_local_search``", "``p`` (float)", "Like ``many_to_one``, but the offspring wins with probability ``p`` even if it is worse."
      "``keep_best``", "``(m+n)``, ``(mu+lambda)``, ``mu+lambda``", "", "Keep the best ``pop_size`` individuals from the union of parents and offspring."
      "``keep_offspring``", "``(m,n)``, ``(mu,lambda)``, ``mu,lambda``", "", "Replace the whole population with the best individuals from the offspring (offspring size must be ≥ population size)."