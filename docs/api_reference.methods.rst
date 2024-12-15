====================================
API reference, Implemented Operators
====================================

Implemented Operators
=====================

Operator Vector Methods
------------------------

These methods are accessed by instantiating the OperatorBinary class with any of the following methods as an attribute.
It is case insensitive and the parameters are mandatory, but if extra ones are added they will be ignored.

.. csv-table::
   :header: "Method name", "Params", "Description"

    "1point", "", "1 point crossover between 2 individuals.",
    "2point", "", "2 point crossover between 2 individuals.",
    "MultiPoint", "", "Multi point crossover between 2 individuals.",
    "WeightedAvg", "F (float)", "Weighted average bewteen 2 individuals.",
    "BLXAlpha", "CR (float)", "BLX-alpha crossover algorithm.",
    "SBX", "CR (float)", "SBX crossover algorithm.",
    "MultiCross", "Nindiv (int)", "Multipoint crossover between 'Nindiv' individuals.",
    "Xor, Flip", "N (int)", "Apply an XOR operator to 'N' components.",
    "XorCross, FlipCross", "", "XOR crossover between 2 individuals.",
    "CrossInterAvg", "N (int)", "IntOpMethods.CROSSINTERAVG",
    "MultiCross", "Nindiv (int)", "Multipoint crossover between 'Nindiv' individuals.",
    "Mutate1Sigma", "epsilon (float), tau(float)", "IntOpMethods.MULTICROSS",
    "MutateNSigmas", "epsilon (float), tau (float), tau_multiple (float)", "IntOpMethods.MULTICROSS",
    "Perm", "N (int)", "Permutate 'N' randomly selected components of a vector.",
    "Gauss", "F (float|ndarray)", "Add gaussian noise with mean 0 and std 1 multiplied by 'F'.",
    "Laplace", "F (float|ndarray)", "Add noise following a laplace distribution with mean 0 and std 1 multiplied by 'F'.",
    "Cauchy", "F (float|ndarray)", "Add noise following a laplace distribution with center 0 and scaling of 1 multiplied by 'F'.",
    "Uniform", "min (int|ndarray), max (int|ndarray)", "Add noise following an uniform distribution between 'min' and 'max'.",
    "Poisson", "F (float)", "Add noise following a poisson distribution with lambda equal to 'F'.",
    "MutNoise, MutRand", "N (int), F (float), distrib (str), :ref:`\[distrib params\]<Probability Distributions>`", "Add random noise following a given distribution on 'N' vector components multiplied by 'F'.",
    "MutSample, RandReset", "N (int), distrib (str), :ref:`\[distrib params\]<Probability Distributions>`", "Replace 'N' vector component with samples from a given probability distribution.",
    "RandNoise", "F (float), distrib (str), :ref:`\[distrib params\]<Probability Distributions>`", "Add random noise following a given distribution multiplied by 'F'.",
    "RandSample", "distrib (str), :ref:`\[distrib params\]<Probability Distributions>`", "Replace the vector with a sample from a given probability distribution.",
    "Generate", "statistic (str)", "Returns an individual which has as components the selected statistic of the parents."
    "DE/rand/1", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Rand/1.",
    "DE/best/1", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Best/1.",
    "DE/rand/2", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Rand/2.",
    "DE/best/2", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Best/2.",
    "DE/current-to-rand/1", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Current-to-Rand/1.",
    "DE/current-to-best/1", "F (float), Cr (float)", "DE (Differential Evolution) operator DE/Current-to-Best/1.",
    "DE/current-to-pbest/1", "F (float), Cr (float), p (float)", "DE (Differential Evolution) operator DE/Current-to-pRand/1.",
    "PSO", "w (float), c1 (float), c2 (float)", "Particle Swarm Optimization step.",
    "Firefly", "a (float), b (float), d (float), g (float)", "Firefly algorithm step.",
    "GlowWorm", "a (float), b (float), d (float), g (float)", "Glow worm algorithm step.",
    "Random", "", "Replace vector with a completely random vector.",
    "RandomMask", "N (int)", "Replace 'N' vector components with completely random values.",
    "Dummy", "F (ndarray)", "Replace vector with the predefined vector 'F'.",
    "Custom", "function (callable)", "Applies the given function to an individual.",
    "Nothing", "", "Keep input as is.",

Probability Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^
The operators that use probability distributions have to use one of the listed distrbutions.
The names are case insensitive and specified with the 'distrib' parameter.

.. csv-table::
   :header: "Distribution name", "Params", "Description"

   "Uniform", "max (float|ndarray), min (float|ndarray)", "`Uniform distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform>`_ in the interval [min, max]"
   "Gauss, Gaussian, Normal", "loc (float), scale (float)", "`Normal distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm>`_ with mean 'loc' and std 'scale'"
   "Cauchy", "loc (float|ndarray), scale (float|ndarray)", "`Cauchy distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html#scipy.stats.cauchy>`_"
   "Laplace", "loc (float|ndarray), scale (float|ndarray)", "`Laplace distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html#scipy.stats.laplace>`_"
   "Gamma", "a (float), loc (float|ndarray), scale (float|ndarray)", "`Gamma distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma>`_"
   "Exp, Expon, Exponential", "loc (float|ndarray), scale (float|ndarray)", "`Exponential distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon>`_"
   "LevyStable, levy_stable", "a (float), b (float), loc (float|ndarray), scale (float|ndarray)", "`Levy-Stable distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html#scipy.stats.levy_stable>`_"
   "Poisson", "mu (int|ndarray), loc (float|ndarray)", "`Poisson distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson>`_"
   "Bernoulli", "p (float|ndarray)", "`Bernoulli distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli>`_"
   "Binomial", "p (float|ndarray), n (int)", "`Binomial distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom>`_"
   "Custom", "distrib_class (scipy.stats.rv_generic), [distribution parameters]", "Predefined probability distribution. Any scipy probability distribution will work."



Operator Perm Methods
-----------------------

These methods are accessed by instantiating the OperatorBinary class with any of the following methods as an attribute.
It is case insensitive and the parameters are mandatory, but if extra ones are added they will be ignored.

.. csv-table::
   :header: "Method name", "Params", "Description"

   "Swap", "", "Swaps 2 vectors components."
   "Insert", "", "Inserts the last position of the vector in a random position."
   "Scramble, Perm", "N (int)", "Swaps 2 vectors components."
   "Invert", "", "Reverts the order of the components."
   "Roll", "N (int)", "Roll the bector components."
   "PMX", "", "Partially Mapped Crossover between 2 individuals."
   "OrderCross", "", "Ordered permutation crossover."
   "Random", "", "Generates a completely random permutation."
   "Dummy", "F (ndarray)", "Replace the vector with 'F'."
   "Custom", "function (callable)", "Apply the given function to an individual."
   "Nothing", "", "Keep input as is."


Operator Meta Methods
-----------------------

These methods are accessed by instantiating the OperatorBinary class with any of the following methods as an attribute.
The methods' names are case insensitive and the parameters are mandatory, but if extra ones are added they will be ignored.

.. csv-table::
   :header: "Method name", "Params", "Description"

   "Branch", "weights (ndarray) or p (float)", "Choose one of the operators at random."   
   "Sequence", "", "Apply all the operators in sequence."
   "Split", "mask (ndarray[int])", "Apply each operator to a subset of vector components."
   "Pick", "", "Manually pick one of the operators to apply by setting the 'chosen_idx' attribute of the Operator instance."


Implemented Selection Methods
=============================

Parent Selection
-----------------------

These methods are accessed by instantiating the OperatorBinary class with any of the following methods as an attribute.
It is case insensitive and the parameters are mandatory, but if extra ones are added they will be ignored.

.. csv-table::
   :header: "Method name", "Params", "Description"

   "Tournament", "amount (int), p (float)", "Tournament selection where 'amount' individuals compete and the best is selected, acepting a bad solution with probability 'p'."
   "Best", "amount (int)", "Take the best 'amount' individuals."
   "Random", "amount (int)", "Take 'amount' individuals at random."
   "Roullete", "amount (int), method (str), F (float)", "Perform roullete selection where the weight of each individual is determined by the method used."
   "SUS", "amount (int), method (str), F (float)", "Perform roullete selection where the weight of each individual is determined by the method used."
   "Nothing", "", "Choose the entire population as parents."

Roullette selection weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Therse are the weighting procedures available in the Roullete and SUS parent selection methods that will assign a 
probability of being chosen to each individual in the population.

.. csv-table::
   :header: "Method name", "Params", "Description"

   "FitnessProp", "", "Fitness proportional scaling. The minumum value is substracted to avoid negative values, then we add an offset of 'f' to each weight"
   "SigmaScaling", "", "Uses the sigma scaling procedure to generate the weights of each individual."
   "LinRank", "", "Takes the order of the individuals as their weight."
   "ExpRank", "", "Takes the exponential of the order of the individuals as their weigh."

Survivor Selection
-----------------------

These methods are accessed by instantiating the OperatorBinary class with any of the following methods as an attribute.
It is case insensitive and the parameters are mandatory, but if extra ones are added they will be ignored.

.. csv-table::
   :header: "Method name", "Params", "Description"

   "Elitism", "amount (int)", "Select 'amount' of the best parents and fill the rest of the population with the offspring."
   "CondElitism", "amount (int)", "Select 'amount' of the best parents and fill the rest of the population with the offspring."
   "One-to-one, HillClimb", "", "Compare each individual with their parent and choose the one with the best fitness."
   "Prob-one-to-one, ProbHillClimb", "p (float)", "Compare each individual with their parent and choose the one with the best fitness accepting the children either way with probability 'p'."
   "Many-to-one, LocalSearch", "", "Compare each individual with their parent and choose the one with the best fitness."
   "Prob-Many-to-one, ProbLocalSearch", "p (float)", "Compare each individual with their parent and choose the one with the best fitness accepting the children either way with probability 'p'."
   "(m+n), KeepBest", "", "Keep the best individuals combining the parents and their offspring."
   "(m,n), KeepOffspring", "", "Take the best individuals produced as the offspring."
   "CRO", "Fd (float), Pd (float), attempts (int), maxPopSize (int)", "Perform the CRO specific survivor selection."
   "Generational, Nothing", "", "Take the entire offspring."
