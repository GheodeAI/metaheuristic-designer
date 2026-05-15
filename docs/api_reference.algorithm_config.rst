.. _algorithm-object-config:

Configuring Algorithms
======================

The :class:`~metaheuristic_designer.algorithm.Algorithm` constructor accepts every
runtime setting in two forms:

* **direct constructor arguments** : pass numeric limits, reporter type, history flags,
  etc. as keyword arguments.  This is the quickest way to get started.
* **pre-built objects** : provide an explicit :class:`~metaheuristic_designer.stopping_condition.StoppingCondition`,
  :class:`~metaheuristic_designer.reporter.Reporter`, :class:`~metaheuristic_designer.history_tracker.HistoryTracker`, or :class:`~metaheuristic_designer.checkpointer.Checkpointer` instance.
  This gives you finer control, reusability across runs, and the ability to extend
  behaviour through custom subclasses.

Both styles can be mixed; parameters given as objects take precedence over raw values.
The rest of this page details the object-based approach because it makes all settings
visible in one place, but everything described also applies when you use the simpler
keyword-argument style.

Quick examples:

.. code-block:: python
   
   from metaheuristic_designer import Algorithm

   # Direct arguments : compact
   algo = Algorithm(
       objfunc,
       strategy,
       stop_cond="max_iterations or real_time_limit",
       max_iterations=500,
       real_time_limit=30.0,
       reporter="tqdm",
       track_median=True,
   )

   # Object-based : explicit and reusable
   from metaheuristic_designer.stopping_condition import StoppingCondition
   from metaheuristic_designer.reporters.tqdm_reporter import TQDMReporter
   from metaheuristic_designer.history_tracker import HistoryTracker

   stop = StoppingCondition(
       condition_str   = "max_iterations or real_time_limit",
       max_iterations = 500,
       real_time_limit = 30.0,
   )
   algo = Algorithm(
       objfunc,
       strategy,
       stopping_condition = stop,
       reporter           = TQDMReporter(),
       history_tracker    = HistoryTracker(track_median=True),
   )

Except for ``objfunc`` and ``strategy``, every other component is optional; if omitted the ``Algorithm`` creates sensible defaults.


Stopping Condition
------------------

A stopping condition is defined by a logical expression that combines **tokens** with
``and``, ``or`` and parentheses.  Every token that appears in the expression must have
its corresponding numeric limit provided; limits for unused tokens can be left as
``None``.

In the following table we indicate the token used in the ``stopping_condition`` string as well as the parameter that configures
when that condition in the :py:class:`~metaheuristic_designer.stopping_condition.StoppingCondition`

.. list-table:: Tokens and their meaning
   :widths: 20 20 60
   :header-rows: 1

   * - Token
     - Parmeter
     - Meaning
   * - ``max_evaluations``
     - ``max_evaluations``
     - Maximum number of objective function evaluations.
   * - ``max_iterations``
     - ``max_iterations``
     - Maximum number of iterations (generations).
   * - ``real_time_limit``
     - ``real_time_limit``
     - Wall-clock time limit in seconds.
   * - ``cpu_time_limit``
     - ``cpu_time_limit``
     - CPU time limit in seconds.
   * - ``objective_target``
     - ``objective_target``
     - Target value for the raw objective. For minimisation (``mode="min"``), the algorithm stops when ``best_objective <= objective_target``; for maximisation (``mode="max"``), when ``best_objective >= objective_target``.
   * - ``convergence``
     - ``max_patience``
     - Stops when the best fitness has not improved for ``max_patience`` consecutive iterations.  This token requires ``max_patience`` to be set.

**How to combine tokens**  
``and``, ``or``, and parentheses work exactly as in a logical expression:

* ``and`` : the algorithm stops only when **both** sides are satisfied.  
  Example ``"max_iterations and convergence"`` → stop when the maximum number of iterations has been reached **and** the convergence criterion is met.

* ``or`` : the algorithm stops as soon as **any** of the sides is satisfied.  
  Example ``"max_iterations or real_time_limit"`` → stop when either the iteration limit is reached or the time runs out, whichever happens first.

* Parentheses allow more complex combinations.  
  Example ``"(max_iterations or real_time_limit) and convergence"`` → stop only if (iterations or time) have passed **and** convergence is reached.

You can chain as many tokens as you like; the expression will be evaluated as a
boolean tree with ``and`` having higher precedence than ``or`` (like standard
Boolean algebra), so ``"a and b or c"`` is equivalent to ``"(a and b) or c"``.

**Note** : ``objective_target`` is compared directly against the raw objective that your
``ObjectiveFunc`` returns.  No internal conversion is applied.  Supply the target in
the same units and sign as your problem’s objective.

Create a ``StoppingCondition`` object:

.. code-block:: python

   from metaheuristic_designer.stopping_condition import StoppingCondition

   stop = StoppingCondition(
       condition_str       = "max_iterations or real_time_limit",
       progress_metric_str = None,               # defaults to condition_str
       max_iterations      = 500,
       real_time_limit     = 30.0,
       max_evaluations     = None,
       cpu_time_limit      = None,
       objective_target    = None,
       optimization_mode   = objfunc.mode,
   )

When using ``convergence``, provide ``max_patience``:

.. code-block:: python

   stop = StoppingCondition(
       condition_str    = "max_iterations and convergence",
       max_iterations   = 200,
       max_patience     = 50,
       optimization_mode= objfunc.mode,
   )

Or a target-driven stop:

.. code-block:: python

   stop = StoppingCondition(
       condition_str    = "objective_target",
       objective_target = 1e-10,
       optimization_mode= objfunc.mode,
   )

.. caution::

   If a token appears in the condition string but its limit is not
   supplied, an error is raised immediately.  For example, forgetting
   ``max_iterations`` when using ``"max_iterations"`` will produce a clear message
   telling you what is missing.  Conversely, parameters that are not required by the
   expression are ignored.

**Progress metric** : The ``progress_metric_str`` controls how the algorithm computes
its progress (a number between 0 and 1).  This progress value is **not** merely for
display; it is consumed internally by **parameter schedules** (e.g., linearly decaying
mutation strength) and by annealing strategies.  Therefore you should choose a
monotonic metric that reflects the expected convergence path of your algorithm.

Tokens like ``max_iterations``, ``real_time_limit``, ``max_evaluations`` and even
``objective_target`` are monotonic (the best objective only improves), so they produce
reliable progress curves.  The only token to avoid in ``progress_metric_str`` is
``convergence``, because the progress would reset each time the best fitness improves,
making parameter schedules erratic.

It is technically allowed to use ``convergence`` in the progress expression, but it
is discouraged.  If you need convergence as a stopping condition, confine it to
``condition_str`` and rely on a monotonic token for ``progress_metric_str``.

You can decouple the two expressions:

.. code-block:: python

   stop = StoppingCondition(
       condition_str       = "objective_target",
       progress_metric_str = "real_time_limit",
       objective_target    = 1e-5,
       real_time_limit     = 60.0,
   )


Reporter
--------

Reporters govern what the algorithm prints or displays.  Three implementations exist:

.. code-block:: python

   from metaheuristic_designer.reporters import TQDMReporter, VerboseReporter, SilentReporter

   # Progress bar (ideal for Jupyter / terminal)
   reporter = TQDMReporter(resolution=1000)

   # Periodic text output at a configurable interval
   reporter = VerboseReporter(verbose_timer=0.5)   # seconds between prints

   # No output
   reporter = SilentReporter()

The ``VerboseReporter`` prints a summary block at most once per ``verbose_timer``
interval.


History Tracker
---------------

The history tracker records metrics across iterations.  All flags default to ``False``
except ``track_best``, which is always active.

.. code-block:: python

   from metaheuristic_designer.history_tracker import HistoryTracker

   history = HistoryTracker(
       track_best            = True,    # always recorded
       track_median          = False,
       track_worst           = False,
       track_diversity       = False,
       track_parameters      = False,   # record scheduled parameter values
       track_full_objective  = False,   # store the full fitness vector per generation
       track_full_population = False,   # store entire population each generation
   )

* ``track_diversity`` adds a ``diversity`` column (e.g., average pairwise distance or
  genotype spread).
* ``track_parameters`` appends a column for each scheduled parameter (mutation strength,
  crossover probability, etc.) to the DataFrame returned by ``to_pandas()``.
* ``track_full_objective`` enables :meth:`to_pandas_full_objective`, which returns a
   wide-format DataFrame (one column per individual) for boxplots or violin plots.

.. warning::

   Activating ``track_full_population`` or ``track_full_objective`` can consume significant
   memory for long runs or large populations.  Use these options only when you need
   the full evolution trace or detailed fitness distributions.

   Be very careful to combine these flags with checkpoints, they can cause checkpoints to be multiple Gigabytes in size
   and storage might consume a long time.


When ``track_full_objective`` is enabled, you can retrieve the data as a DataFrame:

.. code-block:: python

   wide_df = algo.history_tracker.to_pandas_full_objective()
   # wide‑format: iteration, Individual_0, Individual_1, ... (one column per individual)


Checkpointer
------------

If you provide a :class:`~metaheuristic_designer.checkpointer.Checkpointer`, the
algorithm will periodically dump its entire state to disk.  This makes it possible to
resume interrupted runs.  The checkpointer also saves on ``SIGINT`` (Ctrl+C), so you
can manually stop a run and later reload it.

.. code-block:: python

   from metaheuristic_designer.checkpointer import Checkpointer

   check = Checkpointer(
       checkpoint_file         = "my_run.pkl",
       iteration_frequency     = 10,        # save every 10 iterations
       time_frequency          = 300.0,     # save every 5 minutes
   )

Only one of ``iteration_frequency`` or ``time_frequency`` needs to be set, and both
can be active at the same time.

To load a previous checkpoint and continue **you must call** :meth:`resume` **instead of** :meth:`optimize`:

.. code-block:: python

   algo = check.load(
       file_name       = "my_run.pkl",
       history_tracker = HistoryTracker(),   # reuse or fresh
       reporter        = TQDMReporter(),     # re-attach a reporter
   )

   # ⚠️  ABSOLUTELY DO NOT call algo.optimize() here, that would RESTART from scratch!
   algo = algo.resume()

.. warning::

   **DO NOT** call :meth:`~metaheuristic_designer.algorithm.Algorithm.optimize` on a loaded
   algorithm, it will **silently discard the checkpoint and start a brand-new run**!.

   Always use :meth:`~metaheuristic_designer.algorithm.Algorithm.resume` when you want to pick up where you left off.

   After loading, the reporter is **not** saved inside the
   checkpoint.  You must provide it again explicitly when calling ``load()``.
   Running without a reporter is possible but not recommended for long runs.


Fitness vs Objective: a Quick Guide
------------------------------------

The library internally converts every raw objective value to a **fitness** that is
**always maximised**.  This keeps the internal logic simple. Two methods help you
retrieve results in the appropriate representation:

* :meth:`~metaheuristic_designer.population.Population.best_solution`: returns the
  best decoded **solution** (phenotype) and its raw **objective** value.  Use this
  for reporting and for using the result in your application.
* :meth:`~metaheuristic_designer.population.Population.best_individual`: returns the
  best encoded **genotype** and its **fitness** value (the internal maximised metric).
  Use this when you need to inspect the internal representation or implement optimization
  subroutines behaviour.

These methods are available on both :class:`~metaheuristic_designer.population.Population`
and :class:`~metaheuristic_designer.algorithm.Algorithm` objects.  After a run, you
can call them directly on the algorithm instance.

.. code-block:: python

   # Access via the algorithm
   solution, objective = algo.best_solution()
   genotype, fitness   = algo.best_individual()

   # Access via the final population
   solution, objective = population.best_solution()
   genotype, fitness   = population.best_individual()

When you provide a stopping target, use ``objective_target`` with
:class:`~metaheuristic_designer.stopping_condition.StoppingCondition`.  The comparison is direct: for minimisation the algorithm
stops when ``best_objective <= objective_target``; for maximisation when
``best_objective >= objective_target``.  You do not need to think in fitness units
at all.

Putting It All Together
-----------------------

Here is a complete example that minimises the Sphere function with a genetic algorithm,
using object-based configuration.

.. code-block:: python

   import numpy as np
   from metaheuristic_designer import Algorithm, check_random_state
   from metaheuristic_designer.benchmarks import Sphere
   from metaheuristic_designer.strategies import GA
   from metaheuristic_designer.initializers import UniformInitializer
   from metaheuristic_designer.operators import create_operator
   from metaheuristic_designer.parent_selection import create_parent_selection
   from metaheuristic_designer.survivor_selection import create_survivor_selection
   from metaheuristic_designer.stopping_condition import StoppingCondition
   from metaheuristic_designer.history_tracker import HistoryTracker
   from metaheuristic_designer.reporters.tqdm_reporter import TQDMReporter
   from metaheuristic_designer.checkpointer import Checkpointer

   rng = check_random_state(42)
   objfunc = Sphere(5, mode="min")

   # Build the search strategy
   strategy = GA(
       initializer   = UniformInitializer(objfunc.dimension, objfunc.lower_bound, objfunc.upper_bound, population_size=100, random_state=rng),
       mutation_op   = create_operator("mutation.gaussian_mutation", N=1, F=0.1, random_state=rng),
       crossover_op  = create_operator("crossover.uniform_crossover", random_state=rng),
       parent_sel    = create_parent_selection("tournament", amount=50, tournament_size=3, random_state=rng),
       survivor_sel  = create_survivor_selection("elitism", amount=25, random_state=rng),
       mutation_prob = 0.3,
       crossover_prob= 0.9,
       random_state  = rng,
   )

   # Configure runtime objects
   stopping = StoppingCondition(
       condition_str    = "max_iterations or objective_target",
       max_iterations  = 100,
       objective_target = 1e-5,
       optimization_mode= objfunc.mode,
   )

   reporter = TQDMReporter()

   history = HistoryTracker(track_median=True, track_parameters=True)

   checkpointer = Checkpointer("ga_sphere.pkl", iteration_frequency=20)

   # Create and run
   algo = Algorithm(
       objfunc,
       strategy,
       stopping_condition = stopping,
       reporter           = reporter,
       history_tracker    = history,
       checkpointer       = checkpointer,
   )

   population = algo.optimize()
   solution, obj = population.best_solution()
   print(f"Best objective: {obj:.6g}")
   print(algo.history_tracker.to_pandas().tail())

After execution, ``population`` holds the final generation, and you can use
``algo.history_tracker`` to produce convergence plots (see the plotting tutorial).


Parallelism Note
----------------

The ``parallel`` and ``threads`` parameters of ``Algorithm`` are currently
placeholders and have no effect.  They are accepted for future compatibility but
should not be relied upon.