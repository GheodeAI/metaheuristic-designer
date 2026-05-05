.. _plotting-tutorial:

Visualizing Optimization Runs
==============================

This library focuses on the optimisation logic itself and does not ship with built-in
plotting routines.  That said, the data recorded by the
:class:`~metaheuristic_designer.history_tracker.HistoryTracker` works seamlessly with
standard data-science tools, so you can create any visual you need with just a few
lines of code.

The examples on this page use `seaborn <https://seaborn.pydata.org>`_,
`matplotlib <https://matplotlib.org>`_ and `plotly <https://plotly.com/python/>`_,
available as optional dependencies by installing ``metaheuristic_designer[examples]``—
but you are free to choose whatever plotting library you prefer.

.. code-block:: python

   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.set_theme(style="whitegrid")

All examples assume ``algo`` is a completed :class:`~metaheuristic_designer.Algorithm` run
and ``df = algo.history_tracker.to_pandas()``.

Convergence of the Best Objective
---------------------------------

The simplest diagnostic: how does the best solution improve over time?

.. code-block:: python

   fig, ax = plt.subplots(figsize=(8, 5))
   sns.lineplot(data=df, x="iteration", y="best_objective", linewidth=2, ax=ax)
   ax.set_xlabel("Generation")
   ax.set_ylabel("Objective")
   ax.set_title("Convergence Plot")
   plt.show()

For problems where the objective spans orders of magnitude, use a logarithmic y-axis:

.. code-block:: python

   ax.set_yscale("log")

Best, Median, and Worst Objectives
-----------------------------------

If you enabled ``track_median`` and ``track_worst``, you can visualise the spread of
the population.

.. code-block:: python

   fig, ax = plt.subplots(figsize=(8, 5))
   sns.lineplot(data=df, x="iteration", y="best_objective", label="Best", linewidth=2, ax=ax)
   sns.lineplot(data=df, x="iteration", y="median_objective", label="Median", linewidth=1.5, ax=ax)
   sns.lineplot(data=df, x="iteration", y="worst_objective", label="Worst", linewidth=1, ax=ax)
   ax.set_xlabel("Generation")
   ax.set_ylabel("Objective")
   ax.set_title("Best, Median, and Worst")
   ax.legend()
   plt.show()

To highlight the gap between best and worst, shade it:

.. code-block:: python

   ax.fill_between(df["iteration"], df["worst_objective"], df["best_objective"],
                   alpha=0.1, color="steelblue", label="Range")
   ax.legend()

Comparing Multiple Algorithms
-----------------------------

Combine the DataFrames of several algorithms and use ``hue`` to differentiate them.

.. code-block:: python

   ga_df["algorithm"] = "GA"
   de_df["algorithm"] = "DE"
   pso_df["algorithm"] = "PSO"
   combined = pd.concat([ga_df, de_df, pso_df], ignore_index=True)

   fig, ax = plt.subplots(figsize=(8, 5))
   sns.lineplot(data=combined, x="iteration", y="best_objective",
                hue="algorithm", linewidth=2, ax=ax)
   ax.set_xlabel("Generation")
   ax.set_ylabel("Best objective")
   ax.set_title("Algorithm Comparison")
   plt.show()

Fitness Distribution Over Generations
--------------------------------------

When ``track_full_objective`` is enabled, the tracker stores the entire vector of raw
objectives at every generation.  The method :meth:`~metaheuristic_designer.history_tracker.HistoryTracker.to_pandas_full_objective`
returns a wide-format DataFrame where each column corresponds to one individual
(``Individual_0``, ``Individual_1``, …).  To create a boxplot you first melt this
table into long format:

.. code-block:: python

   wide_df = algo.history_tracker.to_pandas_full_objective()

   # Melt to long format for seaborn
   long_df = wide_df.melt(id_vars="iteration",
                          var_name="individual",
                          value_name="objective")

   fig, ax = plt.subplots(figsize=(12, 5))
   # Plot every 5th generation to keep the plot readable
   plot_data = long_df[long_df["iteration"] % 5 == 0]
   sns.boxplot(data=plot_data, x="iteration", y="objective", ax=ax,
               palette="viridis", width=0.6)
   ax.set_xlabel("Generation")
   ax.set_ylabel("Objective")
   ax.set_title("Fitness Distribution")
   plt.show()

This reveals whether the population is converging tightly or still exploring.

Diversity Over Generations
---------------------------

If you enabled ``track_diversity``, the DataFrame contains a ``diversity`` column.
You can plot it alongside the best objective to see how exploration evolves.
A dual-axis plot is often the clearest:

.. code-block:: python

   fig, ax1 = plt.subplots(figsize=(8, 5))
   ax2 = ax1.twinx()

   sns.lineplot(data=df, x="iteration", y="best_objective", ax=ax1,
                linewidth=2, color="tab:blue", label="Objective")
   sns.lineplot(data=df, x="iteration", y="diversity", ax=ax2,
                linewidth=2, color="tab:red", label="Diversity")

   ax1.set_xlabel("Generation")
   ax1.set_ylabel("Objective", color="tab:blue")
   ax2.set_ylabel("Diversity", color="tab:red")
   ax1.set_title("Convergence and Diversity")

   # Combine legends
   lines1, labels1 = ax1.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
   plt.show()

Evolution of Schedulable Parameters
------------------------------------

When you use parameter schedules (see :doc:`parameter_schedules`), the tracker
automatically records the current value of each scheduled parameter at every generation
if ``track_parameters`` is enabled.  The parameter names are built from the component
hierarchy, e.g. ``"mutation.gaussian_mutation.F"`` for the mutation strength
or ``"BranchOperator.p"`` for a branch probability.  These columns appear directly
in the DataFrame returned by ``to_pandas()``.

Plot them alongside convergence to understand how the search adapts over time:

.. code-block:: python

   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

   # Top panel: best objective
   sns.lineplot(data=df, x="iteration", y="best_objective", ax=ax1,
                linewidth=2, color="tab:green")
   ax1.set_ylabel("Objective")
   ax1.set_title("Convergence and Parameter Evolution")

   # Bottom panel: scheduled parameters
   param_cols = ["mutation.gaussian_mutation.F", "BranchOperator.p"]
   for col in param_cols:
       sns.lineplot(data=df, x="iteration", y=col, ax=ax2, label=col)
   ax2.set_ylabel("Parameter value")
   ax2.legend()
   ax2.set_xlabel("Generation")
   plt.tight_layout()
   plt.show()

If you have many parameters, consider using separate subplots or a dual-axis plot to
avoid clutter.

Combining Multiple Metrics in a Dashboard
-----------------------------------------

Put several views into one figure to get a comprehensive picture:

.. code-block:: python

   fig = plt.figure(figsize=(14, 10))

   # Convergence
   ax1 = fig.add_subplot(2, 2, 1)
   sns.lineplot(data=df, x="iteration", y="best_objective", ax=ax1)
   ax1.set_title("Best Objective")

   # Fitness distribution (every 10th generation)
   ax2 = fig.add_subplot(2, 2, 2)
   wide_df = algo.history_tracker.to_pandas_full_objective()
   long_df = wide_df.melt(id_vars="iteration", var_name="individual", value_name="objective")
   plot_data = long_df[long_df["iteration"] % 10 == 0]
   sns.boxplot(data=plot_data, x="iteration", y="objective", ax=ax2, width=0.6)
   ax2.set_title("Fitness Distribution")

   # Parameter 1
   ax3 = fig.add_subplot(2, 2, 3)
   sns.lineplot(data=df, x="iteration", y="mutation.gaussian_mutation.F",
                ax=ax3, color="tab:orange")
   ax3.set_title("Mutation Strength")

   # Parameter 2
   ax4 = fig.add_subplot(2, 2, 4)
   sns.lineplot(data=df, x="iteration", y="BranchOperator.p",
                ax=ax4, color="tab:red")
   ax4.set_title("Branch Probability")

   plt.tight_layout()
   plt.show()

Real-Time Visualization
-----------------------

For interactive exploration, you can step the algorithm manually and update a plot
live.  The core logic is independent of the plotting library; here is a sketch using
Plotly's ``FigureWidget``:

.. code-block:: python

   import plotly.graph_objects as go

   fig = go.FigureWidget()
   fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Best"))

   algo.initialize()
   for gen in range(max_generations):
       algo.step()
       algo.history_tracker.step(algo)
       df = algo.history_tracker.to_pandas()
       fig.data[0].x = df["iteration"]
       fig.data[0].y = df["best_objective"]

This approach keeps the entire history; you can also stream only the latest data.

Customising and Exporting
-------------------------

All plots are standard matplotlib/seaborn objects.  You can:

- Add a horizontal target line: ``ax.axhline(y=target, color="red", linestyle="--")``
- Save to file: ``plt.savefig("convergence.pdf", dpi=300)``
- Annotate the best solution: ``ax.annotate(...)``
- Use the DataFrame for further analysis in Jupyter notebooks.