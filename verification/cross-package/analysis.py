# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: mhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Statistical analysis of DEAP benchmark results
#
# This notebook loads the `deap_stress_results.csv` produced by
# `stress_test_deap.py` and performs the standard statistical tests
# for the paper: mean ranks, Friedman test, pairwise Wilcoxon with
# Holm correction, and a critical difference diagram.
#
# **Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`,
# `autorank` (``pip install autorank``).

# %% [markdown]
# ## 1.  Load the data

# %%
import pandas as pd
import numpy as np

# Adjust path if needed – this is the final CSV from the stress test
df = pd.read_csv("deap_stress_results.csv")

print(f"Loaded {len(df)} runs")
print(f"Columns: {list(df.columns)}")
df.head()

# %% [markdown]
# ## 2.  Quick sanity checks

# %%
# Check that every algorithm/problem/run combination is present
grouped = df.groupby(["algorithm", "problem_name"]).size().unstack(fill_value=0)
assert grouped.min().min() == 30, "Missing runs detected!"
print("All 30 runs present for every combination.")
grouped.head()

# %% [markdown]
# ## 3.  Mean ranks and average performance

# %%
# Compute mean best objective per algorithm/problem
mean_per_alg_prob = df.groupby(["algorithm", "problem_name"])["best_objective"].mean().unstack()

# Rank algorithms per problem (1 = best, lower is better)
# (If objective is negative for maximisation problems, flip sign – our BBOB are min.)
ranks_per_problem = mean_per_alg_prob.rank(axis=0, ascending=True)

# Average rank
avg_ranks = ranks_per_problem.mean(axis=1).sort_values()
print("Average ranks (1 = best):")
print(avg_ranks)

# %% [markdown]
# ## 4.  Visualise the performance distribution

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Boxplot of best objectives across all problems
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="algorithm", y="best_objective", palette="Set2", ax=ax)
ax.set_ylabel("Best objective (minimisation)")
ax.set_title("Distribution of best objective across all problems and runs")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5.  Statistical tests with autorank

# %%
from autorank import autorank, plot_stats

# autorank expects a DataFrame with one column per algorithm.
# We need the best objective per algorithm, paired by (problem, run).
pivot = df.pivot_table(
    index=["problem_name", "run"], columns="algorithm", values="best_objective"
)
pivot = pivot.dropna()  # just in case

result = autorank(pivot, alpha=0.05, verbose=True, order="ascending")
print(result)

# %%
# Critical difference diagram
fig = plot_stats(result, allow_insignificant=False)
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.  Summary table for the paper

# %%
# Table of mean ± std per algorithm
summary = df.groupby("algorithm")["best_objective"].agg(["mean", "std", "min", "max"])
summary["rank"] = avg_ranks
summary = summary.sort_values("rank")
print(summary)

# %% [markdown]
# ## 7.  Next steps
#
# - Copy the resulting CD diagram into your paper.
# - Open `experiment_data_stress/` with **IOHanalyzer** for
#   ECDF curves, fixed‑target analysis, and performance profiles.
# - This notebook can be re‑run with more algorithms once the
#   other wrappers (PyGMO, Nevergrad, SciPy) are added to the
#   experiment loop.