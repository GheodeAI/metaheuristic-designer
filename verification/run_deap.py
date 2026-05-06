"""
Full-scale DEAP benchmark on BBOB (IOH).

Before running:
    pip install ioh tqdm pandas numpy deap

Customise the constants below if you want fewer problems or runs.
"""

import pandas as pd
from tqdm import tqdm

from metaheuristic_designer.benchmarks.ioh_wrapper import IOHObjective
from metaheuristic_designer.analysis import run_experiment
from deap_helpers import (
    canonical_deap_ga,
    canonical_deap_es,
)

# -------------------------------------------------------------------
# 1. Problem suite – BBOB, all 24 functions
# -------------------------------------------------------------------
BBOB_FUNCTIONS = list(range(1, 25))          # 1..24
DIMENSIONS     = [5, 10, 20]                 # standard paper settings
INSTANCES      = list(range(1, 6))           # 5 independent instances

MAX_EVALS_MUL  = 10_000                      # typical: 10_000 * dimension
N_RUNS         = 30                           # independent runs per (prob, algo)
BASE_SEED      = 42
OUTPUT_ROOT    = "experiment_data_stress"
CSV_OUTPUT     = "deap_stress_results.csv"
CHECKPOINT_EVERY = 10                        # save after every 10 algorithm runs

# -------------------------------------------------------------------
# 2. Build the problem list
# -------------------------------------------------------------------
problems = []
for fid in BBOB_FUNCTIONS:
    for dim in DIMENSIONS:
        for ins in INSTANCES:
            problems.append(IOHObjective(fid, dim, instance=ins))

print(f"Total problems: {len(problems)}")
print(f"Estimated runs: {len(problems) * N_RUNS * 3} (3 algorithms)")

# -------------------------------------------------------------------
# 3. Algorithm factories
# -------------------------------------------------------------------
algorithms = {
    "GA (DEAP)": canonical_deap_ga,
    "ES (DEAP)": canonical_deap_es,
}

# -------------------------------------------------------------------
# 4. Run the experiment (with checkpointing and progress bar)
# -------------------------------------------------------------------
# We wrap run_experiment with manual loop over problems to add progress and saving.
# run_experiment already loops problems and runs internally, but checkpointing is easier
# if we write a small helper.

# Simplified: we use the existing run_experiment function but since it returns a single
# DataFrame, we'll run it in batches of problems to avoid memory issues.

all_dfs = []
n_problems = len(problems)
pbar = tqdm(total=n_problems, desc="Problems")
for idx, problem in enumerate(problems):
    pbar.set_postfix_str(f"Problem {idx+1}/{n_problems}: {problem.name}")
    df_chunk = run_experiment(
        problems=[problem],
        algorithms=algorithms,
        max_evals=MAX_EVALS_MUL * problem.dimension,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        output_root=OUTPUT_ROOT,
    )
    all_dfs.append(df_chunk)
    pbar.update(1)
    # Save checkpoint every N problems
    if (idx + 1) % CHECKPOINT_EVERY == 0:
        tmp = pd.concat(all_dfs, ignore_index=True)
        tmp.to_csv(f"{CSV_OUTPUT}.checkpoint_{idx+1}", index=False)
        print(f"Checkpoint saved at problem {idx+1}")

pbar.close()

# Combine and save final
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv(CSV_OUTPUT, index=False)
print(f"Final results saved to {CSV_OUTPUT}")

print("\nSummary (mean best objective per algorithm/dimension):")
summary = final_df.groupby(["algorithm", "dimension"])["best_objective"].mean().unstack()
print(summary)