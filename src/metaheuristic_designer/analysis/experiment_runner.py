import numpy as np
import pandas as pd

# IOH is optional – only imported if needed (the actual import occurs inside
# the loop if a problem has an IOH logger; the module-level import is safe
# because we only reference ioh when has_logger is True).
try:
    import ioh
except ImportError:
    ioh = None


def run_experiment(problems, algorithms, max_evals, n_runs=30, base_seed=42,
                   output_root="experiment_data"):
    """
    Run a fair, reproducible comparison of algorithms on a set of problems.

    Parameters
    ----------
    problems : list of ObjectiveFunc
        Benchmark functions to solve.  IOHObjective instances will produce
        IOH-compatible log files automatically.
    algorithms : dict of str -> callable
        Keys are algorithm names; values are factories
        ``(objfunc, seed, budget) -> solver``.
    max_evals : int
        Common evaluation budget per run.
    n_runs : int
        Number of independent repetitions per (problem, algorithm) pair.
    base_seed : int
        Master seed for reproducibility.
    output_root : str
        Folder where IOH log files are written.  Non-IOH problems are
        silently ignored.

    Returns
    -------
    pd.DataFrame
        Columns: ``algorithm``, ``problem_name``, ``fid``, ``dimension``,
        ``instance``, ``run``, ``best_objective``.
    """
    records = []
    n_problems = len(problems)

    for p_idx, problem in enumerate(problems):
        # Detect whether this problem can be logged by IOH
        has_logger = hasattr(problem, "attach_logger")

        for r in range(n_runs):
            # Deterministic, non‑colliding seed for this (problem, run) pair.
            # Three primes ensure no two (p_idx, r) map to the same seed
            # for any realistic experiment size.
            seed = base_seed * 1103515245 + p_idx * 12345 + r * 1103515249

            # Reset the problem to its initial state (counter + IOH transforms)
            problem.restart()

            for algo_name, algo_factory in algorithms.items():
                # ---- IOH logger (only for IOH-backed problems) ---------
                if has_logger:
                    logger = ioh.logger.Analyzer(
                        root=output_root,
                        folder_name=f"{algo_name}/{problem.name}/run_{r}",
                        algorithm_name=algo_name,
                        algorithm_info="",
                        store_positions=False,
                    )
                    problem.attach_logger(logger)

                # ---- Build solver and optimise -------------------------
                solver = algo_factory(problem, seed, max_evals)
                solver.optimize()
                _, best_obj = solver.best_solution()

                # ---- Detach logger (writes .dat / .info files) ---------
                if has_logger:
                    problem.detach_logger()

                # ---- Record result -------------------------------------
                records.append(
                    {
                        "algorithm": algo_name,
                        "problem_name": problem.name,
                        "fid": getattr(problem, "_fid", None),
                        "dimension": problem.dimension,
                        "instance": getattr(problem, "_instance", None),
                        "run": r,
                        "best_objective": best_obj,
                    }
                )

    return pd.DataFrame(records)