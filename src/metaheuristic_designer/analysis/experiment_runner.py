from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..objective_function import ObjectiveFunc

from ..benchmarks.ioh_wrapper import IOHObjective

# IOH is optional – only imported if needed (the actual import occurs inside
# the loop if a problem has an IOH logger; the module-level import is safe
# because we only reference ioh when has_logger is True).
try:
    import ioh
except ImportError:
    ioh = None


def run_experiment(
    problems: Iterable[ObjectiveFunc],
    algorithms: Iterable[Callable],
    max_evals: int,
    n_runs: int = 30,
    base_seed: int = 42,
    output_root: str = "experiment_data",
):
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

    seed_sequence = np.random.SeedSequence(base_seed)
    for p_idx, problem in enumerate(problems):
        for r in range(n_runs):
            subsequence = seed_sequence.spawn(key=(p_idx, r))[0]
            seed = subsequence.generate_state(1, dtype=np.uint32)[0]

            # Reset the problem to its initial state (counter + IOH transforms)
            problem.restart()

            for algo_name, algo_factory in algorithms.items():
                # ---- IOH logger (only for IOH-backed problems) ---------
                if isinstance(problem, IOHObjective):
                    logger = ioh.logger.Analyzer(
                        root=output_root,
                        folder_name=f"{algo_name}/{problem.name}/run_{r}",
                        algorithm_name=algo_name,
                        algorithm_info="",
                        store_positions=False,
                    )
                    problem.attach_logger(logger)

                # ---- Build solver and optimize -------------------------
                solver = algo_factory(problem, seed, max_evals)
                solver.optimize()
                _, best_obj = solver.best_solution()

                # ---- Detach logger (writes .dat / .info files) ---------
                if isinstance(problem, IOHObjective):
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
