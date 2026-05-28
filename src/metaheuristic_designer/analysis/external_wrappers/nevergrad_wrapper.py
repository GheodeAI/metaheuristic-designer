import numpy as np
import pandas as pd

try:
    import nevergrad as ng

    _NEVERGRAD_AVAILABLE = True
except ImportError:
    _NEVERGRAD_AVAILABLE = False


class NevergradWrapper:
    """
    Duck-typed wrapper around a Nevergrad optimizer.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Must provide ``objective(x) -> float``, ``dimension``,
        ``lower_bound``, ``upper_bound``, and ``mode`` ("min"/"max").
    optimizer_name : str
        Nevergrad optimizer name, e.g. "NGOpt", "DE", "CMA", "TwoPointsDE".
    budget : int
        Total number of function evaluations.
    seed : int or None
        Reproducibility seed.
    name : str
        Display name.
    **opt_kwargs : dict
        Additional arguments passed to the Nevergrad optimizer constructor.
    """

    def __init__(self, objfunc, optimizer_name="NGOpt", budget=1000, seed=None, name="Nevergrad", **opt_kwargs):
        if not _NEVERGRAD_AVAILABLE:
            raise ImportError("The 'nevergrad' library is required. Install with `pip install nevergrad`.")
        self.objfunc = objfunc
        self.optimizer_name = optimizer_name
        self.budget = budget
        self.seed = seed
        self.name = name
        self.opt_kwargs = opt_kwargs

        self.history_df = None
        self.best_x = None
        self.best_obj = None

    def optimize(self):
        dim = self.objfunc.dimension
        param = ng.p.Array(
            shape=(dim,),
            lower=self.objfunc.lower_bound,
            upper=self.objfunc.upper_bound,
        )

        # Seed via random_state (newer Nevergrad) or global numpy seed
        try:
            opt = ng.optimizers.registry[self.optimizer_name](parametrization=param, budget=self.budget, random_state=self.seed, **self.opt_kwargs)
        except TypeError:
            # Fallback for older Nevergrad versions
            if self.seed is not None:
                np.random.seed(self.seed)
            opt = ng.optimizers.registry[self.optimizer_name](parametrization=param, budget=self.budget, **self.opt_kwargs)

        best_so_far = float("inf") if self.objfunc.mode == "min" else float("-inf")
        evals = 0
        trace = []

        while evals < self.budget:
            candidates = opt.ask()
            if not isinstance(candidates, list):
                candidates = [candidates]

            losses = []
            for cand in candidates:
                x = cand.value
                obj = self.objfunc.objective(x)
                losses.append(obj)
                evals += 1
                if self.objfunc.mode == "min":
                    if obj < best_so_far:
                        best_so_far = obj
                        self.best_x = x
                else:
                    if obj > best_so_far:
                        best_so_far = obj
                        self.best_x = x

            for cand, loss in zip(candidates, losses):
                opt.tell(cand, loss)

            trace.append({"iteration": evals, "best_objective": best_so_far})

            if evals >= self.budget:
                break

        self.best_obj = best_so_far
        self.history_df = pd.DataFrame(trace)
        return self

    def best_solution(self):
        return (list(self.best_x) if self.best_x is not None else None, self.best_obj)

    def to_pandas(self):
        return self.history_df
