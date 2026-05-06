import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, basinhopping, dual_annealing


class SciPyWrapper:
    """
    Duck-typed wrapper around SciPy optimisation algorithms.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Must provide ``objective(x) -> float``, ``dimension``,
        ``lower_bound``, ``upper_bound``, and ``mode``.
    method : {"differential_evolution", "basinhopping", "dual_annealing"}
        Which SciPy optimisation routine to use.
    maxiter : int
        Maximum number of iterations (for DE and dual annealing) or
        ``niter`` for basinhopping.
    seed : int or None
        Seed for reproducibility.
    name : str
        Display name.
    **opt_kwargs
        Extra keyword arguments forwarded to the SciPy routine.
    """

    def __init__(self, objfunc, method="differential_evolution", maxiter=1000,
                 seed=None, name="SciPy", **opt_kwargs):
        self.objfunc = objfunc
        self.method = method
        self.maxiter = maxiter
        self.seed = seed
        self.name = name
        self.opt_kwargs = opt_kwargs

        self._history_df = None
        self._best_x = None
        self._best_obj = None

    def optimize(self):
        dim = self.objfunc.dimension
        bounds = [(self.objfunc.lower_bound, self.objfunc.upper_bound)
                  for _ in range(dim)]

        best = float("inf") if self.objfunc.mode == "min" else float("-inf")
        records = []

        def callback(xk, convergence=None):
            nonlocal best
            val = self.objfunc.objective(np.array(xk))
            if self.objfunc.mode == "min":
                if val < best:
                    best = val
                    self._best_x = xk
            else:
                if val > best:
                    best = val
                    self._best_x = xk
            records.append({"iteration": len(records), "best_objective": best})

        if self.method == "differential_evolution":
            res = differential_evolution(
                self.objfunc.objective,
                bounds,
                maxiter=self.maxiter,
                seed=self.seed,
                callback=callback,
                **self.opt_kwargs
            )
        elif self.method == "basinhopping":
            rng = np.random.default_rng(self.seed)
            x0 = rng.uniform(self.objfunc.lower_bound, self.objfunc.upper_bound, size=dim)
            kwargs = dict(self.opt_kwargs)
            if "niter" not in kwargs:
                kwargs["niter"] = self.maxiter
            res = basinhopping(
                self.objfunc.objective,
                x0,
                seed=self.seed,
                callback=callback,
                **kwargs
            )
        elif self.method == "dual_annealing":
            res = dual_annealing(
                self.objfunc.objective,
                bounds,
                maxiter=self.maxiter,
                seed=self.seed,
                callback=callback,
                **self.opt_kwargs
            )
        else:
            raise ValueError(f"Unknown SciPy method: {self.method}")

        self._best_obj = res.fun
        self._history_df = pd.DataFrame(records)
        return self

    def best_solution(self):
        return (list(self._best_x) if self._best_x is not None else None,
                self._best_obj)

    @property
    def history_tracker(self):
        class _Hist:
            def __init__(self, df):
                self._df = df
            def to_pandas(self):
                return self._df.copy()
        return _Hist(self._history_df)