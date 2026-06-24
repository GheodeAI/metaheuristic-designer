import numpy as np
import pandas as pd

try:
    import cma

    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


class CMAWrapper:
    """
    Duck-typed wrapper for the reference cma library.
    """

    def __init__(self, objfunc, sigma0=0.5, max_iterations=500, seed=None, name="cma"):
        if not _CMA_AVAILABLE:
            raise ImportError("The 'cma' library is required. Install with `pip install cma`.")
        self.objfunc = objfunc
        self.sigma0 = sigma0
        self.max_iterations = max_iterations
        self.seed = seed
        self.name = name

        self.history_df = None
        self.best_x = None
        self.best_obj = None

    def optimize(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x0 = np.random.uniform(self.objfunc.lower_bound, self.objfunc.upper_bound, size=self.objfunc.dimension)

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {"maxiter": self.max_iterations, "seed": self.seed})

        while not es.stop():
            solutions = es.ask()
            objs = [self.objfunc.objective(np.array(sol)) for sol in solutions]
            es.tell(solutions, objs)

        # CMA‑ES stores the best per generation in es.result
        if hasattr(es.result, "historical_fbest") and es.result.historical_fbest:
            fbest = es.result.historical_fbest
            self.history_df = pd.DataFrame({"iteration": np.arange(len(fbest)), "best_objective": fbest})
        else:
            self.history_df = pd.DataFrame(columns=["iteration", "best_objective"])

        self.best_x = es.result.xfavorite
        self.best_obj = es.result.fbest

        return self

    def best_solution(self):
        return list(self.best_x), self.best_obj

    def to_pandas(self):
        return self.history_df
