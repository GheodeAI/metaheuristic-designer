import numpy as np
import pandas as pd

try:
    import pygmo as pg
    _PYGMO_AVAILABLE = True
except ImportError:
    _PYGMO_AVAILABLE = False

class PyGMOWrapper:
    """
    Duck-typed wrapper around a PyGMO algorithm.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Must provide ``objective(x) -> float``, ``dimension``,
        ``lower_bound``, ``upper_bound``, and ``mode``.
    algorithm : str or pygmo.algorithm
        PyGMO algorithm name (e.g. "de", "sade", "pso", "cmaes") or
        an already-constructed algorithm object.
    pop_size : int
        Population size.
    generations : int
        Number of generations to evolve.
    seed : int or None
        Random seed passed to PyGMO.
    name : str
        Display name.
    **algo_kwargs
        Extra arguments forwarded to the PyGMO algorithm constructor.
    """

    def __init__(self, objfunc, algorithm="de", pop_size=50, generations=100,
                 seed=None, name="PyGMO", **algo_kwargs):
        if not _PYGMO_AVAILABLE:
            raise ImportError(
                "The 'pygmo' library is required. Install with `pip install pygmo`."
            )
        self.objfunc = objfunc
        self.algorithm = algorithm if isinstance(algorithm, str) else algorithm
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.name = name
        self.algo_kwargs = algo_kwargs

        self._history_df = None
        self._best_x = None
        self._best_obj = None

    def optimize(self):
        # Minimal PyGMO problem: just wraps the objective
        class PyGMOProblem:
            def __init__(self, objfunc):
                self.objfunc = objfunc
                self.dim = objfunc.dimension
                self.lb = np.broadcast_to(objfunc.lower_bound, self.dim)
                self.ub = np.broadcast_to(objfunc.upper_bound, self.dim)

            def fitness(self, x):
                return [self.objfunc.objective(np.array(x))]

            def get_bounds(self):
                return (self.lb, self.ub)

            def get_name(self):
                return "WrappedProblem"

        prob = pg.problem(PyGMOProblem(self.objfunc))

        if self.seed is not None:
            pg.set_global_rng_seed(self.seed)

        if isinstance(self.algorithm, str):
            algo = pg.algorithm(getattr(pg, self.algorithm)(**self.algo_kwargs))
        else:
            algo = self.algorithm
        algo.set_verbosity(0)

        pop = pg.population(prob, size=self.pop_size, seed=self.seed)

        best_obj = float("inf") if self.objfunc.mode == "min" else float("-inf")
        trace = []

        for gen in range(self.generations):
            pop = algo.evolve(pop)
            champ_fit = pop.champion_f[0]
            if self.objfunc.mode == "min" and champ_fit < best_obj:
                best_obj = champ_fit
                self._best_x = pop.champion_x
            elif self.objfunc.mode == "max" and champ_fit > best_obj:
                best_obj = champ_fit
                self._best_x = pop.champion_x

            trace.append({"iteration": gen, "best_objective": best_obj})

        self._best_obj = best_obj
        self._history_df = pd.DataFrame(trace)
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