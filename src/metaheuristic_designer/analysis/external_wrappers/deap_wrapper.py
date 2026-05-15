import random
import numpy as np
import pandas as pd

try:
    from deap import algorithms, tools
    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False

class DEAPWrapper:
    """
    Duck-typed wrapper around a DEAP evolutionary algorithm.

    The user supplies a ``build_fn`` that sets up the toolbox, population,
    statistics, and hall-of-fame. The wrapper runs the DEAP algorithm
    (by default ``eaSimple``) and exposes a minimal interface for the
    experiment loop.

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Must provide ``objective(x) -> float``, ``mode``, ``dimension``.
    build_fn : callable(objfunc, seed) -> (toolbox, pop, stats, hof)
        Receives the objective function and an integer seed.
        It must return a fully configured DEAP ``Toolbox``, an initial
        population, a ``Statistics`` object (with at least a key like
        ``"max"`` for the best fitness), and a ``HallOfFame``.
    ngen : int
        Maximum number of generations.
    seed : int or None
        Reproducibility seed.
    name : str
        Display name.
    algorithm : callable, optional
        DEAP algorithm function (default ``algorithms.eaSimple``).
        It will be called as
        ``algorithm(pop, toolbox, ngen, stats, hof, **algo_kwargs)``.
    **algo_kwargs
        Additional keyword arguments passed to the DEAP algorithm
        (e.g., cxpb, mutpb).
    """

    def __init__(self, objfunc, build_fn, ngen=100, seed=None,
                 name="DEAP", algorithm=None, **algo_kwargs):
        if not _DEAP_AVAILABLE:
            raise ImportError(
                "The 'deap' library is required. Install with `pip install deap`."
            )
        self.objfunc = objfunc
        self.build_fn = build_fn
        self.ngen = ngen
        self.seed = seed
        self.name = name
        self.algorithm = algorithm if algorithm is not None else algorithms.eaSimple
        self.algo_kwargs = algo_kwargs

        self._history_df = None
        self._best_ind = None
        self._best_obj = None

    def optimize(self):
        """Run the DEAP algorithm and store results."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        toolbox, pop, stats, hof = self.build_fn(self.objfunc, self.seed)

        # Run the DEAP algorithm
        _, logbook = self.algorithm(
            pop, toolbox, ngen=self.ngen,
            stats=stats, halloffame=hof, verbose=False,
            **self.algo_kwargs
        )

        self._history_df = self._logbook_to_df(logbook, self.objfunc.mode)

        if len(hof) > 0:
            self._best_ind = hof[0]
            fitness = self._best_ind.fitness.values[0]
            # DEAP maximises fitness internally; convert back to raw objective
            raw = -fitness if self.objfunc.mode == "min" else fitness
            self._best_obj = raw

        return self

    def best_solution(self):
        """Return the best solution and its raw objective value."""
        if self._best_ind is None:
            return None, None
        return list(self._best_ind), self._best_obj

    @property
    def history_tracker(self):
        """Return an object with a to_pandas() method."""
        class _Hist:
            def __init__(self, df):
                self._df = df
            def to_pandas(self):
                return self._df.copy()
        return _Hist(self._history_df)

    @staticmethod
    def _logbook_to_df(logbook, mode):
        """Convert DEAP logbook to a DataFrame with 'iteration' and 'best_objective'."""
        rows = []
        for record in logbook:
            gen = record.get("gen", 0)
            deap_best = record.get("max", record.get("avg"))
            if deap_best is None:
                continue
            raw_obj = -deap_best if mode == "min" else deap_best
            rows.append({"iteration": gen, "best_objective": raw_obj})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from deap import base, creator
    from metaheuristic_designer.benchmarks import Sphere

    def build_ga(objfunc, seed):
        """Example builder: simple GA with blend crossover and Gaussian mutation."""
        random.seed(seed)
        np.random.seed(seed)

        DIM = objfunc.dimension

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -5, 5)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=DIM)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            return (objfunc.objective(ind),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)

        return toolbox, pop, stats, hof

    objfunc = Sphere(dim=5, mode="min")
    solver = DEAPWrapper(
        objfunc,
        build_fn=build_ga,
        ngen=100,
        seed=42,
        cxpb=0.7,
        mutpb=0.3,
        algorithm=algorithms.eaSimple,
    )
    solver.optimize()
    best_x, best_obj = solver.best_solution()
    print(f"Best objective: {best_obj}")