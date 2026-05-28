import random
import numpy as np
import pandas as pd

try:
    from deap import algorithms

    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False


class DEAPWrapper:
    """
    Minimal wrapper that runs a DEAP algorithm and records the best result
    at each generation. You give it everything already set up (toolbox,
    population, stats, hall-of-fame) and it runs the show.
    """

    def __init__(self, objfunc, toolbox, pop, stats, hof, ngen=100, seed=None, algorithm=algorithms.eaSimple, **algo_kwargs):
        if not _DEAP_AVAILABLE:
            raise ImportError("deap is not installed")

        self.objfunc = objfunc
        self.toolbox = toolbox
        self.pop = pop
        self.stats = stats
        self.hof = hof
        self.ngen = ngen
        self.seed = seed
        self.algorithm = algorithm
        self.algo_kwargs = algo_kwargs

        # These are filled after optimize()
        self.history_df = None
        self.best_ind = None
        self.best_obj = None

    def optimize(self):
        """Run the algorithm and store the results."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Run the DEAP algorithm
        _, logbook = self.algorithm(self.pop, self.toolbox, ngen=self.ngen, stats=self.stats, halloffame=self.hof, verbose=False, **self.algo_kwargs)

        # Build a simple history of best objective per generation
        records = []
        for entry in logbook:
            gen = entry.get("gen", 0)
            deap_best = entry.get("max", None)  # DEAP tracks max of fitness
            if deap_best is None:
                continue
            # Convert back to raw objective (minimization => fitness = -obj)
            raw_obj = -deap_best if self.objfunc.mode == "min" else deap_best
            records.append({"iteration": gen, "best_objective": raw_obj})
        self.history_df = pd.DataFrame(records)

        # Get the single best individual from the hall of fame
        if len(self.hof) > 0:
            self.best_ind = self.hof[0]
            fitness_val = self.best_ind.fitness.values[0]
            raw = -fitness_val if self.objfunc.mode == "min" else fitness_val
            self.best_obj = raw

        return self

    def best_solution(self):
        """Return (solution_list, best_raw_objective)."""
        if self.best_ind is None:
            return None, None
        return list(self.best_ind), self.best_obj

    def to_pandas(self):
        return self.history_df
