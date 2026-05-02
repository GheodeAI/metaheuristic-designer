from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING
from ..reporter import Reporter

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm

logger = logging.getLogger(__name__)

class VerboseReporter(Reporter):
    def __init__(self, verbose_timer=0.5, **kwargs):
        self.verbose_timer = verbose_timer
        self.verbose_start = time.time()

    def log_init(self, algorithm: Algorithm):
        self.verbose_start = time.time()

        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name

        print(f"Initializing optimization of {objfunc_name} using {alg_name}")
        print(f"-----------------------------{'-'*len(objfunc_name)}-------{'-'*len(alg_name)}")
        print()

    def log_step(self, algorithm: Algorithm):
        if time.time() - self.verbose_start < self.verbose_timer:
            return
        self.verbose_start = time.time()

        logger.debug("Finished iteration %d.", algorithm.iterations)
        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name
        spent_time = algorithm.stopping_condition.real_time_spent
        spent_cpu_time = algorithm.stopping_condition.cpu_time_spent
        iterations = algorithm.iterations
        evaluations = algorithm.stopping_condition.evaluations

        _, best_fitness = algorithm.best_solution()
        print(f"Optimizing {objfunc_name} using {alg_name}:")
        print(f"\tReal time Spent: {spent_time:.4f}s")
        print(f"\tCPU time Spent:  {spent_cpu_time:.4f}s")
        print(f"\tGeneration: {iterations}")
        print(f"\tBest fitness: {best_fitness}")
        print(f"\tEvaluations of fitness: {evaluations}\n")
        algorithm.search_strategy.extra_step_info()
        print()

    def log_end(self, algorithm: Algorithm):
        """
        Shows a summary of the execution of the algorithm.

        Parameters
        ----------
        show_plots: bool, optional
            Whether to display plots about the algorithm or not.
        """

        objfunc_name = algorithm.objfunc.name
        alg_name = algorithm.name
        iterations_accurate = algorithm.history_tracker.iterations
        spent_time = algorithm.stopping_condition.real_time_spent
        spent_cpu_time = algorithm.stopping_condition.cpu_time_spent
        evaluations = algorithm.stopping_condition.evaluations
        _, best_fitness = algorithm.best_solution()

        print(f"--------------------{'-'*len(objfunc_name)}-------{'-'*len(alg_name)}-")
        print(f"Finished optimizing {objfunc_name} using {alg_name}:")
        print(f"\tNumber of generations: {iterations_accurate}")
        print(f"\tReal time spent: {spent_time:.4f}s")
        print(f"\tCPU time spent: {spent_cpu_time:.4f}s")
        print(f"\tNumber of fitness evaluations: {evaluations}")
        print(f"\tBest fitness: {best_fitness}")
        algorithm.search_strategy.extra_report()
