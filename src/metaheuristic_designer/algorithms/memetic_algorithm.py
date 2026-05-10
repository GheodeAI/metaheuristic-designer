"""
Memetic algorithm that enhances a base optimisation strategy with local search.
"""

from __future__ import annotations
from copy import copy
import logging
from typing import Optional, Tuple

from ..utils import MaskLike
from ..population import Population
from ..objective_function import ObjectiveFunc
from ..search_strategy import SearchStrategy
from ..stopping_condition import StoppingCondition
from ..parent_selection_base import ParentSelection
from ..algorithm import Algorithm
from ..reporter import Reporter
from ..history_tracker import HistoryTracker

logger = logging.getLogger(__name__)


class MemeticAlgorithm(Algorithm):
    """
    A memetic algorithm that interleaves a local search step into the main loop.

    After the usual perturbation and evaluation, a subset of the offspring
    is improved by a separate :class:`SearchStrategy` (the local search).
    The improved individuals can either replace the original offspring
    (Lamarckian, ``keep_improved_solutions=True``) or only transfer their
    fitness (Baldwinian, ``keep_improved_solutions=False``).

    Parameters
    ----------
    objfunc : ObjectiveFunc
        Objective function to optimise.
    search_strategy : SearchStrategy
        The main search strategy.
    local_search : SearchStrategy
        Strategy used for local improvement.
    improvement_selection : ParentSelection
        How to choose which offspring are improved.
    local_search_frequency : int, optional
        Apply local search every *n* generations (default 1).
    local_search_depth : int, optional
        Number of local search iterations per application (default 1).
    keep_improved_solutions : bool, optional
        If ``True`` (Lamarckian), the improved genotypes replace the
        original offspring.  If ``False`` (Baldwinian), only the fitness
        values are transferred.
    name : str, optional
        Display name; defaults to ``"Memetic {strategy_name}"``.
    stop_cond, progress_metric, ... : optional
        See :class:`Algorithm`.
    """

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        search_strategy: SearchStrategy,
        local_search: SearchStrategy,
        improvement_selection: ParentSelection,
        local_search_frequency: int = 1,
        local_search_depth: int = 1,
        keep_improved_solutions: bool = True,
        name: Optional[str] = None,
        stop_cond: str = "real_time_limit",
        progress_metric: Optional[str] = None,
        max_iterations: int = 1000,
        max_evaluations: int = 1e5,
        real_time_limit: float = 60.0,
        cpu_time_limit: float = 60.0,
        objective_target: float = 1e-10,
        max_patience: int = 100,
        track_median: bool = False,
        track_worst: bool = False,
        track_complete: bool = False,
        track_diversity: bool = False,
        stopping_condition: Optional[StoppingCondition] = None,
        reporter: Optional[str | Reporter] = None,
        history_tracker: Optional[HistoryTracker] = None,
        parallel: bool = False,
        threads: int = 8,
    ):
        if name is None:
            name = f"Memetic {search_strategy.name}"

        self.local_search = local_search
        self.improvement_selection = improvement_selection
        self.local_search_frequency = local_search_frequency
        self.local_search_depth = local_search_depth
        self.keep_improved_solutions = keep_improved_solutions

        if not local_search.operator.preserves_order:
            logger.warning(
                "Local search implements an operator that doesn't preserve order (%s). The fitness calculation might be corrupted.",
                local_search.operator.name,
            )

        if not local_search.survivor_sel.preserves_order:
            logger.warning(
                "Local search implements a survivos selection method that doesn't preserve order (%s). The fitness calculation might be corrupted.",
                local_search.survivor_sel.name,
            )

        self.local_search_counter = 0

        super().__init__(
            objfunc=objfunc,
            search_strategy=search_strategy,
            name=name,
            stop_cond=stop_cond,
            progress_metric=progress_metric,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            real_time_limit=real_time_limit,
            cpu_time_limit=cpu_time_limit,
            objective_target=objective_target,
            max_patience=max_patience,
            track_median=track_median,
            track_worst=track_worst,
            track_full_population=track_complete,
            track_diversity=track_diversity,
            stopping_condition=stopping_condition,
            reporter=reporter,
            history_tracker=history_tracker,
            parallel=parallel,
            threads=threads,
        )

    def initialize(self):
        """Create and evaluate the initial population, then sync the local search.

        Returns
        -------
        Population
            The evaluated initial population.
        """
        
        population = super().initialize()
        self.local_search.population = population
        return population

    def _do_local_search(self, offspring) -> Tuple[Population, MaskLike]:
        """Apply local search to selected individuals and merge the result.

        Parameters
        ----------
        offspring : Population
            The offspring after global perturbation and evaluation.

        Returns
        -------
        tuple
            (improved_population, chosen_indices)
        """

        # Select individuals to improve
        selected_to_improve = self.improvement_selection(offspring)
        chosen_idx = self.improvement_selection.last_selection_idx
        self._log_debug("Selected individuals to improve\n%s", selected_to_improve)

        next_selected_population = selected_to_improve
        for _ in range(self.local_search_depth):
            # Apply mutation to individuals
            mutated_offspring = self.local_search.perturb(next_selected_population)
            self._log_debug("Mutated inside local search\n%s", mutated_offspring)

            # Select the best individuals (ensure the population size is the same as the parents)
            improved_offspring = self.local_search.select_individuals(next_selected_population, mutated_offspring)
            self._log_debug("Selected inside local search\n%s", improved_offspring)

            # Assign improved individuals to the population
            offspring = offspring.apply_selection(improved_offspring, chosen_idx)
            offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
            next_selected_population = offspring

        self._log_debug("Applied local search\n%s", offspring)

        return offspring, chosen_idx

    def _log_debug(self, text, population):
        """Log a debug message with the population's compact representation."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(text, population.debug_repr())

    def step(self, population: Population = None, time_start: float = 0, verbose: bool = False) -> Population:
        """Execute one memetic iteration (global step + optional local search).

        Parameters
        ----------
        population : Population, optional
            The population at the start of the iteration.
        time_start : float, optional
            Start time (unused, kept for interface compatibility).
        verbose : bool, optional
            Whether to produce verbose output (unused).

        Returns
        -------
        Population
            The population after the iteration.
        """

        # Get the population of this generation
        if population is None:
            population = self.search_strategy.population
        else:
            self.search_strategy.population = population

        self._log_debug("Original population:\n%s", population)

        # Generate their parents
        parents = self.search_strategy.select_parents(population)
        self._log_debug("Parent selection\n%s", parents)

        # Evolve the selected parents
        offspring = self.search_strategy.perturb(parents)
        self._log_debug("Perturbed\n%s", offspring)

        # Get the fitness of the individuals
        offspring = self.search_strategy.evaluate_population(offspring, self.parallel, self.threads)
        self._log_debug("Evaluated\n%s", offspring)

        # Perform a local search on the best individuals
        offspring_memetic = copy(offspring)
        self.local_search_counter += 1
        if self.local_search_counter % self.local_search_frequency == 0:
            offspring_memetic, chosen_idx = self._do_local_search(offspring_memetic)

            if not self.keep_improved_solutions:
                fitness_obtained = offspring_memetic.fitness
                offspring_memetic = offspring
                offspring_memetic.fitness[chosen_idx] = fitness_obtained[chosen_idx]

        # Select the individuals that remain for the next generation
        new_population = self.search_strategy.select_individuals(population, offspring_memetic)
        self._log_debug("Selected\n%s", new_population)

        # Assign the newly generated population
        self.search_strategy.population = new_population

        # Get information about the algorithm to track it's progress
        self.search_strategy.step(self.progress)
        self.local_search.step(self.progress)
        self._log_debug("Updated end\n%s", new_population)

        return new_population

    def get_state(self, store_population: bool = False) -> dict:
        """Extend :meth:`Algorithm.get_state` with local search data.

        Parameters
        ----------
        store_population : bool, optional
            See :class:`Algorithm`.

        Returns
        -------
        dict
            Dictionary containing the memetic algorithm state.
        """

        data = super().get_state(store_population)

        # Add parent selection method for local search
        data["improvement_selection"] = self.improvement_selection.get_state()

        # Add local search data
        local_search_data = self.local_search.get_state(store_population=False)
        data["local_search_state"] = local_search_data

        return data

    def step_info(self, start_time: int = 0):
        """Print per-generation information including local search details."""
        super().step_info(start_time)
        self.local_search.extra_step_info()
        print()
