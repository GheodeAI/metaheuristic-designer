"""
Hybrid search strategy based on memetic theory from biology.
"""

from copy import copy
from typing import Optional, Tuple

from ...population import Population
from ...operator import Operator
from ...parent_selection import ParentSelection
from ...utils import MaskLike, RNGLike
from ...search_strategy import SearchStrategy


class MemeticStrategy(SearchStrategy):
    """Strategy that combines a main search strategy with a local search procedure that improves
    solutions after they are evolved.

    Parameters
    ----------
    main_strategy : SearchStrategy
        Main search strategy used in the optimization algorithm.
    local_search_heuristic : SearchStrategy
        Local search procedure used to improve solutions after evolution.
    local_search_depth : int, optional
        Number of times to repeat the local search procedure per iteration, by default 1
    keep_improved_solutions : str, optional
        Whether to keep the improved solutions for the next iteration (Lamarkian memetic algorithms) or
        to just update the fitness keeping the original solution values (Baldwinian memetic algorithms), by default True
    improvement_selection : ParentSelection, optional
        Selection method with which to pick the solutions that will be improved with local search, by default None
    random_state : Operator[RNGLike], optional
        Random number generator, by default None
    """

    def __init__(
        self,
        main_strategy: SearchStrategy,
        local_search_heuristic: SearchStrategy,
        local_search_depth: int = 1,
        local_search_frequency: int = 1,
        keep_improved_solutions: bool = True,
        improvement_selection: ParentSelection = None,
        random_state: Optional[RNGLike] = None,
    ):
        self.main_strategy = main_strategy
        self.local_search_heuristic = local_search_heuristic
        self.improvement_selection = improvement_selection
        self.keep_improved_solutions = keep_improved_solutions
        self.local_search_counter = 0

        super().__init__(
            main_strategy.initializer,
            random_state=random_state,
            # Forced kwargs
            local_search_depth=local_search_depth,
<<<<<<< HEAD
            local_search_frequency=local_search_frequency
=======
            local_search_frequency=local_search_frequency,
>>>>>>> feature/stats
        )

    def _do_local_search(self, offspring: Population) -> Tuple[Population, MaskLike]:
        """Apply the local search procedure to a set of solutions.

        Parameters
        ----------
        offspring : Population
            Population of solutions to improve.

        Returns
        -------
        Tuple[Population, MaskLike]
            A pair of the improved complete population and mask indicating which solutions were chosen.
        """

        selected_to_improve = self.improvement_selection(offspring)
        chosen_idx = self.improvement_selection.last_selection_idx

        prev_population = selected_to_improve
        for _ in range(self.params.local_search_depth):
            population = self.local_search_heuristic.parent_sel.select(prev_population)
            population = self.local_search_heuristic.operator.evolve(population)
<<<<<<< HEAD
            improved_offspring = self.local_search_heuristic.survivor_sel.select(population, prev_population)
=======
            improved_offspring = self.local_search_heuristic.survivor_sel.select(prev_population, population)
>>>>>>> feature/stats

            # Assign improved individuals to the population
            offspring = offspring.apply_selection(improved_offspring, chosen_idx)
            offspring = offspring.calculate_fitness()
            prev_population = improved_offspring

        return offspring, chosen_idx

    def update(self, progress: float):
        super().update(progress)
        self.main_strategy.update(progress)
        self.local_search_heuristic.update(progress)
        self.improvement_selection.update(progress)

    def step(self, prev_population: Population):
        population = self.main_strategy.parent_sel.select(prev_population)  # implicit copy
        population = self.main_strategy.operator.evolve(population, self.initializer)

        # Do local search on perturbed individuals
        population_memetic = copy(population)
        self.local_search_counter += 1
        if self.local_search_counter >= self.params.local_search_frequency:
            population_memetic, chosen_idx = self._do_local_search(population_memetic)

            if not self.keep_improved_solutions:
                fitness_obtained = population_memetic.fitness
                population_memetic = population
                population_memetic.fitness[chosen_idx] = fitness_obtained[chosen_idx]

            self.local_search_counter = 0

        population_memetic = population_memetic.repair_solutions()
        population_memetic = population_memetic.calculate_fitness()

        population = self.main_strategy.survivor_sel.select(population=prev_population, offspring=population_memetic)
        return population

    def get_state(self) -> dict:
        data = {
            "main_strategy": self.main_strategy.get_state(),
            "local_search_heuristic": self.local_search_heuristic.get_state(),
            "improvement_selection": self.improvement_selection.get_state(),
        }

        return data
