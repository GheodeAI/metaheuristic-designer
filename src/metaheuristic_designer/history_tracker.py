from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .algorithm import Algorithm

class HistoryTracker:
    def __init__(
        self,
        track_best = True,
        track_median = False,
        track_worst = False,
        track_complete = False,
        track_diversity = False,
    ):
        self.track_best = track_best
        self.track_median = track_median
        self.track_worst = track_worst
        self.track_complete = track_complete
        self.track_diversity = track_diversity

        self.best_solutions = []
        self.median_solutions = []
        self.worst_solutions = []

        self.best_fitness = []
        self.median_fitness = []
        self.worst_fitness = []

        self.complete_population = []
        self.diversity = []

    def step(self, algorithm: Algorithm):
        population = algorithm.population
        solutions = population.decode()
        fitness_array = population.fitness
        fitness_order = np.argsort(fitness_array)

        if self.track_complete:
            self.complete_population.append(solutions)

        if self.track_best:
            best_idx = fitness_order[0]
            self.best_solutions.append(solutions[best_idx])
            self.best_fitness.append(fitness_array[best_idx])

        if self.track_median:
            half_size = len(fitness_array)//2
            if len(fitness_array) % 2 == 0:
                median_idx = fitness_order[half_size-1] 
            else:
                median_idx = fitness_order[half_size]
            self.median_solutions.append(solutions[median_idx])
            self.median_fitness.append(fitness_array[median_idx])

        if self.track_worst:
            worst_idx = fitness_order[-1]
            self.worst_solutions.append(solutions[worst_idx])
            self.worst_fitness.append(fitness_array[worst_idx])
        
        if self.track_diversity:
            raise NotImplementedError()


    def get_state(self):
        data = {
            "class_name": self.__class__.__name__,
        }

        if self.track_best:
            data["best_solutions"] = self.best_solutions
            data["best_fitness"] = self.best_fitness

        if self.track_median:
            data["median_solutions"] = self.median_solutions
            data["median_fitness"] = self.median_fitness

        if self.track_worst:
            data["worst_solutions"] = self.worst_solutions
            data["worst_fitness"] = self.worst_fitness
        
        if self.track_complete:
            data["populations"] = self.complete_population
        
        if self.track_diversity:
            data["divesity"] = self.diversity
        
        return data


