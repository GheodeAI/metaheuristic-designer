from __future__ import annotations
from typing import Tuple, Any
from copy import copy
import numpy as np
from numpy import ndarray
from .encodings import DefaultEncoding
from .utils import RAND_GEN


def evaluate_indiv(indiv):
    calculation_done = not indiv.fitness_calculated
    indiv.calculate_fitness()
    return indiv, calculation_done


class Population:
    """
    Individual that holds a tentative solution with its fitness.

    Parameters
    ----------
    objfunc: ObjectiveFunc
    genotype: Any
    speed: ndarray, optional
    encoding: Encoding, optional
    """

    objfunc: ObjectiveFunc
    genotype_set: Any
    speed_set: Any
    ages: ndarray
    encoding: Encoding
    pop_size: int
    vec_size: int
    fitness: ndarray
    fitness_calculated: ndarray
    historical_best_set: Any
    historical_best_fitness: ndarray
    best: Any
    best_fitness: float | ndarray

    def __init__(
        self,
        objfunc: ObjectiveFunc,
        genotype_set: Any,
        speed_set: ndarray = None,
        ages: ndarray = None,
        encoding: Encoding = None,
    ):
        """
        Constructor of the Individual class.
        """

        # Objective function
        self.objfunc = objfunc

        # Population of solutions
        self.genotype_set = genotype_set

        # Size of the population
        self.pop_size = len(genotype_set)

        # Speed of the individuals
        if isinstance(genotype_set, ndarray):
            self.vec_size = genotype_set.shape[1]

            if speed_set is None:
                speed_set = RAND_GEN.random(genotype_set.shape)
            self.speed_set = speed_set
        else:
            self.vec_size = None
            self.speed_set = None

        # Fitness of each individual in the population
        self.fitness = np.full(self.pop_size, -np.inf)
        self.fitness_calculated = np.zeros(self.pop_size)

        # Best solution found so far
        if len(genotype_set) > 0:
            self.best = self.genotype_set[0]
        else:
            self.best = None
        self.best_fitness = None

        # Best inidividual in each spot of the population
        self.historical_best_set = genotype_set
        self.historical_best_fitness = np.full(self.pop_size, -np.inf)

        # Ages of the individuals
        if ages is None:
            ages = np.zeros(self.pop_size)
        self.ages = ages

        # Encoding to use
        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

    def __len__(self):
        return len(self.genotype_set)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index <= len(self):
            return self.genotype_set[self.index - 1]
        raise StopIteration

    def __copy__(self) -> Population:
        """
        Returns a copy of the Individual.
        """

        copied_pop = Population(self.objfunc, copy(self.genotype_set), copy(self.speed_set), ages=copy(self.ages), encoding=self.encoding)
        copied_pop.historical_best_set = copy(self.historical_best_set)
        copied_pop.fitness_calculated = copy(self.fitness_calculated)
        copied_pop.fitness = copy(self.fitness)
        copied_pop.best = copy(self.best)
        copied_pop.best_fitness = copy(self.best_fitness)

        return copied_pop

    def best_solution(self) -> Tuple[Any, float]:
        best_fitness = self.best_fitness
        if self.objfunc.mode == "min":
            best_fitness *= -1

        return self.best, best_fitness

    def update_genotype_set(self, genotype_set, speed_set=None):
        if speed_set is None and len(genotype_set) == len(self.genotype_set):
            speed_set = self.speed_set

        # Create copy of the population
        new_population = Population(self.objfunc, copy(genotype_set), copy(speed_set), ages=copy(self.ages), encoding=self.encoding)

        # Check which individuals have been changed
        if len(genotype_set) != len(self.genotype_set):
            new_population.ages = np.zeros(len(genotype_set))
            new_population.fitness_calculated = np.zeros(len(genotype_set))
        elif isinstance(genotype_set, ndarray):
            new_population.fitness_calculated = np.all(self.genotype_set == new_population.genotype_set, axis=1)
        else:
            new_population.fitness_calculated = np.asarray(
                [new_genotype != genotype] for new_genotype, genotype in zip(self.genotype_set, new_population.genotype_set)
            )

        # Calculate the fitness of the new individuals
        old_fitness = self.fitness
        new_fitness = new_population.calculate_fitness()
        new_population.fitness = new_fitness

        # Update the historical best individuals
        if len(genotype_set) != len(self.genotype_set):
            new_population.historical_best_fitness = new_fitness
        elif isinstance(genotype_set, ndarray):
            new_population.historical_best_set = np.where((new_fitness < old_fitness)[:, None], new_population.genotype_set, genotype_set)
            new_population.historical_best_fitness = np.maximum(old_fitness, new_fitness)
        else:
            new_population.historical_best_fitness = np.maximum(old_fitness, new_fitness)

        return new_population

    def take_selection(self, selection_idx):
        sel_genotype_set = copy(self.genotype_set[selection_idx])
        sel_speed_set = copy(self.speed_set[selection_idx])
        sel_ages = copy(self.ages[selection_idx])

        selected_pop = Population(self.objfunc, sel_genotype_set, sel_speed_set, ages=sel_ages, encoding=self.encoding)
        selected_pop.historical_best_set = copy(self.historical_best_set[selection_idx])
        selected_pop.historical_best_fitness = copy(self.historical_best_fitness[selection_idx])
        selected_pop.fitness_calculated = copy(self.fitness_calculated[selection_idx])
        selected_pop.fitness = copy(self.fitness[selection_idx])
        selected_pop.best = copy(self.best)
        selected_pop.best_fitness = copy(self.best_fitness)

        return selected_pop

    @staticmethod
    def _join(population1, population2):
        joined_genotype_set = np.concatenate((population1.genotype_set, population2.genotype_set), axis=0)
        joined_speed_set = np.concatenate((population1.speed_set, population2.speed_set), axis=0)
        joined_ages = np.concatenate((population1.ages, population2.ages))

        joined_pop = Population(population1.objfunc, joined_genotype_set, joined_speed_set, ages=joined_ages, encoding=population1.encoding)
        joined_pop.historical_best_set = np.concatenate((population1.historical_best_set, population2.historical_best_set), axis=0)
        joined_pop.historical_best_fitness = np.concatenate((population1.historical_best_fitness, population2.historical_best_fitness))
        joined_pop.fitness_calculated = np.concatenate((population1.fitness_calculated, population2.fitness_calculated))
        joined_pop.fitness = np.concatenate((population1.fitness, population2.fitness))
        if population1.best_fitness > population2.best_fitness:
            joined_pop.best = population1.best
            joined_pop.best_fitness = population1.best_fitness
        else:
            joined_pop.best = population2.best
            joined_pop.best_fitness = population2.best_fitness

        return joined_pop

    def join(self, other_population):
        return Population._join(self, other_population)

    def sort_population(self):
        fitness_order = np.argsort(self.fitness)

        self.genotype_set = self.genotype_set[fitness_order]
        self.speed_set = self.speed_set[fitness_order]
        self.ages = self.ages[fitness_order]
        self.historical_best_set = self.historical_best_set[fitness_order]
        self.historical_best_fitness = self.historical_best_fitness[fitness_order]
        self.fitness_calculated = self.fitness_calculated[fitness_order]
        self.fitness = self.fitness[fitness_order]

        return self

    def update_best_from_parents(self, parents):
        if self.best is None or (parents.best is not None and self.best_fitness < parents.best_fitness):
            self.best = parents.best
            self.best_fitness = parents.best_fitness
        return self

    @staticmethod
    def select_survivors(parents, offspring, selection, size_limit=None):
        """
        selection: (n_individuals, 1)
        """

        if size_limit is None:
            size_limit = parents.pop_size
        min_size = min(parents.pop_size, offspring.pop_size)

        parent_genotypes = parents.genotype_set
        offspring_genotypes = offspring.genotype_set

        new_genotype_set = np.where

    def increase_age(self):
        self.ages += 1
        self.ages[self.fitness_calculated == 1] = 0

    def repeat(self, amount=2):
        if isinstance(self.genotype_set, ndarray):
            genotype_set = np.tile(self.genotype_set, (amount, 1))
            speed_set = np.tile(self.speed_set, (amount, 1))
        else:
            genotype_set = self.genotype_set * amount
            speed_set = None
        ages = np.tile(self.ages, amount)
        return Population(self.objfunc, genotype_set, speed_set, ages=ages, encoding=self.encoding)

    def calculate_fitness(self, parallel=False, threads=8) -> float:
        """
        Calculates the fitness of the individual if it has not been calculated before
        """

        self.fitness = self.objfunc.fitness(self, parallel=parallel, threads=threads)

        if self.best_fitness is None or np.any(self.fitness > self.best_fitness):
            best_idx = np.argmax(self.fitness)
            self.best = self.genotype_set[best_idx]
            self.best_fitness = self.fitness[best_idx]

        return self.fitness

    def repair_solutions(self):
        """
        Repairs the solutions in the population
        """

        for idx, indiv in enumerate(self.genotype_set):
            self.genotype_set[idx] = self.objfunc.repair_solution(indiv)
            if self.speed_set is not None:
                self.speed_set[idx] = self.objfunc.repair_speed(self.speed_set[idx])

        return self

    def get_state(self, show_speed: bool = True, show_best: bool = False) -> dict:
        """
        Gets the current state of the algorithm as a dictionary.

        Parameters
        ----------
        show_speed: bool, optional
            Save the speed of the individual.
        show_best: bool, optional
            Save the best parent of this individual.

        Returns
        -------
        state: dict
            The current state of this individual.
        """

        data = {"genotype": self._genotype, "fitness": self._fitness}

        if show_speed:
            data["speed"] = self.speed

        if show_best:
            data["best_genotype"] = self.best

        return data

    def __repr__(self):
        return (
            "Population{"
            f"\n\tobjfunc = {self.objfunc.name}"
            f"\n\tgenotype_set = {self.genotype_set}"
            f"\n\tspeed_set = {self.speed_set}"
            f"\n\tages = {self.ages}"
            f"\n\tpop_size = {self.pop_size}"
            f"\n\tvec_size = {self.vec_size}"
            f"\n\tfitness = {self.fitness}"
            f"\n\tfitness_calculated = {self.fitness_calculated}"
            f"\n\thistorical_best_set = {self.historical_best_set}"
            f"\n\thistorical_best_fitness = {self.historical_best_fitness}"
            f"\n\tbest = {self.best}"
            f"\n\tbest_fitness = {self.best_fitness}"
            "\n}"
        )
