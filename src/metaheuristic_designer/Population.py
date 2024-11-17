from __future__ import annotations
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

        self.objfunc = objfunc
        print(genotype_set)

        self.genotype_set = genotype_set
        self.historical_best_set = genotype_set
        if isinstance(genotype_set, ndarray):
            # assert genotype_set.ndim == 2
            self.pop_size = genotype_set.shape[0]
            self.vec_size = genotype_set.shape[1]

            if speed_set is None:
                speed_set = RAND_GEN.uniform(0, 1, size=genotype_set.shape)
            self.speed_set = speed_set
        else:
            self.pop_size = len(genotype_set)
            self.speed_set = None
        
        self.best_fitness = None
        self.fitness_calculated = np.zeros(self.pop_size)
        self.fitness = np.empty(self.pop_size)

        if ages is None:
            ages = np.zeros(self.pop_size)
        self.ages = ages

        self.best = self.genotype_set[0]

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
            return self.genotype_set[self.index-1]
        raise StopIteration

    def __copy__(self) -> Population:
        """
        Returns a copy of the Individual.
        """

        copied_pop = Population(self.objfunc, copy(self.genotype_set), copy(self.speed_set), ages=copy(self.ages), encoding=self.encoding)
        copied_pop.historical_best_set = copy(self.historical_best_set)
        copied_pop.best_fitness = copy(self.best_fitness)
        copied_pop.fitness_calculated = copy(self.fitness_calculated)
        copied_pop.fitness = copy(self.fitness)
        copied_pop.best = copy(self.best)

        return copied_pop
    
    def update_genotype_set(self, genotype_set, speed_set=None):
        if speed_set is None:
            speed_set = copy(self.speed_set)
        
        new_population = Population(self.objfunc, genotype_set, speed_set, ages=copy(self.ages), encoding=self.encoding)
        if len(genotype_set) != len(self.genotype_set):
            new_population.ages = np.zeros(len(genotype_set))
            new_population.fitness_calculated = np.zeros(len(genotype_set))
        elif isinstance(genotype_set, ndarray):
            new_population.fitness_calculated = np.any(genotype_set != new_population.genotype_set, axis=1)
        else:
            new_population.fitness_calculated = np.asarray([new_genotype != genotype] for new_genotype, genotype in zip(genotype_set, new_population.genotype_set))
        new_population.calculate_fitness()

        # print(new_population.ages)
        # print(genotype_set)
        # print(new_population.fitness_calculated)
        new_population.ages += 1
        new_population.ages[new_population.fitness_calculated == 1] = 0

        return new_population

    def calculate_fitness(self, parallel=False, threads=8) -> float:
        """
        Calculates the fitness of the individual if it has not been calculated before
        """

        # if parallel:
        #     with Pool(threads) as p:
        #         result_pairs = p.map(evaluate_indiv, population)
        #     population, calculated = map(list, zip(*result_pairs))
        #     objfunc.counter += sum(calculated)
        # else:
        #     # [population.calculate_fitness() for indiv in population]
        #     population.calculate_fitness()

        # for idx, genotype in enumerate(self.genotype_set):
        #     if not self.fitness_calculated[idx]:
        #         self.fitness[idx] = self.objfunc(self)
        self.fitness = self.objfunc(self)
        
        if self.best_fitness is None or np.any(self.fitness > self.best_fitness):
            best_idx = np.argmax(self.fitness)
            self.best = self.genotype_set[best_idx]
            self.best_fitness = self.fitness[best_idx]

        #TODO: Update historical best set
        
        return self.fitness
    
    def repair_solutions(self):
        """
        Repairs the solutions in the population
        """

        # for indiv in population:
        #     indiv.genotype = objfunc.repair_solution(indiv.genotype)
        #     indiv.speed = objfunc.repair_speed(indiv.speed)

        for idx, indiv in self.genotype_set:
            self.genotype_set[idx] = self.objfunc.repair_solution(indiv)

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
