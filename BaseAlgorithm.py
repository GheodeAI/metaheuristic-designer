from __future__ import annotations
from typing import Tuple, List
from abc import ABC, abstractmethod
from .Individual import Indiv
import time


class BaseAlgorithm(ABC):
    """
    Population of the Genetic algorithm
    Note: for methods that use only one solution at a time, use a population of length 1 to store it.
    """

    def __init__(self, name: str="some algorithm", popSize: int = 100):
        """
        Constructor of the GeneticPopulation class
        """

        self.name = name
        self.popsize = popSize
        self.population = []
        self.best = None


    def best_solution(self) -> Tuple(Indiv, float):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best.fitness
        if self.best.objfunc.opt == "min":
            best_fitness *= -1        

        return self.best.vector, best_fitness


    def initialize(self, objfunc: ObjectiveFunc):
        """
        Generates a random population of individuals
        """

        self.population = []
        self.best = None
        for i in range(self.popsize):
            genotype = objfunc.decoder.encode(objfunc.random_solution())
            new_indiv = Indiv(objfunc, genotype)

            if self.best is None or self.best.fitness < new_indiv.fitness:
                self.best = new_indiv

            self.population.append(new_indiv)
    

    def select_parents(self, population: List[Indiv], progress: float = 0, history: List[float] = None) -> List[Indiv]:
        """
        Selects the individuals that will be perturbed in this generation
        Returns the whole population if not implemented.
        """
        return population
    
    @abstractmethod
    def perturb(self, parent_list: List[Indiv], progress: float, objfunc: ObjectiveFunc, history: List[float]) -> List[Indiv]:
        """
        Applies operators to the population in some way
        Returns the offspring generated.
        """
            

    def select_individuals(self, population: List[Indiv], offspring: List[Indiv], progress: float = 0, history: List[float] = None)-> List[Indiv]:
        """
        Selects the individuals that will pass to the next generation.
        Returns the offspring if not implemented.
        """

        return offspring

    @abstractmethod
    def update_params(self, progress: float):
        """
        Updates the parameters and the operators
        """


    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """
    

    def extra_report(self, show_plots: bool):
        """
        Specific information to display relevant to this algorithm
        """