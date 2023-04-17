from __future__ import annotations
from abc import ABC, abstractmethod
from .Individual import Individual
from .ParamScheduler import ParamScheduler
from .ParentSelection import ParentSelection
from .Operator import Operator
from .SurvivorSelection import SurvivorSelection
import time

class Algorithm(ABC):
    """
    Population of the Genetic algorithm
    Note: for methods that use only one solution at a time,
          use a population of length 1 to store it.
    """

    def __init__(self, name: str="some algorithm", popSize: int = 100, params: Union[ParamScheduler, dict]=None, population=None):
        """
        Constructor of the GeneticPopulation class
        """

        self.name = name
        self.popsize = popSize

        if population is None:
            self.population = []
        else:
            self.population = population
        
        self.best = None

        self.param_scheduler = None
        if params is None:
            self.params = {}
        elif isinstance(params, ParamScheduler):
            self.param_scheduler = params
            self.params = self.param_scheduler.get_params()
        else:
            self.params = params

        self._find_operator_attributes()

    
    def _find_operator_attributes(self):
        attr_dict = vars(self).copy()

        self.parent_sel = []
        self.operators = []
        self.surv_sel = []

        for var_key in attr_dict:
            
            attr = attr_dict[var_key]

            if attr:
                # We have a parent selection method
                if isinstance(attr, ParentSelection):
                    self.parent_sel.append(attr)

                # We have an operator
                if isinstance(attr, Operator):
                    self.operators.append(attr)

                # We have a survivor selection method
                if isinstance(attr, SurvivorSelection):
                    self.surv_sel.append(attr)
                
                # We have a list of operators
                if isinstance(attr, list) and isinstance(attr[0], Operator):
                    self.operators += attr


    def best_solution(self) -> Tuple(Individual, float):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_fitness = self.best.fitness
        if self.best.objfunc.mode == "min":
            best_fitness *= -1

        return self.best.genotype, best_fitness

    # def initialize(self, objfunc: ObjectiveFunc):
    #     """
    #     Generates a random population of individuals
    #     """

    #     self.population = []
    #     for i in range(self.popsize):
    #         genotype = objfunc.decoder.encode(objfunc.random_solution())
    #         speed = objfunc.decoder.encode(objfunc.random_solution())
    #         new_indiv = Individual(objfunc, genotype, speed)

    #         if self.best is None or self.best.fitness < new_indiv.fitness:
    #             self.best = new_indiv

    #         self.population.append(new_indiv)

    def initialize(self, population: List[Individual]):
        """
        Generates a random population of individuals
        """
        
        self.population = population

        self.best = max(self.population, key=lambda x: x.fitness)

    def select_parents(self, population: List[Individual], progress: float = 0, history: List[float] = None) -> List[Individual]:
        """
        Selects the individuals that will be perturbed in this generation
        Returns the whole population if not implemented.
        """

        return population, range(len(population))

    @abstractmethod
    def perturb(self, parent_list: List[Individual], progress: float, objfunc: ObjectiveFunc, history: List[float]) -> List[Individual]:
        """
        Applies operators to the population in some way
        Returns the offspring generated.
        """

    def select_individuals(self, population: List[Individual], offspring: List[Individual], progress: float = 0, history: List[float] = None) -> List[Individual]:
        """
        Selects the individuals that will pass to the next generation.
        Returns the offspring if not implemented.
        """

        return offspring

    def update_params(self, progress: float):
        """
        Updates the parameters and the operators
        """
        

    def get_state(self):
        """
        Gets the current state of the algorithm as a dictionary.
        """

        data = {
            "name": self.name,
            "population_size": self.popsize,
        }

        if self.param_scheduler:
            data["param_scheduler"] = self.param_scheduler.get_state()
            data["params"] = self.param_scheduler.get_params()
        else:
            data["params"] = self.params

        if self.parent_sel:
            data["parent_sel"] = [par.get_state() for par in self.parent_sel]
        
        if self.operators:
            data["operators"] = [op.get_state() for op in self.operators]
        
        if self.surv_sel:
            data["survivor_sel"] = [surv.get_state() for surv in self.surv_sel]

        data["best_individual"] = self.best.get_state()

        data["population"] = [ind.get_state() for ind in self.population]

        # if self.param_scheduler:
        #     data["param_scheduler"] = self.param_scheduler.get_state()
        #     data["params"] = self.param_scheduler.get_params()
        # else:
        #     data["params"] = self.params

        return data

    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """

    def extra_report(self, show_plots: bool):
        """
        Specific information to display relevant to this algorithm
        """
