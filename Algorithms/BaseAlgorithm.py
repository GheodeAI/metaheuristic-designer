from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    Population of the Genetic algorithm
    Note: for methods that use only one solution at a time, use a population of length 1 to store it.
    """

    def __init__(self, name="some algorithm"):
        """
        Constructor of the GeneticPopulation class
        """

        self.name = name
        self.population = []

    @abstractmethod
    def best_solution(self, objfunc):
        """
        Gives the best solution found by the algorithm and its fitness
        """

    @abstractmethod
    def initialize(self, objfunc):
        """
        Generates a random population of individuals
        """
    

    def select_parents(self, population, progress=0, history=None):
        """
        Selects the individuals that will be perturbed in this generation
        Returns the whole population if not implemented.
        """

        return population
    
    @abstractmethod
    def perturb(self, parent_list, progress, objfunc, history):
        """
        Applies operators to the population in some way
        Returns the offspring generated.
        """
            

    def select_individuals(self, population, offspring, progress=0, history=None):
        """
        Selects the individuals that will pass to the next generation.
        Returns the offspring if not implemented.
        """

        return offspring


    @abstractmethod
    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """


    def extra_step_info(self):
        """
        Specific information to display relevant to this algorithm
        """
    
    def extra_report(self, show_plots):
        """
        Specific information to display relevant to this algorithm
        """