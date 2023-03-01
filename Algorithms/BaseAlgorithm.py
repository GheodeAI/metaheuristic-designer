from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    Population of the Genetic algorithm
    """

    def __init__(self, objfunc, name="some algorithm"):
        """
        Constructor of the GeneticPopulation class
        """

        self.name = name
        self.objfunc = objfunc
    

    @abstractmethod
    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """


    @abstractmethod
    def initialize(self):
        """
        Generates a random population of individuals
        """
    

    @abstractmethod
    def step(self, progress, history=None):
        """
        Performs a step of the algorithm
        """


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