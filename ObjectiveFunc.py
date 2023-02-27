from abc import ABC, abstractmethod


class ObjectiveFunc(ABC):
    """
    Abstract Fitness function class.

    For each problem a new class will inherit from this one
    and implement the fitness function, random solution generation,
    mutation function and crossing of solutions.
    """

    def __init__(self, input_size, opt, name="some function"):
        """
        Constructor for the AbsObjectiveFunc class
        """

        self.name = name
        self.counter = 0
        self.input_size = input_size
        self.factor = 1
        self.opt = opt
        if self.opt == "min":
            self.factor = -1
    

    def __call__(self, solution, adjusted=True):
        """
        Shorthand for executing the objective function on a vector.
        """
        result = None
        if adjusted:
            result = self.fitness(solution)
        else:
            result = self.objective(solution)
        
        return result


    def fitness(self, solution):
        """
        Returns the value of the objective function given a vector changing the sign so that
        the optimization problem is solved by maximizing the fitness function.
        """
        self.counter += 1
        return self.factor * self.objective(solution)
    

    @abstractmethod
    def objective(self, vector):
        """
        Implementation of the objective function without adjusting the sign.
        """
    

    @abstractmethod
    def random_solution(self):
        """
        Returns a random vector that represents one viable solution to our optimization problem. 
        """
    

    @abstractmethod
    def check_bounds(self, vector):
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.
        """
