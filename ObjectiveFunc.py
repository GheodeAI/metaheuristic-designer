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
    

    def __call__(self, indiv, adjusted=True):
        """
        Shorthand for executing the objective function on a vector.
        """
        
        result = None
        
        result = self.fitness(indiv, adjusted)
        
        return result


    def decode(self, indiv):
        """
        Transforms the vector contained inside an individual in our algorithm to an
        usable format for our objective function.
        """

        return indiv.vector
    

    def fitness(self, indiv, adjusted=True):
        """
        Returns the value of the objective function given a vector changing the sign so that
        the optimization problem is solved by maximizing the fitness function.
        """

        self.counter += 1
        value = self.objective(self.decode(indiv))

        if adjusted:
            value = self.factor * value
        
        return value
    
    
    def apply_fitness(self, indiv, adjusted=True):
        """
        Calculates the fitness for a given individual updating its properties in the process.
        """

        if not indiv.fitness_calculated:
            fit_value = self.fitness(indiv) - self.penalize(indiv)
            indiv.fitnes = fit_value
            indiv.fitness_calculated = True
        
        return indiv
    

    @abstractmethod
    def objective(self, vector):
        """
        Implementation of the objective function.
        """
    

    @abstractmethod
    def random_solution(self):
        """
        Returns a random vector that represents one viable solution to our optimization problem. 
        """
    

    @abstractmethod
    def repair_solution(self, vector):
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.
        """
    

    def penalize(self, indiv):
        """
        Gives a penalization to the fitness value of an individual if it violates any constraints propotional
        to how far it is to a viable solution.

        If not implemented always returns 0.
        """

        return 0
