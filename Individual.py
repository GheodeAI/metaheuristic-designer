import numpy as np
from copy import copy
from .ObjectiveFunc import ObjectiveFunc
from .Operators import Operator


class Indiv:
    """
    Individual that holds a tentative solution with 
    its fitness.
    """

    def __init__(self, objfunc: ObjectiveFunc, vector: np.ndarray, speed: np.ndarray=0, operator: Operator=None):
        """
        Constructor of the Individual class.
        """

        self.objfunc = objfunc
        self._vector = vector
        self.speed = speed
        self.operator = operator
        self._fitness = 0
        self.fitness_calculated = False
        self.best = vector
        self.is_dead = False
    

    def __copy__(self):
        """
        Returns a copy of the Individual.
        """

        copied_ind = Indiv(self.objfunc, copy(self._vector), copy(self.speed), self.operator)
        copied_ind._fitness = self._fitness
        copied_ind.fitness_calculated = self.fitness_calculated
        copied_ind.best = copy(self.best)
        return copied_ind
    

    @property
    def vector(self):
        """
        Gets the value of the vector.
        """

        return self._vector


    @vector.setter
    def vector(self, vector):
        """
        Sets the value of the vector.
        """

        self.fitness_calculated = False
        self._vector = vector
    

    def store_best(self, past_indiv):
        """
        Stores the vector that yeided the best fitness between the one the indiviudal has and another input vector
        """

        if self.fitness < past_indiv.fitness:
            self.best = past_indiv.vector


    def reproduce(self, population):
        """
        Apply the operator to obtain a new individual.
        """

        new_vector = self.operator(self, population, self.objfunc)
        new_vector = self.objfunc.check_bounds(new_vector)
        return Indiv(self.objfunc, new_vector, self.speed, self.operator)


    def apply_speed(self):
        """
        Apply the speed to obtain an individual with a new position.
        """

        return Indiv(self.objfunc, self._vector + self.speed, self.speed, self.operator)


    @property
    def fitness(self):
        """
        Obtain the fitness of the individual, optimized to be calculated 
        only once per individual.
        """

        if not self.fitness_calculated:
            self.fitness = self.objfunc(self)
        return self._fitness
    
    @fitness.setter
    def fitness(self, fit):
        """
        Obtain the fitness of the individual, optimized to be calculated 
        only once per individual.
        """
        
        self._fitness = fit
        self.fitness_calculated = True