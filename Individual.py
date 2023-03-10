from copy import copy


class Indiv:
    """
    Individual that holds a tentative solution with 
    its fitness.
    """

    def __init__(self, objfunc, vector, speed=0, operator=None):
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

        old_vector = self.vector
        old_fitness = self.fitness

        self._vector = vector
        self.fitness_calculated = False
        
        if old_fitness < self.fitness:
            self.best = old_vector
        
        return self._vector


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

        new_vector = self.objfunc.check_bounds(self._vector + self.speed)
        new_indiv = Indiv(self.objfunc, new_vector, self.speed, self.operator)
        if self.fitness > new_indiv.fitness:
            new_indiv.best = self._vector
        return new_indiv


    @property
    def fitness(self):
        """
        Obtain the fitness of the individual, optimized to be calculated 
        only once per individual.
        """

        if not self.fitness_calculated:
            self._fitness = self.objfunc(self)
            self.fitness_calculated = True
        return self._fitness
    
    @fitness.setter
    def fitness(self, fit):
        """
        Obtain the fitness of the individual, optimized to be calculated 
        only once per individual.
        """
        
        self._fitness = fit