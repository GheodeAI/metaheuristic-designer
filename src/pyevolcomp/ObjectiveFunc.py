from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
# from .Decoders import DefaultDecoder
from .Initializers import UniformVectorInitializer


class ObjectiveFunc(ABC):
    """
    Abstract Fitness function class.

    For each problem a new class will inherit from this one
    and implement the fitness function, random solution generation,
    mutation function and crossing of solutions.
    """

    def __init__(self, mode: str = "max", name: str = "some function"):
        """
        Constructor for the AbsObjectiveFunc class
        """

        self.name = name
        self.counter = 0
        self.factor = 1
        
        self.mode = mode
        if mode not in ["max", "min"]:
            raise ValueError("Optimization objective (mode) must be \"min\" or \"max\".")
        
        if self.mode == "min":
            self.factor = -1

    def __call__(self, indiv: Indiv, adjusted: bool = True) -> float:
        """
        Shorthand for executing the objective function on a vector.
        """

        return self.fitness(indiv, adjusted)

    def fitness(self, indiv: Indiv, adjusted: bool = True) -> float:
        """
        Returns the value of the objective function given a vector changing the sign so that
        the optimization problem is solved by maximizing the fitness function.
        """

        self.counter += 1
        solution = indiv.encoding.decode(indiv.genotype)
        value = self.objective(solution)

        if adjusted:
            value = self.factor * value - self.penalize(indiv)

        return value

    @abstractmethod
    def objective(self, vector: np.ndarray) -> float:
        """
        Implementation of the objective function.
        """

    @abstractmethod
    def random_solution(self) -> np.ndarray:
        """
        Returns a random vector that represents one viable solution to our optimization problem.
        """

    @abstractmethod
    def repair_solution(self, vector: np.ndarray) -> np.ndarray:
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.
        """

    def repair_speed(self, speed):
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.
        """

        return self.repair_solution(speed)

    def penalize(self, indiv: Indiv) -> float:
        """
        Gives a penalization to the fitness value of an individual if it violates any constraints propotional
        to how far it is to a viable solution.

        If not implemented always returns 0.
        """

        return 0


class ObjectiveVectorFunc(ObjectiveFunc):
    def __init__(self, vecsize: int, mode: str = "max", low_lim: float = -100, up_lim: float = 100, name: str = "some function"):
        super().__init__(mode, name)

        self.vecsize = vecsize
        self.low_lim = low_lim
        self.up_lim = up_lim
    
    def random_solution(self) -> np.ndarray:
        return np.random.uniform(self.low_lim, self.up_lim, self.vecsize)

    def repair_solution(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(vector, self.low_lim, self.up_lim)


class ObjectiveFromLambda(ObjectiveVectorFunc):
    def __init__(self, obj_func: Callable, input_size: int, opt: str = "max", low_lim: float = -100, up_lim: float = 100, name: str = None):
        """
        Constructor for the AbsObjectiveFunc class
        """

        if name is None:
            name = obj_func.__name__

        super().__init__(input_size, opt, low_lim, up_lim, name)

        self.obj_func = obj_func
    
    def objective(self, vector):
        return self.obj_func(vector)
    
    def random_solution(self):
        return np.random.uniform(self.low_lim, self.up_lim, self.input_size)
    
    def repair_solution(self, vector):
        return np.clip(vector, self.low_lim, self.up_lim)
