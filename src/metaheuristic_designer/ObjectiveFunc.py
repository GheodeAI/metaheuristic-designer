from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from .initializers import UniformVectorInitializer


class ObjectiveFunc(ABC):
    """
    Abstract Fitness function class.

    For each problem a new class will inherit from this one
    and implement the fitness function, random solution generation,
    mutation function and crossing of solutions.

    Parameters
    ----------
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(self, mode: str = "max", name: str = "some function"):
        """
        Constructor for the ObjectiveFunc class
        """

        self.name = name
        self.counter = 0
        self.factor = 1

        self.mode = mode
        if mode not in ["max", "min"]:
            raise ValueError('Optimization objective (mode) must be "min" or "max".')

        if self.mode == "min":
            self.factor = -1

    def __call__(self, indiv: Individual, adjusted: bool = True) -> float:
        """
        Shorthand for executing the objective function on a vector.
        """

        return self.fitness(indiv, adjusted)

    def fitness(self, indiv: Individual, adjusted: bool = True) -> float:
        """
        Returns the value of the objective function given an individual.
        If the fitness is adjusted, the sign will be switched for minimization problems
        and a penalty will be applied if the solution violates any restriction.

        Parameters
        ----------
        indiv: Individual
            The individual for which the fitness will be calculated.
        adjusted: bool, optional
            Whether to adjust the fitness value or not.

        Returns
        -------
        fitness: float
            Fitness value of the individual.
        """

        self.counter += 1
        solution = indiv.encoding.decode(indiv.genotype)
        value = self.objective(solution)

        if adjusted:
            value = self.factor * (value - self.penalize(solution))

        return value

    @abstractmethod
    def objective(self, solution: Any) -> float:
        """
        Implementation of the objective function.

        Parameters
        ----------
        solution: Any
            The solution for which the fitness will be calculated.

        Returns
        -------
        objective_value: float
            Value of the objective function given a solution.
        """

    @abstractmethod
    def repair_solution(self, solution: Any) -> Any:
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.

        Parameters
        ----------
        solution: Any
            A solution that could be violating the restrictions of the problem.

        Returns
        -------
        repaired_solution: Any
            A modified version of the solution passed that satisfies the restrictions of the problem.
        """

        return vector

    def repair_speed(self, speed: ndarray) -> ndarray:
        """
        Transforms an invalid vector into one that satisfies the restrictions of the problem.

        Parameters
        ----------
        speed: ndarray
            A speed vector that could be violating the restrictions of the problem.

        Returns
        -------
        repaired_speed: ndarray
            A modified version of the speed vector passed that satisfies the restrictions of the problem.
        """

        result = None
        if speed is not None:
            result = self.repair_solution(speed)
        return result

    def penalize(self, solution: Any) -> float:
        """
        Gives a penalization to the fitness value of an individual if it violates any constraints propotional
        to how far it is to a viable solution.

        If not implemented always returns 0.

        Parameters
        ----------
        solution: Any
            A solution that could be violating the restrictions of the problem.

        Returns
        -------
        penalty: float
            The penalty associated to the degree that the solution violates the restrictions of the problem.
        """

        return 0


class ObjectiveVectorFunc(ObjectiveFunc):
    """
    Objective function that accepts vectors as an input.

    Parameters
    ----------
    vecsize: int
        The dimension of the vectors accepted by the objective function.
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    low_lim: float, optional
        Lower limit restriction for the vectors.
    up_lim: float, optional
        Upper limit restriction for the vectors.
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(
        self,
        vecsize: int,
        mode: str = "max",
        low_lim: float = -100,
        up_lim: float = 100,
        name: str = "some function",
    ):
        """
        Constructor for the ObjectiveVectorFunc class
        """

        super().__init__(mode, name)

        self.vecsize = vecsize
        self.low_lim = low_lim
        self.up_lim = up_lim

    def repair_solution(self, vector: ndarray) -> ndarray:
        return np.clip(vector, self.low_lim, self.up_lim)


class ObjectiveFromLambda(ObjectiveVectorFunc):
    """
    Objective function that accepts vectors as an input defined with a callable object.

    Parameters
    ----------
    obj_func: callable
        Objective function as a callable object.
    vecsize: int
        The dimension of the vectors accepted by the objective function.
    mode: str, optional
        Whether to maximize or minimize the function (using the string 'max' or 'min').
    low_lim: float, optional
        Lower limit restriction for the vectors.
    up_lim: float, optional
        Upper limit restriction for the vectors.
    name: str, optional
        The name that will be displayed to represent this function.
    """

    def __init__(
        self,
        obj_func: callable,
        vecsize: int,
        mode: str = "max",
        low_lim: float = -100,
        up_lim: float = 100,
        name: str = "some function",
    ):
        """
        Constructor for the ObjectiveFromLambda class
        """

        if name is None:
            name = obj_func.__name__

        super().__init__(vecsize, mode, low_lim, up_lim, name)

        self.obj_func = obj_func

    def objective(self, vector):
        return self.obj_func(vector)

    def repair_solution(self, vector):
        return np.clip(vector, self.low_lim, self.up_lim)
