import numpy as np
from numba import jit
from ..ObjectiveFunc import ObjectiveVectorFunc


class Three_SAT(ObjectiveVectorFunc):
    def __init__(self):
        super().__init__(1, name="3-SAT")

    def objective(self, solution):
        """
        Not implemented.
        """
    
    def repair_solution(self, solution):
        """
        Not implemented.
        """


class Clique(ObjectiveVectorFunc):
    def __init__(self):
        super().__init__(1, name="Clique")

    def objective(self, solution):
        """
        Not implemented.
        """
    
    def repair_solution(self, solution):
        """
        Not implemented.
        """


class TSP(ObjectiveVectorFunc):
    def __init__(self):
        super().__init__(1, name="TSP")

    def objective(self, solution):
        """
        Not implemented.
        """
    
    def repair_solution(self, solution):
        """
        Not implemented.
        """


class Knapsack_problem(ObjectiveVectorFunc):
    def __init__(self):
        super().__init__(1, name="Knapsack Problem")

    def objective(self, solution):
        """
        Not implemented.
        """
    
    def repair_solution(self, solution):
        """
        Not implemented.
        """