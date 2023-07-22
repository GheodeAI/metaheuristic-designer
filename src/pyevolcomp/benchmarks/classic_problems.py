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


class Bin_Knapsack_problem(ObjectiveVectorFunc):
    def __init__(self, cost, value, max_weight):
        cost = np.asarray(cost)
        value = np.asarray(value)
        if cost.size != value.size:
            raise ValueError("The value vector must have the same dimension as the cost vector.")
        
        self.cost = cost
        self.value = value
        self.max_weight = max_weight
        super().__init__(cost.size, mode="max", name="0-1 Knapsack Problem")

    def objective(self, solution):
        valid = np.inner(solution, self.cost) < self.max_weight
        
        if valid:
            result = np.inner(solution, self.value)
        else:
            result = -np.inner(solution, self.cost)
        
        return result
    
    def repair_solution(self, solution):
        return (np.round(solution) != 0).astype(int)
    
    # def penalize(self, solution):
    #     valid_int = (np.inner(solution, self.cost) > self.max_weight).astype(int)
    #     return valid_int*max(self.value)*len(self.value)
        