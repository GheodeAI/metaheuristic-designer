import numpy as np
import warnings
from ..ObjectiveFunc import ObjectiveVectorFunc


class Three_SAT(ObjectiveVectorFunc):
    """
    Format based on https://www.cs.ubc.ca/%7Ehoos/SATLIB/benchm.html
    """

    def __init__(self, clauses):
        if not isinstance(clauses, np.ndarray) or clauses.shape[1] != 3:
            raise ValueError("The caluses must be represented as an array of size (n_clauses, 3).")
        
        self.clauses = clauses
        self.n_vars = np.abs(clauses).max()

        super().__init__(self.n_vars, name="3-SAT")
    
    @staticmethod
    def from_cnf_file(path):
        n_vars = 0
        n_clauses = 0
        clauses = []
        with open(path, "r") as cnf_file:
            for line in cnf_file:
                line_splitted = " ".join(line.split()).split()
                if line[0] == "p":
                    n_vars = int(line_splitted[2])
                    n_clauses = int(line_splitted[3])
                    
                elif line[0] != "c" and "%" not in line and len(line_splitted) >= 3:
                    clauses.append([int(i) for i in line_splitted[:3]])
        
        clauses_arr = np.asarray(clauses)
        
        if len(clauses) != n_clauses:
            warnings.warn("The number of clauses in the file was incorrect.", stacklevel=2)

        if np.abs(clauses_arr).max() != n_vars:
            warnings.warn("The number of variables in the file was incorrect.", stacklevel=2)
        
        return Three_SAT(clauses_arr)

    def objective(self, solution):
        n_satisfied = 0
        for clause in self.clauses:
            bool_vals = solution[np.abs(clause)-1].astype(bool)
            satisfied = np.logical_xor(bool_vals, clause < 0) # Flip if negative
            n_satisfied += np.any(satisfied).astype(int)
        
        return n_satisfied/self.clauses.shape[0]


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
        