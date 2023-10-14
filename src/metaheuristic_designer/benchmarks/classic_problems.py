import numpy as np
import warnings
from ..ObjectiveFunc import ObjectiveVectorFunc


class ThreeSAT(ObjectiveVectorFunc):
    """
    This is the 3-SAT problem that consist in finding if a logical expression
    given in 3CNF (conjunctive normal form with 3 variables per clause) is satisfiable,
    in other words, if there is a combination of boolean variables that makes it true.

    Format based on https://www.cs.ubc.ca/%7Ehoos/SATLIB/benchm.html

    Parameters
    ----------
    clauses: ndarray
        A representation of the clauses that defines the logical expression, this will be a matrix
        of size (n,3), where each component is the index (starting at 1) of the variable in this clause,
        negation is represented as negative numbers.

        For example, the expression (¬a ∨ b ∨ ¬c) ∧ (a ∨ ¬b ∨ d) is represented as [[-1, 2, -3], [1, -2, 4]]
    """

    def __init__(self, clauses):
        if not isinstance(clauses, np.ndarray) or clauses.shape[1] != 3:
            raise ValueError(
                "The caluses must be represented as an array of size (n_clauses, 3)."
            )

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
            warnings.warn(
                "The number of clauses in the file was incorrect.", stacklevel=2
            )

        if np.abs(clauses_arr).max() != n_vars:
            warnings.warn(
                "The number of variables in the file was incorrect.", stacklevel=2
            )

        return ThreeSAT(clauses_arr)

    def objective(self, solution):
        """
        Calculates the percentage of clauses satisfied in the logical expression.

        Parameters
        ----------
        solution: ndarray
            A binary vector representing the value of each binary variable.

        Returns
        -------
        perc_satisfied:
            The percentage of clauses satisfied with this assignment of variables.
        """

        n_satisfied = 0
        for clause in self.clauses:
            bool_vals = solution[np.abs(clause) - 1].astype(bool)
            satisfied = np.logical_xor(bool_vals, clause < 0)  # Flip if negative
            n_satisfied += np.any(satisfied).astype(int)

        return n_satisfied / self.clauses.shape[0]


class BinKnapsack(ObjectiveVectorFunc):
    """
    This is the 0-1 Knapsack problem that consist in choosing from set of elements
    which have a certain cost and value to maximize the value without reaching a weight threshold.

    Parameters
    ----------
    cost: ndarray
        The cost associated to each of the elements.
    value: ndarray
        The value associated to each of the elements.
    max_weight: float
        The maximum weight.
    """

    def __init__(self, cost, value, max_weight):
        cost = np.asarray(cost)
        value = np.asarray(value)
        if cost.size != value.size:
            raise ValueError(
                "The value vector must have the same dimension as the cost vector."
            )

        self.cost = cost
        self.value = value
        self.max_weight = max_weight
        super().__init__(cost.size, mode="max", name="0-1 Knapsack Problem")

    def objective(self, solution):
        """
        Calculates the total value of the selection of elements. If the weight is higher
        than the maxmimum weight, the value is replaced by the negative weight of the elements.

        Parameters
        ----------
        solution: ndarray
            A binary vector deciding whether to choose or not each element.

        Returns
        -------
        value: float
            The total value of the objects.
        """

        valid = np.inner(solution, self.cost) < self.max_weight

        if valid:
            result = np.inner(solution, self.value)
        else:
            result = -np.inner(solution, self.cost)

        return result

    def repair_solution(self, solution):
        return (np.round(solution) != 0).astype(int)


class MaxClique(ObjectiveVectorFunc):
    """
    This is the Maximum clique problem which consists on finding the size of the largest
    subgraph that has all its nodes interconected (a clique).

    Parameters
    ----------
    adjacency_matrix: ndarray
        The adjacency matrix of the graph.
    """

    def __init__(self, adjacency_matrix):
        self.adj_mat = adjacency_matrix
        super().__init__(adjacency_matrix.shape[0], name="Max Clique")

    def objective(self, solution):
        """
        Parameters
        ----------
        solution: ndarray
            A sequence of nodes that will be read from left to right, starting from the
            first node that creates a clique of size 0, checking if adding a new node from
            the sequence produces a clique of a larger size. If this is not the case the
            algorithm ends.

        Returns
        -------
        clique_size: int
            The size of the clique generated with this sequence
        """

        max_clique_size = 0

        n_cliques = 1
        is_clique = True
        while n_cliques < self.vecsize and is_clique:
            for i in range(1, n_cliques):
                idx_i = solution[i]
                idx_j = solution[n_cliques]
                is_clique = is_clique and self.adj_mat[idx_i, idx_j] != 0

            if is_clique:
                n_cliques += 1

        return n_cliques


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
