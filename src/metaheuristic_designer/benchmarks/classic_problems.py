from typing import Iterable
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from ..objective_function import ObjectiveFunc
from ..utils import MatrixLike, VectorLike

__all__ = ["ThreeSAT", "BinKnapsack", "MaxClique", "TSP"]


class ThreeSAT(ObjectiveFunc):
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
            raise ValueError("The clauses must be represented as an array of size (n_clauses, 3).")

        self.clauses = clauses
        self.n_vars = np.abs(clauses).max()

        super().__init__(self.n_vars, lower_bound=0, upper_bound=1, name="3-SAT")

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


class BinKnapsack(ObjectiveFunc):
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
            raise ValueError("The value vector must have the same dimension as the cost vector.")

        self.cost = cost
        self.value = value
        self.max_weight = max_weight
        super().__init__(cost.size, lower_bound=0, upper_bound=1, mode="max", name="0-1 Knapsack Problem")

    def objective(self, solution):
        """
        Calculates the total value of the selection of elements. If the weight is higher
        than the maximum weight, the value is replaced by the negative weight of the elements.

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


class MaxClique(ObjectiveFunc):
    """
    This is the Maximum clique problem which consists on finding the size of the largest
    subgraph that has all its nodes interconnected (a clique).

    Parameters
    ----------
    adjacency_matrix: ndarray
        The adjacency matrix of the graph.
    """

    def __init__(self, adjacency_matrix):
        self.adj_mat = adjacency_matrix
        super().__init__(adjacency_matrix.shape[0], lower_bound=0, upper_bound=adjacency_matrix.shape[0], name="Max Clique")

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
        while n_cliques < self.dimension and is_clique:
            for i in range(1, n_cliques):
                idx_i = solution[i]
                idx_j = solution[n_cliques]
                is_clique = is_clique and self.adj_mat[idx_i, idx_j] != 0

            if is_clique:
                n_cliques += 1

        return n_cliques


class TSP(ObjectiveFunc):
    def __init__(self, adjacency_matrix: MatrixLike, name: str = None, mode: str = "min"):
        if name is None:
            name = "TSP"

        self.adjacency_matrix = adjacency_matrix
        n_nodes = adjacency_matrix.shape[0]
        super().__init__(dimension=n_nodes, lower_bound=0, upper_bound=n_nodes - 1, name=name, mode=mode, vectorized=True)

    @classmethod
    def from_csv(cls, problem_path: Path, name: str = None, mode: str = "min"):
        """
        Constructs the objective function from a .csv file.

        Parameters
        ----------
        problem_path : Path
            Path to the .csv file containing the weights of each edge.
            The expected format of the file is a table with 3 columns: Edge 1, Edge 2, Weights
        name : str, optional
            Name to use when showing the user which function is being optimized, by default None
        mode : str, optional
            Optimization mode to use, by default "min"

        Returns
        -------
        An object of type TSP (Objective function on vectors)
        """

        graph_df = pd.read_csv(problem_path)

        n_nodes = max(graph_df.iloc[:, 0].max(), graph_df.iloc[:, 1].max()) + 1

        upper_bound_weights = graph_df.iloc[:, 2].max() * (n_nodes + 1)
        adjacency_matrix = np.full((n_nodes, n_nodes), upper_bound_weights)
        for _, row in graph_df.iterrows():
            in_node, out_node, w = row["Edge1"].astype(int), row["Edge2"].astype(int), row["Weight"]
            adjacency_matrix[in_node, out_node] = w
            adjacency_matrix[out_node, in_node] = w

        np.fill_diagonal(adjacency_matrix, 0)

        return cls(adjacency_matrix, name=name, mode=mode)

    def objective(self, solutions: MatrixLike) -> VectorLike:
        edge_costs = self.adjacency_matrix[solutions[:, :-1], solutions[:, 1:]]

        objective_vector = edge_costs.sum(axis=1) + self.adjacency_matrix[solutions[:, -1], solutions[:, 0]]

        return objective_vector
