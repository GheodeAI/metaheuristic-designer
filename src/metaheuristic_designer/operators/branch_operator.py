"""
Operator that randomly applies one operator from a list to each individual.
"""

from __future__ import annotations
from typing import Iterable, Optional
import enum
from enum import Enum
from copy import copy
import numpy as np

from ..encoding import Encoding
from ..initializer import Initializer
from ..population import Population
from ..utils import RNGLike
from ..operator import Operator


class BranchOpMethods(Enum):
    RANDOM = enum.auto()
    PICK = enum.auto()

    @staticmethod
    def from_str(str_input: str) -> BranchOpMethods:
        str_input = str_input.lower()

        if str_input not in branch_ops_map:
            raise ValueError(f'Operator on operators "{str_input}" not defined')

        return branch_ops_map[str_input]


branch_ops_map = {"random": BranchOpMethods.RANDOM, "rand": BranchOpMethods.RANDOM, "pick": BranchOpMethods.PICK, "choose": BranchOpMethods.PICK}


class BranchOperator(Operator):
    """Operator that stochastically selects among several operators.

    For each individual, one operator from `op_list` is chosen
    according to the configured method (random with given probability,
    or manually picked).  This allows e.g. applying mutation with a
    certain probability while leaving the rest untouched.

    Parameters
    ----------
    op_list : list of Operator
        The candidate operators.
    method : str, optional
        Branching method, ``"random"`` or ``"pick"`` (default ``"random"``).
    name : str, optional
        Display name; defaults to ``"method(op_names)"``.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random number generator.
    idx : int, optional
        Index of the operator to use when method is ``"pick"`` (default -1).
    p : float, optional
        Probability of selecting the first operator (default 0.5).
        The second operator (usually :class:`NullOperator`) gets
        probability ``1 - p``.
    **kwargs
        Additional keyword arguments stored as schedulable parameters.
    """

    def __init__(
        self,
        op_list: Iterable[Operator],
        method: str = None,
        name: str = None,
        encoding: Optional[Encoding] = None,
        rng: Optional[RNGLike] = None,
        idx: int = -1,
        p: float = 0.5,
        **kwargs,
    ):
        self.op_list = op_list

        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"{method}({joined_names})"

        super().__init__(name=name, encoding=encoding, rng=rng, p=p, **kwargs)

        if method is None:
            self.method = BranchOpMethods.RANDOM
        else:
            self.method = BranchOpMethods.from_str(method)

        self.chosen_idx = idx
        self.weights = np.array([self.params.p, 1 - self.params.p])

    def gather_params(self) -> dict:
        """Collect parameters from this operator and all sub-operators.

        Returns
        -------
        dict
            Flat dictionary with dotted keys.
        """

        all_params = self.get_params()
        for op in self.op_list:
            all_params.update(op.gather_params())

        return all_params

    def evolve(self, population: Population) -> Population:
        """Apply a random operator to each individual according to the branch method.

        Parameters
        ----------
        population : Population
            The current population.
        initializer : Initializer, optional
            The population initializer.

        Returns
        -------
        Population
            The modified population.
        """

        new_population = copy(population)

        if self.method == BranchOpMethods.RANDOM:
            self.chosen_idx = self.rng.choice(range(len(self.op_list)), size=(population.population_size,), replace=True, p=self.weights)

        if isinstance(self.chosen_idx, np.ndarray) and self.chosen_idx.ndim > 0:
            chosen_idx = self.chosen_idx
        else:
            chosen_idx = np.asarray([self.chosen_idx] * len(population))

        for idx, op in enumerate(self.op_list):
            split_mask = chosen_idx == idx

            if np.any(split_mask):
                split_population = population.take_selection(split_mask)
                split_population = op.evolve(split_population)
                new_population = new_population.apply_selection(split_population, split_mask)

        return new_population

    def choose_index(self, idx: int):
        """
        Manually chooses the operator to use next

        Parameters
        ----------
        idx : int
            Index of the operator in the list.
        """

        self.chosen_idx = idx

    def update(self, progress: float):
        """Update schedulable parameters and propagate to sub-operators.

        Parameters
        ----------
        progress : float
            Current progress of the algorithm (0-1).
        """

        super().update(progress)

        for op in self.op_list:
            if isinstance(op, Operator):
                op.update(progress)

        self.weights = np.array([self.params.p, 1 - self.params.p])

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")

        return data
