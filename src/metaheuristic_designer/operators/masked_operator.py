"""
Operator that applies different operators to disjoint slices of the genotype.
"""

from __future__ import annotations
from typing import Iterable
from copy import copy
import numpy as np

from ..population import Population
from ..operator import Operator
from ..utils import MaskLike


class MaskedOperator(Operator):
    """Operator that partitions the genotype and applies different operators.

    A mask (integer array of length `vec_size`) specifies which operator
    (index into `op_list`) handles each gene.  This is used internally
    by :class:`ExtendedOperator` to separate the solution from auxiliary
    parameters.

    Parameters
    ----------
    op_list : list of Operator
        Operators to apply, one per mask index.
    mask : array of int
        Array of length `vec_size` assigning each gene to an operator.
    name : str, optional
        Display name; defaults to ``"Split (op_names)"``.
    \\*\\*kwargs
        Forwarded to :class:`Operator`.
    """

    def __init__(self, op_list: Iterable[Operator], mask: MaskLike, name: str = None, **kwargs):
        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"Split ({joined_names})"

        self.op_list = op_list
        super().__init__(name, mask=mask, **kwargs)

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
        """Apply the appropriate operator to each slice of the genotype.

        Parameters
        ----------
        population : Population
            The current population.

        Returns
        -------
        Population
            The modified population.
        """

        new_population = copy(population)

        # In masked_operator.py, inside the loop over op_list:
        for idx_op, op in enumerate(self.op_list):
            split_mask = self.params.mask == idx_op
            if np.any(split_mask):
                split_population = new_population.take_slice(split_mask)
                split_population = op.evolve(split_population)
                new_population = new_population.apply_slice(split_population, split_mask)

        return new_population

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

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")

        return data
