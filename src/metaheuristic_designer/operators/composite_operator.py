"""
Operator that applies a sequence of operators one after another.
"""

from __future__ import annotations
from typing import Iterable, Optional
from copy import copy

from metaheuristic_designer.initializer import Initializer
from metaheuristic_designer.population import Population

from ..encoding import Encoding
from ..utils import RNGLike
from ..operator import Operator


class CompositeOperator(Operator):
    """Operator that sequentially applies a list of operators.

    Each operator in `op_list` receives the population returned by
    the previous one.  This is the canonical way to chain crossover
    and mutation, or to build more complex pipelines.

    Parameters
    ----------
    op_list : list of Operator
        The operators to apply in order.
    name : str, optional
        Display name; defaults to ``"Sequence (op_names)"``.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random number generator (shared with sub-operators).
    """

    def __init__(self, op_list: Iterable[Operator], name: str = None, encoding: Optional[Encoding] = None, rng: Optional[RNGLike] = None):
        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"Sequence ({joined_names})"

        # We need to define the op_list before the constructor since it's used in the update method
        self.op_list = op_list

        super().__init__(name=name, encoding=encoding, rng=rng)

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
        """Apply each operator in sequence.

        Parameters
        ----------
        population : Population
            The current population.

        Returns
        -------
        Population
            The population after all operators have been applied.
        """

        new_population = copy(population)

        for op in self.op_list:
            new_population = op.evolve(new_population)

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
