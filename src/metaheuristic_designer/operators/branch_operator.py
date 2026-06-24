"""
Operator that randomly applies one operator from a list to each individual.
"""

from __future__ import annotations
from typing import Iterable, Optional
from copy import copy
import numpy as np

from ..encoding import Encoding
from ..population import Population
from ..utils import RNGLike, ScalarLike, VectorLike
from ..operator import Operator


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
    random_pick : bool, optional
        Whether to pick an operator at random or by specifying an index (default True).
    name : str, optional
        Display name; defaults to ``"method(op_names)"``.
    encoding : Encoding, optional
        Encoding applied to the genotype.
    rng : RNGLike, optional
        Random number generator.
    weights: VectorLike, optional
        Weights of each operator when choosing at random.
    p : float, optional
        Probability of selecting the first operator (default 0.5).
        Only applied when ``op_list`` has length 2 and no weights are specified.
    **kwargs
        Additional keyword arguments stored as schedulable parameters.
    """

    def __init__(
        self,
        op_list: Iterable[Operator],
        random_pick: bool = True,
        name: str = None,
        encoding: Optional[Encoding] = None,
        rng: Optional[RNGLike] = None,
        weights: Optional[VectorLike] = None,
        p: float = None,
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
            name = f"Branch({joined_names})"

        self.random_pick = random_pick
        self.uses_binary_p = len(op_list) == 2 and p is not None

        super().__init__(name=name, encoding=encoding, rng=rng, p=p, weights=weights, chosen_idx=0, **kwargs)

        if self.uses_binary_p:
            weights = np.array([self.params.p, 1 - self.params.p])

        if weights is None:
            weights = np.ones(len(op_list)) / len(op_list)

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

        Returns
        -------
        Population
            The modified population.
        """

        new_population = copy(population)

        if self.random_pick:
            self.chosen_idx = self.rng.choice(range(len(self.op_list)), size=(population.population_size,), replace=True, p=self.params.weights)
        else:
            self.chosen_idx = self.params.chosen_idx

        if not isinstance(self.chosen_idx, np.ndarray) or self.chosen_idx.ndim != 1:
            self.chosen_idx = np.asarray([self.chosen_idx] * len(population))

        for idx, op in enumerate(self.op_list):
            choice_mask = self.chosen_idx == idx

            if np.any(choice_mask):
                split_population = population.take_selection(choice_mask)
                split_population = op.evolve(split_population)
                new_population = new_population.apply_selection(split_population, choice_mask)

        return new_population

    def choose_index(self, idx: VectorLike | ScalarLike):
        """
        Manually chooses the operator to use next

        Parameters
        ----------
        idx : int
            Index of the operator in the list.
        """

        self.update_kwargs(chosen_idx=idx)

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

        if self.uses_binary_p:
            weights = np.array([self.params.p, 1 - self.params.p])
            self.update_kwargs(weights=weights)

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            data["op_list"].append(op.get_state())

        return data
