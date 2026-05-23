"""
Base operator for algorithms that split the genotype into solution and parameters.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..initializer import Initializer
from ..population import Population
from ..operator import Operator
from ..encodings import ParameterExtendingEncoding
from .masked_operator import MaskedOperator


class ExtendedOperator(Operator):
    """Operator that handles a genotype split into solution and extra parameters.

    A mask is built from the encoding to separate the solution part
    from the parameter blocks.  The solution is processed by
    `base_operator`, while each parameter block can be mutated/adapted
    by its own operator.

    Parameters
    ----------
    base_operator : Operator
        Operator applied to the solution part.
    param_operators : dict
        Mapping from parameter names to their mutation operators.
    encoding : ParameterExtendingEncoding
        The encoding that defines the genotype layout.
    name : str, optional
        Display name; defaults to the base operator's name.
    **kwargs
        Forwarded to :class:`Operator`.
    """

    def __init__(self, base_operator: Operator, param_operators: dict, encoding: ParameterExtendingEncoding, name: str = None, **kwargs):
        if not isinstance(encoding, ParameterExtendingEncoding):
            raise TypeError("The encoding must inherit from ParameterExtendingEncoding.")

        if name is None:
            name = f"{base_operator.name}"

        mask = np.zeros(encoding.dimension + encoding.nparams)

        counter = encoding.dimension
        for idx, (_, param_num) in enumerate(encoding.param_sizes):
            mask[counter : counter + param_num] = idx + 1
            counter = counter + param_num

        operator_list = [base_operator] + [param_operators[param_name] for idx, (param_name, _) in enumerate(encoding.param_sizes)]

        self.main_operator = MaskedOperator(operator_list, mask=mask)
        self.mask = mask

        self.base_operator = base_operator
        self.param_operators = param_operators
        self.param_encoding = encoding

        super().__init__(name=name, encoding=encoding, **kwargs)

    def gather_params(self) -> dict:
        """Collect parameters from the base operator and all parameter operators.

        Returns
        -------
        dict
            Flat dictionary with dotted keys.
        """

        all_params = self.get_params()
        for op in self.param_operators.values():
            all_params.update(op.gather_params())

        return all_params

    def evolve(self, population: Population, initializer: Optional[Initializer] = None) -> Population:
        """Apply the main masked operator (solution + parameter mutations).

        Parameters
        ----------
        population : Population
            The current population.
        initializer : Initializer, optional
            The population initializer.

        Returns
        -------
        Population
            The evolved population.
        """

        return self.main_operator.evolve(population, initializer=initializer)

    def update(self, progress: float):
        """Update schedulable parameters and propagate to sub-operators.

        Parameters
        ----------
        progress : float
            Current progress of the algorithm (0-1).
        """
        super().update(progress)

        self.base_operator.update(progress)

        for _, op in self.param_operators.items():
            if isinstance(op, Operator):
                op.update(progress)
