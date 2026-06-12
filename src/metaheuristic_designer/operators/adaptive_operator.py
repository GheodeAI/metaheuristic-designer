"""
Adaptive operator that updates its parameters from the genotype.
"""

from __future__ import annotations

from ..population import Population
from .extended_operator import ExtendedOperator


class AdaptiveOperator(ExtendedOperator):
    """Operator that dynamically adapts its base operator's parameters.

    At each generation, the parameters encoded in the genotype are
    decoded and used to update the base operator before applying it
    to the population.  This enables self-adaptive algorithms (e.g.,
    Evolution Strategies with evolving mutation strengths).

    See :class:`ExtendedOperator` for constructor parameters.
    """

    def evolve(self, population: Population) -> Population:
        """Decode parameters, update the base operator, then apply it.

        Parameters
        ----------
        population : Population
            The current population (whose genotype contains the parameters).

        Returns
        -------
        Population
            The evolved population.
        """

        # Update operator parameters
        params = self.param_encoding.decode_params(population.genotype_matrix)
        self.base_operator.update_kwargs(**params)

        # Evolve population
        return super().evolve(population=population)
