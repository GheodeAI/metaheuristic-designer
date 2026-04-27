from __future__ import annotations
import numpy as np
from .extended_operator import ExtendedOperator


class AdaptativeOperator(ExtendedOperator):
    def evolve(self, population, initializer=None):
        # Update operator parameters
        params = self.param_encoding.decode_params(population.genotype_matrix)
        self.base_operator.update_kwargs(**params)

        # Evolve population
        return super().evolve(population=population, initializer=initializer)
