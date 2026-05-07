from __future__ import annotations
import numpy as np
from ..encodings import ParameterExtendingEncoding
from ..initializer import Initializer


class ExtendedInitializer(Initializer):
    def __init__(self, solution_init: Initializer, param_init_dict: dict, encoding: ParameterExtendingEncoding, random_state=None):
        assert isinstance(encoding, ParameterExtendingEncoding), "An `ExtendedEncoding` instance must be used with this type of initializer"
        super().__init__(
            dimension=solution_init.dimension + encoding.nparams, population_size=solution_init.population_size, encoding=encoding, random_state=random_state
        )
        self.solution_init = solution_init
        self.param_init_dict = param_init_dict

    def generate_random(self):
        solution_vector = self.solution_init.generate_random()
        full_vector = np.hstack(
            [solution_vector] + [self.param_init_dict[param_name].generate_random() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector

    def generate_individual(self):
        solution_vector = self.solution_init.generate_individual()
        full_vector = np.hstack(
            [solution_vector] + [self.param_init_dict[param_name].generate_individual() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector
