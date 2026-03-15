from __future__ import annotations
import numpy as np
from ..encodings import ParameterExtendingEncoding
from ..initializer import Initializer

class ExtendedInitializer(Initializer):
    def __init__(self, solution_init: Initializer, param_init_dict: dict, encoding: ParameterExtendingEncoding):
        assert isinstance(encoding, ParameterExtendingEncoding), "An `ExtendedEncoding` instance must be used with this type of initializer"

        self.solution_init = solution_init
        self.param_init_dict = param_init_dict
        super().__init__(pop_size=self.solution_init.pop_size, encoding=encoding)

    def generate_random(self):
        solution_vector = self.solution_init.generate_random()
        full_vector = np.hstack(
            [solution_vector] +
            [self.param_init_dict[param_name].generate_random() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector

    def generate_individual(self):
        solution_vector = self.solution_init.generate_individual()
        full_vector = np.hstack(
            [solution_vector] +
            [self.param_init_dict[param_name].generate_individual() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector