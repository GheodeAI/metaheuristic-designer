from __future__ import annotations
from ..Initializer import Initializer
from ..Encoding import Encoding

class ExtendedInitializer(Initializer):
    def __init__(self, solution_init: Initializer, param_init_list: Iterable[Initializer], encoding: Encoding = None):
        self.solution_init = solution_init
        self.param_init_list = param_init_list
        super().__init__(pop_size=self.solution_init.pop_size, encoding=encoding)

    def generate_random(self):
        solution_vector = self.solution_init.generate_random()
        full_vector = np.hstack(
            [solution_vector] +
            [param_init.generate_random() for param_init in self.param_init_list]
        )
        return full_vector

    def generate_individual(self):
        solution_vector = self.solution_init.generate_individual()
        full_vector = np.hstack(
            [solution_vector] +
            [param_init.generate_individual() for param_init in self.param_init_list]
        )
        return full_vector

