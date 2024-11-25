from __future__ import annotations
from typing import Any
from ..Initializer import Initializer
from ..Encoding import Encoding


class InitializerFromLambda(Initializer):
    """
    Initializer that generates individuals with vectors following an user-defined distribution.

    Parameters
    ----------
    generator: callable
        Function that samples an user-defined probability distribution to generate individuals.
    pop_size: int, optional
        Number of individuals to be generated.
    encoding: Encoding, optional
        Encoding that will be passed to each individual.
    """

    def __init__(self, generator: callable, pop_size: int = 1, encoding: Encoding = None):
        self.generator = generator

        super().__init__(pop_size, encoding)

    def generate_random(self) -> Any:
        return self.generator()
