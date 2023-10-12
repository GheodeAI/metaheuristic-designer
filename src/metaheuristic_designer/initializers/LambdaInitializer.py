from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual
from ..encodings import DefaultEncoding


class LambdaInitializer(Initializer):
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

    def __init__(
        self, generator: callable, pop_size: int = 1, encoding: Encoding = None
    ):
        self.pop_size = pop_size
        self.generator = generator

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding

    def generate_random(self, objfunc: ObjectiveFunc) -> Individual:
        return Individual(objfunc, self.generator(), encoding=self.encoding)
