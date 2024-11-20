from __future__ import annotations
import numpy as np
from ..Operator import Operator
from ..ParamScheduler import ParamScheduler
from copy import copy
import enum
from enum import Enum
from ..utils import RAND_GEN


class OperatorFromLambda(Operator):
    """
    Operator class that applies a custom operator specified as a function.

    Parameters
    ----------
    fn: callable
        Function that will be applied when operating on an individual.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, fn: callable, params: Union[ParamScheduler, dict] = None, name: str = None, vectorized: bool = True):
        """
        Constructor for the OperatorLambda class
        """

        self.fn = fn
        self.vectorized = vectorized

        if name is None:
            name = fn.__name__

        super().__init__(params, name)

    def evolve(self, population, initializer=None):
        if self.vectorized:
            new_population = self.fn(population.genotype_set, **self.params)
        else:
            new_population = [self.evolve_single(indiv, population, intializer, global_best) for indiv in population]

        return new_population

    def evolve_single(self, indiv, population, objfunc, initializer=None, global_best=None):
        new_indiv = copy(indiv)
        others = [i for i in population if i != indiv]

        new_indiv.genotype = self.fn(indiv, others, objfunc, **self.params)

        return new_indiv
