"""
Initializer for genotypes that contain both a solution and extra parameters.
"""

from __future__ import annotations
import numpy as np
from ..encodings import ParameterExtendingEncoding
from ..initializer import Initializer


class ExtendedInitializer(Initializer):
    """
    Initializer that combines a solution initializer with one or more
    parameter initializers.

    This is used with :class:`ParameterExtendingEncoding` to produce
    genotypes that store extra information (e.g., velocity for PSO,
    mutation strengths for self-adaptation).

    Parameters
    ----------
    solution_init : Initializer
        Initializer for the solution part of the genotype.
    param_init_dict : dict
        Mapping of parameter names to their corresponding initializers.
    encoding : ParameterExtendingEncoding
        The extended encoding that defines the parameter layout.
    random_state : RNGLike, optional
        Random number generator.
    """

    def __init__(self, solution_init: Initializer, param_init_dict: dict, encoding: ParameterExtendingEncoding, random_state=None):
        assert isinstance(encoding, ParameterExtendingEncoding), "An `ExtendedEncoding` instance must be used with this type of initializer"
        super().__init__(
            dimension=solution_init.dimension + encoding.nparams,
            population_size=solution_init.population_size,
            encoding=encoding,
            random_state=random_state,
        )
        self.solution_init = solution_init
        self.param_init_dict = param_init_dict
    def generate_random(self):
        """Generate a random genotype vector with solution and parameter parts.

        Returns
        -------
        ndarray
            A 1-D array with the solution followed by the extra parameters.
        """

        solution_vector = self.solution_init.generate_random()
        full_vector = np.hstack(
            [solution_vector] + [self.param_init_dict[param_name].generate_random() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector

    def generate_individual(self):
        """Generate an individual (by default identical to :meth:`generate_random`).

        Returns
        -------
        ndarray
            A 1-D array with the solution and parameter parts.
        """

        solution_vector = self.solution_init.generate_individual()
        full_vector = np.hstack(
            [solution_vector] + [self.param_init_dict[param_name].generate_individual() for param_name, _ in self.encoding.param_sizes]
        )
        return full_vector
