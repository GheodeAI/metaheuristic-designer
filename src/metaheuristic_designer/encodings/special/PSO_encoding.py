"""
Encoding for Particle Swarm Optimisation that appends a velocity vector to the genotype.
"""

from __future__ import annotations
from typing import Optional
from ..parameter_extending_encoding import ParameterExtendingEncoding
from ...encoding import Encoding


class PSOEncoding(ParameterExtendingEncoding):
    """
    Encoding for Particle Swarm Optimisation that stores a velocity vector.

    The genotype is split into the solution vector and a velocity vector
    of the same dimension.  Both are used by the PSO operator.

    Parameters
    ----------
    dimension : int
        Number of decision variables.
    base_encoding : Encoding, optional
        Encoding applied to the solution part.  Defaults to
        :class:`DefaultEncoding`.
    """

    def __init__(self, dimension: int, base_encoding: Optional[Encoding] = None):
        named_params = [("speed", dimension)]

        super().__init__(dimension, named_params, base_encoding)
