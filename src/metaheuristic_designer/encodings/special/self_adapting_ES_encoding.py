"""
Encoding for self-adapting Evolution Strategies that appends mutation strength parameters.
"""

from __future__ import annotations
from typing import Optional
from ..parameter_extending_encoding import ParameterExtendingEncoding
from ...encoding import Encoding


class SelfAdaptingESEncoding(ParameterExtendingEncoding):
    """
    Encoding for self-adapting Evolution Strategies.

    Appends one or more mutation strength values (``F``) to the solution
    vector.  When ``single_sigma=True`` a single step size is shared by
    all dimensions; otherwise each dimension gets its own step size.

    Parameters
    ----------
    dimension : int
        Number of decision variables.
    single_sigma : bool, optional
        If ``True`` (default), a single ``F`` value is added.
        If ``False``, ``dimension`` values are added.
    base_encoding : Encoding, optional
        Encoding applied to the solution part.  Defaults to
        :class:`DefaultEncoding`.
    """

    def __init__(self, dimension: int, single_sigma: bool = True, base_encoding: Optional[Encoding] = None):
        self.single_sigma = single_sigma
        if single_sigma:
            named_params = [("F", 1)]
        else:
            named_params = [("F", dimension)]

        super().__init__(dimension, named_params, base_encoding)
