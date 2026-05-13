from __future__ import annotations
from typing import Optional
import numpy as np
from ..parameter_extending_encoding import ParameterExtendingEncoding
from ...encoding import Encoding


class PSOEncoding(ParameterExtendingEncoding):
    """
    Encoding used to implement the PSO algorithm.
    """

    def __init__(self, dimension: int, base_encoding: Optional[Encoding] = None):
        named_params = [("speed", dimension)]

        super().__init__(dimension, named_params, base_encoding)
