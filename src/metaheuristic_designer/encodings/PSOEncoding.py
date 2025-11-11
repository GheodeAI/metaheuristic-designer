from __future__ import annotations
import numpy as np
from ..Encoding import ExtendedEncoding


class PSOEncoding(ExtendedEncoding):
    """
    Encoding used to implement the PSO algorithm.
    """
    def __init__(self, vecsize, base_encoding):
        params = {
            "speed": vecsize,
        }

        super().__init__(self, vecsize, named_params, base_encoding)

