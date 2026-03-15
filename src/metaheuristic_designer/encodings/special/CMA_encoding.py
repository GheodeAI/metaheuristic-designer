from __future__ import annotations
import numpy as np
from .. import ParameterExtendingEncoding


class CMAEncoding(ParameterExtendingEncoding):
    """
    Decoder used to implement the CMA-ES algorithm.
    """
    def __init__(self, vecsize, base_encoding):
        named_params = [
            ("mean", vecsize),
            ("cov", 0.5*(vecsize**2)-vecsize),
            ("F", 1)
        ]

        super().__init__(vecsize, named_params, base_encoding)
