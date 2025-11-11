from __future__ import annotations
import numpy as np
from ..Encoding import ExtendedEncoding


class GlowWormEncoding(ExtraEncoding):
    """
    Encoding used to implement the GWO algorithm.
    """
    def __init__(self, vecsize, base_encoding):
        named_params = {
            "luciferin": vecsize,
        }

        super().__init__(self, vecsize, named_params, base_encoding)

