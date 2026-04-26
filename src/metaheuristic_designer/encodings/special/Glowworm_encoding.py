from __future__ import annotations
from .. import ParameterExtendingEncoding


class GlowwormEncoding(ParameterExtendingEncoding):
    """
    Encoding used to implement the GWO algorithm.
    """

    def __init__(self, vecsize, base_encoding):
        named_params = [("luciferin", vecsize)]

        super().__init__(vecsize, named_params, base_encoding)
