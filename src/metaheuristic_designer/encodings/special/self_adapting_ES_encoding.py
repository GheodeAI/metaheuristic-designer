from __future__ import annotations
from typing import Optional
from ..parameter_extending_encoding import ParameterExtendingEncoding
from ...encoding import Encoding


class SelfAdaptingESEncoding(ParameterExtendingEncoding):
    """
    Encoding used to implement self adapting ES algorithms.
    """

    def __init__(self, dimension: int, single_sigma: bool = True, base_encoding: Optional[Encoding] = None):
        self.single_sigma = single_sigma
        if single_sigma:
            named_params = [("F", 1)]
        else:
            named_params = [("F", dimension)]

        super().__init__(dimension, named_params, base_encoding)
