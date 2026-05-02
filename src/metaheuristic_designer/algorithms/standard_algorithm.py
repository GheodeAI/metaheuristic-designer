import warnings
from ..algorithm import Algorithm

class StandardAlgorithm(Algorithm):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "StandardAlgorithm is deprecated, use Algorithm directly.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
