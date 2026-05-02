from .benchmark_funcs import *
from . import benchmark_funcs

from .img_funcs import *
from . import img_funcs

from .classic_problems import *
from . import classic_problems

__all__ = [
    *benchmark_funcs.__all__,
    *img_funcs.__all__,
    *classic_problems.__all__
]