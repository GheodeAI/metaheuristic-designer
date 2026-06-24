"""
Operator interfaces and base implementations.
"""

from ..operator import Operator, OperatorFromLambda, NullOperator

from .composite_operator import CompositeOperator
from .branch_operator import BranchOperator
from .masked_operator import MaskedOperator
from .extended_operator import ExtendedOperator
from .adaptive_operator import AdaptiveOperator
from .BO_operator import BOOperator

from .operator_functions import *
from . import operator_functions

from .factories import *
from . import factories

__all__ = [
    "NullOperator",
    "Operator",
    "OperatorFromLambda",
    "AdaptiveOperator",
    "BOOperator",
    "BranchOperator",
    "CompositeOperator",
    "ExtendedOperator",
    "MaskedOperator",
    *operator_functions.__all__,
    *factories.__all__,
]
