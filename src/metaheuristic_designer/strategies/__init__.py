from ..search_strategy import SearchStrategy

from .hill_climb import HillClimb
from .local_search import LocalSearch

from .static_population import StaticPopulation
from .variable_population import VariablePopulation

from .no_search import NoSearch

from .classic import *
from . import classic

from .swarm import *
from . import swarm

from .EDA import *
from . import EDA

from .bayesian_optimization import *
from . import bayesian_optimization

__all__ = [
    "HillClimb",
    "LocalSearch",
    "NoSearch",
    "SearchStrategy",
    "StaticPopulation",
    "VariablePopulation",
    *classic.__all__,
    *swarm.__all__,
    *EDA.__all__,
    *bayesian_optimization.__all__,
]
