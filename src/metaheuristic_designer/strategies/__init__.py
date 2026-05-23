"""
Built-in search strategy implementations.
"""

from ..search_strategy import SearchStrategy

from .no_search import NoSearch
from .single_solution_strategy import SingleSolutionStrategy
from .static_population_strategy import StaticPopulationStrategy
from .variable_population_strategy import VariablePopulationStrategy
from .eda_strategy import EDAStrategy


from .classic import *
from . import classic

from .swarm import *
from . import swarm

from .EDA import *
from . import EDA

from .bayesian_optimization import *
from . import bayesian_optimization

from .hybrid import *
from . import hybrid

__all__ = [
    "NoSearch",
    "SearchStrategy",
    "SingleSolutionStrategy",
    "VariablePopulationStrategy",
    "VariablePopulationStrategy",
    "EDAStrategy",
    *classic.__all__,
    *swarm.__all__,
    *EDA.__all__,
    *bayesian_optimization.__all__,
    *hybrid.__all__,
]
