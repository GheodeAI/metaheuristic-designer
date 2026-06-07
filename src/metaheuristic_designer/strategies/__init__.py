"""
Built-in search strategy implementations.
"""

from ..search_strategy import SearchStrategy

from .no_search import NoSearch
from .single_solution_strategy import SingleSolutionStrategy
from .population_based_strategy import PopulationBasedStrategy
from .shuffled_population_strategy import ShuffledPopulationStrategy
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
    "ShuffledPopulationStrategy",
    "ShuffledPopulationStrategy",
    "EDAStrategy",
    *classic.__all__,
    *swarm.__all__,
    *EDA.__all__,
    *bayesian_optimization.__all__,
    *hybrid.__all__,
]
