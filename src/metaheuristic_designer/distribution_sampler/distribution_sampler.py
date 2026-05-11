
from ..initializer import Initializer
from .probability_distributions import Distribution
from ..parametrizable_mixin import ParametrizableMixin


class DistributionSampler(ParametrizableMixin, Initializer):
    def __init__(self, dimension, sample_size, distribution: Distribution, encoding: Optional[Encoding] = None, random_state: Optional[RNGLike] = None):
        super().__init__(dimension=dimension, population_size=sample_size,)
        self.distribution = distribution
    
    def estimate_parameters(self):
        pass


    