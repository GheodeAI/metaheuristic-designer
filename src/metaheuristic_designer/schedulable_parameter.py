from abc import ABC, abstractmethod
from typing import Optional, Any
from .utils import check_random_state, RNGLike


class SchedulableParameter(ABC):
    def __init__(self, random_state: Optional[RNGLike] = None):
        self.random_state = check_random_state(random_state)

    def __call__(self, progress) -> Any:
        return self.evaluate(progress)

    @abstractmethod
    def evaluate(self, progress: float) -> Any:
        """

        Parameters
        ----------
        progress : float
            _description_
        """
