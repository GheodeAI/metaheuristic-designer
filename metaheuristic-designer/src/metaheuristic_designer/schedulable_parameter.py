from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from .utils import check_random_state, RNGLike


class SchedulableParameter(ABC):
    def __init__(self, random_state: Optional[RNGLike] = None):
        self.random_state = check_random_state(random_state)

    def __call__(self, progress: float) -> Any:
        return self.evaluate(progress)

    @abstractmethod
    def evaluate(self, progress: float) -> Any:
        """

        Parameters
        ----------
        progress : float
            _description_
        """


class ParameterFromLambda(SchedulableParameter):
    def __init__(self, schedule_fn: Callable, random_state: Optional[RNGLike] = None):
        super().__init__()
        self.schedule_fn = schedule_fn

    def evaluate(self, progress: float) -> Any:
        return self.schedule_fn(progress)
