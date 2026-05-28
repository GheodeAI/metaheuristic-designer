"""Module for schedule-dependent parameters whose value changes with progress."""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any
from .utils import check_random_state, RNGLike


class SchedulableParameter(ABC):
    """Abstract base for parameters that depend on the optimization progress.

    A schedulable parameter is a callable that receives a progress
    value between 0 and 1 and returns the parameter's value at that
    point.  Subclasses implement :meth:`evaluate`.

    Parameters
    ----------
    random_state : RNGLike, optional
        Random number generator, made available for subclasses that
        need stochastic schedules.
    """

    def __init__(self, random_state: Optional[RNGLike] = None):
        self.random_state = check_random_state(random_state)

    def __call__(self, progress: float) -> Any:
        """Shorthand for :meth:`evaluate`."""
        return self.evaluate(progress)

    @abstractmethod
    def evaluate(self, progress: float) -> Any:
        """Return the parameter value at the given progress.

        Parameters
        ----------
        progress : float
            Current progress, a number between 0 (start) and 1 (end).

        Returns
        -------
        Any
            The parameter value at this stage of the optimization.

        Notes
        -----
        The return value is **not** restricted to numbers.  You can
        return:
        * a **float** (e.g., a linearly decaying mutation strength),
        * an **int** (e.g., a discrete number of mutated components),
        * a **bool** (e.g., switching on/off a feature after a threshold),
        * a **string** (e.g., switching between strategies), or
        * any other object that the consuming component expects.

        This makes schedules suitable for changing discrete algorithm
        choices as well as continuous numerical parameters.
        """


class ParameterFromLambda(SchedulableParameter):
    """Schedulable parameter that wraps a user-supplied function.

    Parameters
    ----------
    schedule_fn : callable
        A function ``(progress) -> value`` that defines the schedule.
    random_state : RNGLike, optional
        Random number generator (passed through to the base class).
    """

    def __init__(self, schedule_fn: Callable, random_state: Optional[RNGLike] = None):
        super().__init__(random_state)
        self.schedule_fn = schedule_fn

    def evaluate(self, progress: float) -> Any:
        return self.schedule_fn(progress)
