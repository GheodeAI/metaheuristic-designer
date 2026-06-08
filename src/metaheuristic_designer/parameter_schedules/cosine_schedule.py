"""
Schedule that modulates a value with a cosine wave.
"""

import numpy as np
from ..schedulable_parameter import SchedulableParameter


class CosineSchedule(SchedulableParameter):
    """Schedule that models the parameter as a cosine wave in the range [0, 1].

    Parameters
    ----------
    amplitude : float
        Amplitude of the cosine wave.
    frequency : float
        Frequency of the cosine wave.
    phase : float
        Phase of the cosine wave.
    offset : float
        Offset of the cosine wave.
    """

    def __init__(self, amplitude: float = 1, frequency: float = None, phase: float = 0, offset: float = 0):
        super().__init__(rng=None)
        self.amplitude = amplitude
        self.frequency = frequency if frequency is not None else 1
        self.phase = phase
        self.offset = offset

    def evaluate(self, progress: float) -> float:
        return self.amplitude * np.cos(2 * np.pi * self.frequency * progress + self.phase) + self.offset
