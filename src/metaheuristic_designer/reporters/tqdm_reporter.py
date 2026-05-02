from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from math import floor
from tqdm import tqdm
from ..reporter import Reporter

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm

logger = logging.getLogger(__name__)

class TQDMReporter(Reporter):
    def __init__(self, resolution: int = 1000):
        if not isinstance(resolution, int):
            resolution = int(resolution)
            logger.warning("Implicitly converted progress value to int.")
        self.resolution = resolution
        self.rounded_progress = 0
        self.bar_tracker = None
        
    def log_init(self, _algorithm: Algorithm):
        self.bar_tracker = tqdm(total=self.resolution)
        self.rounded_progress = 0

    def log_step(self, algorithm: Algorithm):
        clipped_progress = min(max(0, algorithm.progress), 1)
        next_rounded_progress = floor(clipped_progress * self.resolution)

        if next_rounded_progress > self.rounded_progress:
            self.bar_tracker.update(next_rounded_progress - self.rounded_progress)
            self.rounded_progress = next_rounded_progress

    def log_end(self, _algorithm: Algorithm):
        remaining = self.resolution - self.rounded_progress
        if remaining > 0:
            self.bar_tracker.update(remaining)
        self.bar_tracker.close()