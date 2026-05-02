from __future__ import annotations
from ..reporter import Reporter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaheuristic_designer.algorithm import Algorithm

class SilentReporter(Reporter):
    def log_init(self, algorithm: Algorithm): pass
    def log_step(self, algorithm: Algorithm): pass
    def log_end(self, algorithm: Algorithm): pass