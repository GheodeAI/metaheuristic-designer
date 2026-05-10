from __future__ import annotations
import logging
from pickle import PicklingError
from typing import TYPE_CHECKING, Optional
import cloudpickle
import os
import time
from .history_tracker import HistoryTracker
from .reporters import create_reporter
from .reporter import Reporter

if TYPE_CHECKING:
    from .algorithm import Algorithm

logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, checkpoint_file: str, iteration_frequency: Optional[int] = None, time_frequency: Optional[float] = None):
        self.checkpoint_file = checkpoint_file

        if iteration_frequency is None and time_frequency is None:
            logger.warning("Checkpointing frequency not configured. No checkpoints will happen.")

        self.iteration_frequency = iteration_frequency
        self.time_frequency = time_frequency
        self.timer = time.time()

    def restart(self):
        self.timer = time.time()

    def checkpoint(self, algorithm: Algorithm):
        iterations = algorithm.stopping_condition.iterations

        saving_iteration = self.iteration_frequency is not None and iterations % self.iteration_frequency == 0 and iterations > 0
        saving_time = self.time_frequency is not None and time.time() - self.timer > self.time_frequency
        if not (saving_iteration or saving_time):
            return

        if saving_time:
            self.timer = time.time()

        self.save(algorithm)

    def save(self, algorithm: Algorithm):
        """_summary_

        Parameters
        ----------
        algorithm : Algorithm
            _description_
        file_name : str, optional
            _description_, by default "checkpoint.pkl"
        """

        # Temporarily remove problematic components for serialization.
        reporter = algorithm.reporter
        history_tracker = algorithm.history_tracker
        is_parallel = algorithm.parallel
        checkpointer = algorithm.checkpointer
        algorithm.reporter = None
        algorithm.parallel = False
        algorithm.checkpointer = None

        # Store checkpoint to a temp file without overwriting the previous one yet
        try:
            tmp_file = self.checkpoint_file + ".tmp"
            with open(tmp_file, "wb") as f:
                cloudpickle.dump(algorithm, f, protocol=5)
            # Once we know the checkpoint has finished writing we can replace the preivous one.
            os.replace(tmp_file, self.checkpoint_file)
        except (OSError, PermissionError, PicklingError, TypeError, MemoryError) as e:
            logger.error("Failed to save checkpoint: %s", e)
        finally:
            # Restore dropped attributes
            algorithm.reporter = reporter
            algorithm.parallel = is_parallel
            algorithm.checkpointer = checkpointer

    def load(
        self, file_name: str = None, reporter: Reporter | str = "silent", parallel: bool = False
    ) -> Algorithm:
        if file_name is None:
            file_name = self.checkpoint_file

        with open(file_name, "rb") as f:
            algorithm: Algorithm = cloudpickle.load(f)

        if isinstance(reporter, str):
            reporter = create_reporter(reporter)
        algorithm.reporter = reporter
        algorithm.checkpointer = self
        algorithm.parallel = parallel

        return algorithm
