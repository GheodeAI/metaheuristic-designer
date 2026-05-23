"""
Module for checkpointing and resuming optimizations runs.
"""

from __future__ import annotations
import logging
from pickle import PicklingError
from typing import TYPE_CHECKING, Optional
import cloudpickle
import os
import time
from .reporters import create_reporter
from .reporter import Reporter

if TYPE_CHECKING:
    from .algorithm import Algorithm

logger = logging.getLogger(__name__)


class Checkpointer:
    """Periodically save and restore the state of an optimizations run.

    The checkpointer can be triggered by iteration count, elapsed
    wall-clock time, or both.  It writes the entire
    :class:`Algorithm` object to disk using ``cloudpickle``, so that
    an interrupted run can be resumed later without losing progress.

    Parameters
    ----------
    checkpoint_file : str
        Path to the file where the checkpoint will be saved (e.g.,
        ``"run.pkl"``).
    iteration_frequency : int, optional
        Save a checkpoint every *n* iterations.
    time_frequency : float, optional
        Save a checkpoint when at least this many seconds have
        elapsed since the last save.

    Notes
    -----
    At least one of *iteration_frequency* or *time_frequency* must
    be provided; otherwise the checkpointer does nothing and logs a
    warning.
    """

    def __init__(self, checkpoint_file: str, iteration_frequency: Optional[int] = None, time_frequency: Optional[float] = None):
        self.checkpoint_file = checkpoint_file

        if iteration_frequency is None and time_frequency is None:
            logger.warning("Checkpointing frequency not configured. No checkpoints will happen.")

        self.iteration_frequency = iteration_frequency
        self.time_frequency = time_frequency
        self.timer = time.time()

    def restart(self):
        """Reset the internal timer so that time-based checkpoints are measured from this moment onward."""
        self.timer = time.time()

    def checkpoint(self, algorithm: Algorithm):
        """Evaluate whether a checkpoint should be saved, and perform the save
        if necessary.

        Parameters
        ----------
        algorithm : Algorithm
            The running algorithm instance.
        """

        iterations = algorithm.stopping_condition.iterations

        saving_iteration = self.iteration_frequency is not None and iterations % self.iteration_frequency == 0 and iterations > 0
        saving_time = self.time_frequency is not None and time.time() - self.timer > self.time_frequency
        if not (saving_iteration or saving_time):
            return

        if saving_time:
            self.timer = time.time()

        self.save(algorithm)

    def save(self, algorithm: Algorithm):
        """Serialize the algorithm to disk using cloudpickle.

        A temporary file is written first and atomically moved to the
        final location, preventing corruption if the process crashes
        mid-write.  The reporter, parallel flag, and
        the checkpointer itself are temporarily removed before pickling
        to avoid serialization issues, and then restored.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm to save.
        """

        # Temporarily remove problematic components for serialization.
        reporter = algorithm.reporter
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
            # Once we know the checkpoint has finished writing we can replace the previous one.
            os.replace(tmp_file, self.checkpoint_file)
        except (OSError, PermissionError, PicklingError, TypeError, MemoryError) as e:
            logger.error("Failed to save checkpoint: %s", e)
        finally:
            # Restore dropped attributes
            algorithm.reporter = reporter
            algorithm.parallel = is_parallel
            algorithm.checkpointer = checkpointer

    def load(
        self,
        file_name: Optional[str] = None,
        reporter: Reporter | str = "silent",
        parallel: bool = False,
    ) -> Algorithm:
        """Restore a previously saved algorithm from a checkpoint file.

        Parameters
        ----------
        file_name : str, optional
            Path to the checkpoint file.  If not provided, the path
            given at construction is used.
        reporter : Reporter or str, optional
            Reporter to attach to the restored algorithm (a
            :class:`Reporter` instance or a string like ``"tqdm"``,
            ``"silent"``).  Default is ``"silent"``.
        parallel : bool, optional
            Whether parallel evaluation should be enabled after
            restoration.  Default is ``False``.

        Returns
        -------
        Algorithm
            The deserialized algorithm, ready to continue from where it
            was saved. Ensure you run the algorithm with `.resume()` so
            data is not lost.
        """

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
