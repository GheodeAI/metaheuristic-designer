import logging

logger = logging.getLogger(__name__)

from .experiment_runner import run_experiment

try:
    from . import external_wrappers
    from .external_wrappers import *
except ImportError as e:
    logger.error("Tried to load analysis package, but found missing dependencies: %s", e)
    print("Tried to load analysis package, but found missing dependencies:", e)
    ioh_present = False
