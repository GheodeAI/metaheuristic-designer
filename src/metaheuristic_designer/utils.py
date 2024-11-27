import json
from enum import Enum

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Enum):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


RAND_GEN = np.random.default_rng()

def get_rng():
    """
    Get the global random number generator of the package
    """

    return RAND_GEN

def reset_seed(seed=0):
    """
    Resets the seed of the random generator.

    Parameters
    ----------
    seed: int, optional

    Returns
    -------
    RAND_GEN: RandomGenerator
        random generator
    """

    bit_gen = type(RAND_GEN.bit_generator)

    RAND_GEN.bit_generator.state = bit_gen(seed).state

    return RAND_GEN
