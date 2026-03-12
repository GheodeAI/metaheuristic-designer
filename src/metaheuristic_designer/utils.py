import json
from enum import Enum

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Enum):
            return str(o)
        return json.JSONEncoder.default(self, o)


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
