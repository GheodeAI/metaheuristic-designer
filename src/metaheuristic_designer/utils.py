from enum import Enum
import numpy as np
import json


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

    BitGen = type(RAND_GEN.bit_generator)

    RAND_GEN.bit_generator.state = BitGen(seed).state

    return RAND_GEN
