import json
from numbers import Integral
from enum import Enum
import numpy as np
import numpy.random


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

def check_random_state(seed):
    """Turn seed into an np.random.RandomState instance.

    Original implementation adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/validation.py
    BSD 3-Clause License, Copyright (c) 2007-2025 The scikit-learn developers.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )