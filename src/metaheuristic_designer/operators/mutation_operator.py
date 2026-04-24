""" 
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from dataclasses import dataclass
from .operator_functions.utils import OperatorVectorDef
from .operator_functions.mutation import *
from ..operator import OperatorFromLambda

mutation_ops_map = {
    "xor": OperatorVectorDef(xor_mask),
    "flip": OperatorVectorDef(xor_mask, forced_params={"BinRep": "bin"}),
    "gauss": OperatorVectorDef(rand_noise, forced_params={"distrib": "Gauss"}),
    "laplace": OperatorVectorDef(rand_noise, forced_params={"distrib": "Laplace"}),
    "cauchy": OperatorVectorDef(rand_noise, forced_params={"distrib": "Cauchy"}),
    "uniform": OperatorVectorDef(rand_noise, forced_params={"distrib": "Uniform"}),
    "poisson": OperatorVectorDef(rand_noise, forced_params={"distrib": "Poisson"}),
    "bernoulli": OperatorVectorDef(mutate_sample, params={"N", 1}, forced_params={"distrib": "bernoulli"}),
    "mutrand": OperatorVectorDef(mutate_noise),
    "mutnoise": OperatorVectorDef(mutate_noise),
    "mutsample": OperatorVectorDef(mutate_sample),
    "randnoise": OperatorVectorDef(rand_noise),
    "randsample": OperatorVectorDef(rand_sample),
    "mutate1sigma": OperatorVectorDef(mutate_1_sigma),
    "mutatensigmas": OperatorVectorDef(mutate_n_sigmas),
    "samplesigma": OperatorVectorDef(sample_1_sigma),
}

def create_mutation_operator(method, encoding=None, **kwargs):
    """_summary_

    Parameters
    ----------
    method
        _description_
    encoding, optional
        _description_, by default None

    Returns
    -------
        _description_
    """
    
    return OperatorFromLambda(
        operator_fn=mutation_ops_map[method.lower()],
        name=method,
        vectorized=True,
        encoding=encoding,
        **kwargs
    )
