""" 
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from .operator_functions.utils import OperatorVectorDef
from .operator_functions.mutation import (
    ProbDist,
    xor_mask,
    rand_noise,
    mutate_noise,
    mutate_sample,
    rand_sample,
    mutate_1_sigma,
    mutate_n_sigmas,
    sample_1_sigma
)
from ..operator import OperatorFromLambda

mutation_ops_map = {
    "xor":                  OperatorVectorDef(xor_mask),
    "byte_xor":             OperatorVectorDef(xor_mask, forced_params={"BinRep": "byte"}),
    "int_xor":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "int"}),
    "bit_xor":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "bin"}),
    "bitflip":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "bin"}),

    "gauss":                OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal":               OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian":             OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_noise":       OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussmut":             OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal_mutation":      OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_mutation":    OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),

    "laplace":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplacemut":           OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),

    "cauchy":               OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_noise":         OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchymut":            OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_mutation":      OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),

    "uniform":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniformmut":           OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),

    "poisson":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poissonmut":           OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),

    "bernoulli":            OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_noise":      OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoullimut":         OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_mutation":   OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip":             OperatorVectorDef(rand_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip_mutation":    OperatorVectorDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflipmut":          OperatorVectorDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),

    "mutnoise":             OperatorVectorDef(mutate_noise),
    "noise_mutation":       OperatorVectorDef(mutate_noise),
    "additive_noise_mutation": OperatorVectorDef(mutate_noise),

    "mutsample":            OperatorVectorDef(mutate_sample),
    "sampling_mutation":    OperatorVectorDef(mutate_sample),
    "replacement_mutation": OperatorVectorDef(mutate_sample),

    "randnoise":            OperatorVectorDef(rand_noise),
    "random_noise":         OperatorVectorDef(rand_noise),
    "additive_noise":       OperatorVectorDef(rand_noise),

    "randsample":           OperatorVectorDef(rand_sample),
    "random_sampling":      OperatorVectorDef(rand_sample),
    "regenerate":           OperatorVectorDef(rand_sample),

    "mutate1sigma":         OperatorVectorDef(mutate_1_sigma),
    "mutatensigmas":        OperatorVectorDef(mutate_n_sigmas),
    "samplesigma":          OperatorVectorDef(sample_1_sigma),
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
