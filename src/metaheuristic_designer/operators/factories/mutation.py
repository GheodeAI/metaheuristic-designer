"""
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from ..operator_functions.utils import OperatorVectorDef
from ..operator_functions.mutation import (
    ProbDist,
    xor_mask,
    rand_noise,
    mutate_noise,
    mutate_sample,
    rand_sample,
    mutate_1_sigma,
    mutate_n_sigmas,
    sample_1_sigma,
)
from ...operator import OperatorFromLambda

# fmt: off
mutation_ops_map = {
    # XOR and bitflip mutation
    "xor":                  OperatorVectorDef(xor_mask),
    "byte_xor":             OperatorVectorDef(xor_mask, forced_params={"BinRep": "byte"}),
    "int_xor":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "int"}),
    "bit_xor":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "bin"}),
    "bitflip":              OperatorVectorDef(xor_mask, forced_params={"BinRep": "bin"}),

    # Gaussian distribution
    "gauss":                OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal":               OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian":             OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gauss_noise":          OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal_noise":         OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_noise":       OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gauss_mut":            OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal_mutation":      OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_mutation":    OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),

    # Laplace distribution
    "laplace":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_mut":          OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),

    # Cauchy distribution
    "cauchy":               OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_noise":         OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_mut":           OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_mutation":      OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),

    # Uniform distribution
    "uniform":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_mut":          OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),

    # Poisson distribution
    "poisson":              OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_noise":        OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_mut":          OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_mutation":     OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),

    # Bernoulli distribution
    "bernoulli":            OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_noise":      OperatorVectorDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_mut":        OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_mutation":   OperatorVectorDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip":             OperatorVectorDef(rand_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip_mut":         OperatorVectorDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip_mutation":    OperatorVectorDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),

    # Additive noise mutation
    "mutnoise":             OperatorVectorDef(mutate_noise),
    "noise_mutation":       OperatorVectorDef(mutate_noise),
    "additive_noise_mutation": OperatorVectorDef(mutate_noise),

    # Resampling mutation
    "mutsample":            OperatorVectorDef(mutate_sample),
    "sampling_mutation":    OperatorVectorDef(mutate_sample),
    "replacement_mutation": OperatorVectorDef(mutate_sample),
    
    # Full additive noise
    "randnoise":            OperatorVectorDef(rand_noise),
    "random_noise":         OperatorVectorDef(rand_noise),
    "additive_noise":       OperatorVectorDef(rand_noise),
    "full_additive_noise":  OperatorVectorDef(rand_noise),
    "full_mutation":        OperatorVectorDef(rand_noise),

    # Full resampling
    "randsample":           OperatorVectorDef(rand_sample),
    "random_sampling":      OperatorVectorDef(rand_sample),
    "regenerate":           OperatorVectorDef(rand_sample),
    "full_resampling":      OperatorVectorDef(rand_sample),
    "full_random_sampling": OperatorVectorDef(rand_sample),

    # Adaptative operators
    "mutate1sigma":         OperatorVectorDef(mutate_1_sigma),
    "mutate_1_sigma":         OperatorVectorDef(mutate_1_sigma),
    "mutatensigmas":        OperatorVectorDef(mutate_n_sigmas),
    "mutate_n_sigmas":        OperatorVectorDef(mutate_n_sigmas),
    "sample1sigma":          OperatorVectorDef(sample_1_sigma),
    "sample_1_sigma":          OperatorVectorDef(sample_1_sigma),
}
# fmt: on


def create_mutation_operator(method, encoding=None, name=None, **kwargs):
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

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=mutation_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=True, **kwargs)
