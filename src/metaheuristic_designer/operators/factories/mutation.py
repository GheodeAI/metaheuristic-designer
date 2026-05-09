"""
Implementation of generic vector operators.

Provides a factory method to generate the operator from a name.
"""

from ..operator_functions.utils import OperatorFnDef
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
    "xor":                  OperatorFnDef(xor_mask),
    "byte_xor":             OperatorFnDef(xor_mask, forced_params={"BinRep": "byte"}),
    "int_xor":              OperatorFnDef(xor_mask, forced_params={"BinRep": "int"}),
    "bit_xor":              OperatorFnDef(xor_mask, forced_params={"BinRep": "bin"}),
    "bitflip":              OperatorFnDef(xor_mask, forced_params={"BinRep": "bin"}),

    # Gaussian distribution
    "gauss":                OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal":               OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian":             OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gauss_noise":          OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal_noise":         OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_noise":       OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gauss_mut":            OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "normal_mutation":      OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),
    "gaussian_mutation":    OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.GAUSS}),

    # Laplace distribution
    "laplace":              OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_noise":        OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_mut":          OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),
    "laplace_mutation":     OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.LAPLACE}),

    # Cauchy distribution
    "cauchy":               OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_noise":         OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_mut":           OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),
    "cauchy_mutation":      OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.CAUCHY}),

    # Uniform distribution
    "uniform":              OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_noise":        OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_mut":          OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),
    "uniform_mutation":     OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.UNIFORM}),

    # Poisson distribution
    "poisson":              OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_noise":        OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_mut":          OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),
    "poisson_mutation":     OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.POISSON}),

    # Bernoulli distribution
    "bernoulli":            OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_noise":      OperatorFnDef(rand_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_mut":        OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "bernoulli_mutation":   OperatorFnDef(mutate_noise, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip":             OperatorFnDef(rand_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip_mut":         OperatorFnDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),
    "coinflip_mutation":    OperatorFnDef(mutate_sample, forced_params={"distrib": ProbDist.BERNOULLI}),

    # Additive noise mutation
    "mutnoise":             OperatorFnDef(mutate_noise),
    "noise_mutation":       OperatorFnDef(mutate_noise),
    "additive_noise_mutation": OperatorFnDef(mutate_noise),

    # Resampling mutation
    "mutsample":            OperatorFnDef(mutate_sample),
    "sampling_mutation":    OperatorFnDef(mutate_sample),
    "replacement_mutation": OperatorFnDef(mutate_sample),
    
    # Full additive noise
    "randnoise":            OperatorFnDef(rand_noise),
    "random_noise":         OperatorFnDef(rand_noise),
    "additive_noise":       OperatorFnDef(rand_noise),
    "full_additive_noise":  OperatorFnDef(rand_noise),
    "full_mutation":        OperatorFnDef(rand_noise),

    # Full resampling
    "randsample":           OperatorFnDef(rand_sample),
    "random_sampling":      OperatorFnDef(rand_sample),
    "regenerate":           OperatorFnDef(rand_sample),
    "full_resampling":      OperatorFnDef(rand_sample),
    "full_random_sampling": OperatorFnDef(rand_sample),

    # Adaptative operators
    "mutate1sigma":         OperatorFnDef(mutate_1_sigma),
    "mutate_1_sigma":         OperatorFnDef(mutate_1_sigma),
    "mutatensigmas":        OperatorFnDef(mutate_n_sigmas),
    "mutate_n_sigmas":        OperatorFnDef(mutate_n_sigmas),
    "sample1sigma":          OperatorFnDef(sample_1_sigma),
    "sample_1_sigma":          OperatorFnDef(sample_1_sigma),
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
