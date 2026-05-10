"""
Mutation operator registry and factory.
"""

from __future__ import annotations
from typing import Optional

from ...encoding import Encoding
from ..operator_functions.utils import OperatorFnDef
from ..operator_functions.mutation import (
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
    "byte_xor":             OperatorFnDef(xor_mask, forced_params={"mode": "byte"}),
    "int_xor":              OperatorFnDef(xor_mask, forced_params={"mode": "int"}),
    "bit_xor":              OperatorFnDef(xor_mask, forced_params={"mode": "bin"}),
    "bitflip":              OperatorFnDef(xor_mask, forced_params={"mode": "bin"}),

    # Gaussian distribution
    "gauss":                OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "normal":               OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "gaussian":             OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "gauss_noise":          OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "normal_noise":         OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "gaussian_noise":       OperatorFnDef(rand_noise, forced_params={"distribution": "normal"}),
    "gauss_mut":            OperatorFnDef(mutate_noise, forced_params={"distribution": "normal"}),
    "normal_mutation":      OperatorFnDef(mutate_noise, forced_params={"distribution": "normal"}),
    "gaussian_mutation":    OperatorFnDef(mutate_noise, forced_params={"distribution": "normal"}),

    # Laplace distribution
    "laplace":              OperatorFnDef(rand_noise, forced_params={"distribution": "laplace"}),
    "laplace_noise":        OperatorFnDef(rand_noise, forced_params={"distribution": "laplace"}),
    "laplace_mut":          OperatorFnDef(mutate_noise, forced_params={"distribution": "laplace"}),
    "laplace_mutation":     OperatorFnDef(mutate_noise, forced_params={"distribution": "laplace"}),

    # Cauchy distribution
    "cauchy":               OperatorFnDef(rand_noise, forced_params={"distribution": "cauchy"}),
    "cauchy_noise":         OperatorFnDef(rand_noise, forced_params={"distribution": "cauchy"}),
    "cauchy_mut":           OperatorFnDef(mutate_noise, forced_params={"distribution": "cauchy"}),
    "cauchy_mutation":      OperatorFnDef(mutate_noise, forced_params={"distribution": "cauchy"}),

    # Uniform distribution
    "uniform":              OperatorFnDef(rand_noise, forced_params={"distribution": "uniform"}),
    "uniform_noise":        OperatorFnDef(rand_noise, forced_params={"distribution": "uniform"}),
    "uniform_mut":          OperatorFnDef(mutate_noise, forced_params={"distribution": "uniform"}),
    "uniform_mutation":     OperatorFnDef(mutate_noise, forced_params={"distribution": "uniform"}),

    # Poisson distribution
    "poisson":              OperatorFnDef(rand_noise, forced_params={"distribution": "poisson"}),
    "poisson_noise":        OperatorFnDef(rand_noise, forced_params={"distribution": "poisson"}),
    "poisson_mut":          OperatorFnDef(mutate_noise, forced_params={"distribution": "poisson"}),
    "poisson_mutation":     OperatorFnDef(mutate_noise, forced_params={"distribution": "poisson"}),

    # Bernoulli distribution
    "bernoulli":            OperatorFnDef(rand_noise, forced_params={"distribution": "bernoulli"}),
    "bernoulli_noise":      OperatorFnDef(rand_noise, forced_params={"distribution": "bernoulli"}),
    "bernoulli_mut":        OperatorFnDef(mutate_noise, forced_params={"distribution": "bernoulli"}),
    "bernoulli_mutation":   OperatorFnDef(mutate_noise, forced_params={"distribution": "bernoulli"}),
    "coinflip":             OperatorFnDef(rand_sample, forced_params={"distribution": "bernoulli"}),
    "coinflip_mut":         OperatorFnDef(mutate_sample, forced_params={"distribution": "bernoulli"}),
    "coinflip_mutation":    OperatorFnDef(mutate_sample, forced_params={"distribution": "bernoulli"}),

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


def create_mutation_operator(
    method: str,
    encoding: Optional[Encoding] = None,
    name: Optional[str] = None,
    **kwargs
) -> OperatorFromLambda:
    """
    Create a mutation operator by name.

    Parameters
    ----------
    method : str
        Key into :data:`mutation_ops_map`.
    encoding : Encoding, optional
        Encoding applied to the genotype after mutation.
    name : str, optional
        Display name; defaults to *method*.
    **kwargs
        Parameters forwarded to the mutation function (e.g.,
        ``N``, ``F``, ``distribution``).

    Returns
    -------
    OperatorFromLambda
        The wrapped mutation operator.
    """

    if name is None:
        name = method

    return OperatorFromLambda(operator_fn=mutation_ops_map[method.lower()], name=method, encoding=encoding, preserves_order=True, **kwargs)
