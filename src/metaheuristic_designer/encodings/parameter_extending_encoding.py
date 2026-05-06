from __future__ import annotations
from abc import ABC
from typing import Iterable, Tuple, Optional
import numpy as np
from ..encoding import Encoding, DefaultEncoding
from ..utils import MatrixLike


class ParameterExtendingEncoding(Encoding, ABC):
    """
    Abstract Extended Encoding class.

    This kind of encoding will represent solutions as a vector with the solution and some other information concatenated to the vector.
    This interface is intended to be used in swarm-based or adaptative algorithms.
    """

    def __init__(
        self, dimension: int, param_sizes: Iterable[Tuple[str, int]], base_encoding: Optional[Encoding] = None, verify: bool = False, **kwargs
    ):
        self.dimension = dimension
        self.param_sizes = param_sizes
        self.extended_parameters = [p for p, _ in param_sizes]
        self.nparams = sum([param_size for _, param_size in param_sizes])
        if base_encoding is None:
            base_encoding = DefaultEncoding()
        self.base_encoding = base_encoding
        self.verify = verify

        super().__init__(**kwargs)

    def encode(self, solution: Iterable, params: Optional[dict] = None) -> MatrixLike:
        solution_encoded = self.base_encoding.encode(solution)
        if params is None:
            params_encoded = np.zeros((solution_encoded.shape[0], self.nparams))
        else:
            params_encoded = self.encode_params(params)

        return np.hstack([solution_encoded, params_encoded])

    def decode(self, population_matrix: MatrixLike) -> Iterable:
        solution_matrix = self.extract_solution(population_matrix)
        return self.base_encoding.decode(solution_matrix)

    def decode_params(self, genotype: MatrixLike, copy: bool = True) -> dict:
        param_dict = {}
        param_vec = self.extract_params(genotype)

        for name, length in self.param_sizes:
            param_matrix = param_vec[:, :length]
            param_dict[name] = param_matrix
            param_vec = param_vec[:, length:]

        return param_dict

    def extract_solution(self, population_matrix: MatrixLike) -> MatrixLike:
        return population_matrix[:, : self.dimension]

    def extract_params(self, population_matrix: MatrixLike) -> MatrixLike:
        return population_matrix[:, self.dimension :]

    def encode_params(self, param_dict: dict) -> MatrixLike:
        if self.verify:
            assert param_dict.keys() == {name for name, _ in self.param_sizes}

        # check the first available parameter to obtain the population size
        sample_param_name, _ = self.param_sizes[0]
        sample_param_vector = param_dict[sample_param_name]
        population_size, _ = sample_param_vector.shape

        if self.verify:
            assert param_dict.keys() == {name for name, _ in self.param_sizes}
            for name, size in self.param_sizes:
                arr = param_dict[name]
                assert arr.ndim == 2, f"Parameter '{name}' must be a 2D array"
                assert arr.shape[0] == population_size, f"Population size mismatch for '{name}'"
                assert arr.shape[1] == size, f"Wrong block size for '{name}'"

        vcounter = 0
        result = np.empty((population_size, self.nparams))
        for param_name, param_size in self.param_sizes:
            result[:, vcounter : vcounter + param_size] = param_dict[param_name]
            vcounter += param_size

        return result
