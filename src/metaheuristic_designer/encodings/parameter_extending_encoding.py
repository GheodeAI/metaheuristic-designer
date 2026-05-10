"""
Encoding that splits a genotype into a solution part and auxiliary parameters.
"""

from __future__ import annotations
from abc import ABC
from typing import Iterable, Tuple, Optional
import numpy as np
from ..encoding import Encoding, DefaultEncoding
from ..utils import MatrixLike


class ParameterExtendingEncoding(Encoding, ABC):
    """
    Encoding that appends extra parameters to the solution genotype.

    The genotype vector is split into two parts: the first *dimension*
    elements hold the actual solution, and the remaining elements store
    one or more named parameter blocks (e.g., velocity for PSO, mutation
    strengths for self-adaptation).  A base encoding is applied to the
    solution part; the parameters are stored raw.

    Parameters
    ----------
    dimension : int
        Number of decision variables in the solution.
    param_sizes : iterable of ``(name, length)``
        Named blocks of extra parameters appended to the genotype.
    base_encoding : Encoding, optional
        Encoding applied to the solution part.  Defaults to
        :class:`DefaultEncoding`.
    verify : bool, optional
        If ``True``, additional shape and key checks are performed
        in :meth:`encode_params`.
    **kwargs
        Forwarded to :class:`Encoding`.
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
        """Extract the auxiliary parameter blocks from a genotype matrix.

        Parameters
        ----------
        genotype : MatrixLike
            The full genotype matrix (solution + parameters).
        copy : bool, optional
            Whether to return copies of the parameter arrays.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their sub-arrays.
        """

        param_dict = {}
        param_vec = self.extract_params(genotype)

        for name, length in self.param_sizes:
            param_matrix = param_vec[:, :length]
            param_dict[name] = param_matrix
            param_vec = param_vec[:, length:]

        return param_dict

    def extract_solution(self, population_matrix: MatrixLike) -> MatrixLike:
        """Return only the solution part of the genotype matrix.

        Parameters
        ----------
        population_matrix : MatrixLike
            The full genotype matrix.

        Returns
        -------
        MatrixLike
            The first ``dimension`` columns containing the solution.
        """

        return population_matrix[:, : self.dimension]

    def extract_params(self, population_matrix: MatrixLike) -> MatrixLike:
        """Return only the auxiliary-parameter part of the genotype matrix.

        Parameters
        ----------
        population_matrix : MatrixLike
            The full genotype matrix.

        Returns
        -------
        MatrixLike
            The columns beyond ``dimension`` that contain the extra parameters.
        """

        return population_matrix[:, self.dimension :]

    def encode_params(self, param_dict: dict) -> MatrixLike:
        """Stack a dictionary of parameter arrays into a single matrix.

        Parameters
        ----------
        param_dict : dict
            Mapping from parameter names to 2-D arrays.

        Returns
        -------
        MatrixLike
            A (population_size, total_param_size) array.
        """

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
