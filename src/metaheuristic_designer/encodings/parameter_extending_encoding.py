from __future__ import annotations
from abc import ABC
from typing import Iterable, Tuple, Any
import numpy as np
from ..encoding import Encoding, DefaultEncoding

class ParameterExtendingEncoding(Encoding, ABC):
    """
    Abstract Extended Encoding class.

    This kind of encoding will represent solutions as a vector with the solution and some other information concatenated to the vector.
    This interface is intended to be used in swarm-based or adaptative algorithms.
    """

    def __init__(self, vecsize: int, param_sizes: Iterable[Tuple[str, int]], base_encoding: Encoding = None, verify = False):
        self.vecsize = vecsize
        self.param_sizes = param_sizes
        self.nparams = sum([param_size for _, param_size in param_sizes])
        if base_encoding is None:
            base_encoding = DefaultEncoding()
        self.base_encoding = base_encoding
        self.verify = verify

        super().__init__(vectorized=base_encoding.vectorized)

    def extract_solution(self, population_matrix: np.ndarray) -> np.ndarray:
        if self.vectorized:
            result = population_matrix[:, :self.vecsize]
        else:
            result = population_matrix[:self.vecsize]
        return result

    def extract_params(self, population_matrix: np.ndarray) -> np.ndarray:
        if self.vectorized:
            result = population_matrix[:, self.vecsize:]
        else:
            result = population_matrix[self.vecsize:]
        return result

    def encode_params_func(self, param_dict: dict) -> np.ndarray:
        if self.verify:
            assert param_dict.keys() == {name for name, _ in self.param_sizes}
        
        # check the first available parameter to obtain the population size
        sample_param_name, _ = self.param_sizes[0]
        sample_param_vector = param_dict[sample_param_name] 
        if sample_param_vector.ndim == 2:
            population_size, nparams = sample_param_vector.shape
        else:
            population_size = 1
            nparams = sample_param_vector.shape[0]
        
        if self.verify:
            assert nparams == self.nparams

        vcounter = 0
        result = np.empty((population_size, nparams))
        for param_name, param_size in self.param_sizes:
            result[:, vcounter:vcounter+param_size] = param_dict[param_name]
            vcounter += param_size
        
        return result
    
    def decode_params_func(self, genotype: np.np.ndarray) -> dict:
        param_dict = {}
        param_vec = self.extract_params(genotype)
        for name, length in self.param_sizes:
            param_dict[name] = param_vec[:, :length]
            param_vec = param_vec[:, length:]
        
        return param_dict
    
    def encode_params(self, param_dict: dict) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: np.ndarray
            Population that should be decoded.

        Returns
        -------
        solutions: Iterable
            List/array of solutions.
        """

        return self.encode_params_func(param_dict)

    def decode_params(self, population: np.ndarray) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: np.ndarray
            Population that should be decoded.

        Returns
        -------
        solutions: Iterable
            List/array of solutions.
        """

        param_dict = None
        if self.vectorized:
            param_dict = self.decode_params_func(population)
        else:
            param_list = [self.decode_params_func(indiv) for indiv in population]
            param_dict = {key: np.array([p_dict[key] for p_dict in param_list]) for key in param_list[0]}

            if self.verify:
                assert param_dict.keys() == set(map(lambda x: x[0], self.param_sizes))

        return param_dict
    
    def encode_func(self, solution: Any, params: dict = None) -> np.np.ndarray:
        solution_encoded = self.base_encoding.encode_func(solution)
        if params is None:
            params_encoded = np.zeros((solution_encoded.shape[0], self.nparams))
        else:
            params_encoded = self.encode_params(params)

        return np.hstack([solution_encoded, params_encoded])

    def decode_func(self, indiv: np.np.ndarray) -> np.np.ndarray:
        solution_matrix = self.extract_solution(indiv)
        return self.base_encoding.decode_func(solution_matrix)

    def encode(self, solutions: Iterable, params: dict = None) -> np.ndarray:
        """
        Encodes a list of solutions to our problem to an population matrix.

        Parameters
        ----------
        solutions: Iterable
            Solutions that should be encoded.

        Returns
        -------
        population: np.ndarray
            Population array.
        """

        if self.vectorized:
            population = self.encode_func(solutions, params=params)
        else:
            population = np.asarray([self.encode_func(indiv) for indiv in solutions])

        return population
