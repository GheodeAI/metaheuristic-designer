from __future__ import annotations
from typing import Iterable, Any
from abc import ABC, abstractmethod
import warnings
import numpy as np
from numpy import ndarray


class Encoding(ABC):
    """
    Abstract Encoding class

    This class transforms between phenotype and genotype.
    """

    def __init__(self, vectorized=False, decode_as_array=False):
        self.vectorized = vectorized
        self.decode_as_array = decode_as_array

    @abstractmethod
    def encode_func(self, solution: Any) -> ndarray:
        """
        Convert a solution into an individual. (If vectorized is set it converts a list of solutions into a matrix)

        Parameters
        ----------
        solution: Any
            Solutions that should be encoded.

        Returns
        -------
        individual: ndarray
            Individual vector.
        """

    @abstractmethod
    def decode_func(self, indiv: ndarray) -> Any:
        """
        Convert an individual as a vector into an individual. (If vectorized is set it converts a list of solutions into a matrix)

        Parameters
        ----------
        solution: Any
            Solutions that should be encoded.

        Returns
        -------
        individual: ndarray
            Individual vector.
        """

    def encode(self, solutions: Iterable) -> ndarray:
        """
        Encodes a list of solutions to our problem to an population matrix.

        Parameters
        ----------
        solutions: Iterable
            Solutions that should be encoded.

        Returns
        -------
        population: ndarray
            Population array.
        """

        population = None
        if self.vectorized:
            population = self.encode_func(solutions)
        else:
            population = np.asarray([self.encode_func(indiv) for indiv in solutions])

        return population

    def decode(self, population: ndarray) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: ndarray
            Population that should be decoded.

        Returns
        -------
        solutions: Iterable
            List/array of solutions.
        """

        solutions = None
        if self.vectorized:
            solutions = self.decode_func(population)
        else:
            solutions = [self.decode_func(indiv) for indiv in population]

        if self.decode_as_array:
            solutions = np.asarray(solutions)

        return solutions

    def update(self, population):
        return population.genotype_matrix


class EncodingFromLambda(Encoding):
    """
    Decoder that uses user specified functions.
    """

    def __init__(self, encode_fn: callable, decode_fn: callable, vectorized: bool = False):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

        super().__init__(vectorized=vectorized)

    def encode_func(self, solution: Any) -> Any:
        return self.encode_fn(solution)

    def decode_func(self, population: Any) -> Any:
        return self.encode_fn(population)


class DefaultEncoding(Encoding):
    """
    Default encoder that uses the genotype directly as a solution.
    """

    def __init__(self, decode_as_array=True):
        super().__init__(vectorized=True, decode_as_array=decode_as_array)

    def encode_func(self, solution: Any) -> Any:
        return solution

    def decode_func(self, population: Any) -> Any:
        return population


class ExtendedEncoding(Encoding, ABC):
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

    def encode_func(self, solution: Any, params: dict = None) -> np.ndarray:
        solution_encoded = self.base_encoding.encode_func(solution)
        if params is None:
            params_encoded = np.zeros((1, self.nparams))
        else:
            params_encoded = self.encode_params(params)

        solution_encoded = np.hstack([solution_encoded, params_encoded])
        
        return solution_encoded

    def decode_func(self, genotype: np.ndarray) -> np.ndarray:
        if self.vectorized:
            return self.base_encoding.decode(genotype[:, : self.vecsize])
        else:
            return self.base_encoding.decode(genotype[: self.vecsize])

    def encode(self, solutions: Iterable, params: dict = None) -> ndarray:
        """
        Encodes a list of solutions to our problem to an population matrix.

        Parameters
        ----------
        solutions: Iterable
            Solutions that should be encoded.

        Returns
        -------
        population: ndarray
            Population array.
        """

        population = None
        if self.vectorized:
            population = self.encode_func(solutions, params=params)
        else:
            population = np.asarray([self.encode_func(indiv) for indiv in solutions])

        return population

    def extract_solution(self, population_matrix: ndarray) -> ndarray:
        return population_matrix[:, :self.vecsize]

    def extract_params(self, population_matrix: ndarray) -> ndarray:
        return population_matrix[:, self.vecsize:]

    def encode_params(self, param_dict: dict) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: ndarray
            Population that should be decoded.

        Returns
        -------
        solutions: Iterable
            List/array of solutions.
        """

        param_matrix = self.encode_params_func(param_dict)

        return param_matrix
    
    def encode_params_func(self, param_dict: dict) -> ndarray:
        if self.verify:
            assert param_dict.keys() == set(map(lambda x: x[0], self.param_sizes))
        
        # check the first available parameter to obtain the population size
        sample_param = self.param_sizes[0]
        sample_param_name = sample_param[0]
        population_size = len(param_dict[sample_param_name])

        vcounter = 0
        result = np.empty((population_size, self.nparams))
        for param_name, param_size in self.param_sizes:
            result[:, vcounter:vcounter+param_size] = param_dict[param_name]
            vcounter += param_size
        
        return result

    
    def decode_params(self, population: ndarray) -> Iterable:
        """
        Decodes a population matrix into a list/array of solutions.

        Parameters
        ----------
        population: ndarray
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
    
    
    def decode_params_func(self, genotype: np.ndarray) -> dict:
        param_dict = {}
        param_vec = genotype[:, self.vecsize:]
        for name, length in self.param_sizes:
            param_dict[name] = param_vec[:, :length]
            param_vec = param_vec[:, length:]
        
        return param_dict
    
