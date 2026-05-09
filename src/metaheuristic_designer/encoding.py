"""
Base class for the Encoding module.

This module implements a way to have a different representation in the inner working
of the algorithm and the result of the procedure.
"""

from __future__ import annotations
from typing import Iterable, Callable
from abc import ABC, abstractmethod
from .parametrizable_mixin import ParametrizableMixin
from .utils import MatrixLike


class Encoding(ParametrizableMixin, ABC):
    """
    Abstract Encoding class for phenotype-genotype transformations.

    An encoding defines how solutions are represented:
    
    - **Genotype**: Internal representation used by the algorithm 
      (typically numerical vectors for evolutionary computation)
    - **Phenotype**: Problem-specific solution representation 
      (may be discrete, combinatorial, or custom formats)
    
    This abstraction allows algorithms to work with diverse problem types
    while maintaining a unified numerical interface internally.

    Parameters
    ----------
    decode_as_array : bool, optional
        Whether to convert decoded solutions to numpy arrays. Default is False.
    name : str, optional
        Descriptive name for the encoding.
    **kwargs
        Additional encoding-specific parameters.
    
    See Also
    --------
    ParametrizableMixin : Base class for parameter management.
    
    Notes
    -----
    The encoding typically maintains three key properties:
    
    1. **Directionality**: encode() and decode() should be inverse operations
    2. **Consistency**: The same solution always encodes/decodes identically
    3. **Completeness**: All encoded genotypes should map to valid phenotypes
    """

    def __init__(self, decode_as_array: bool = False, name=None, **kwargs):
        """
        Constructor for the Encoding class.
        """
        super().__init__()
        self.name = name
        self.decode_as_array = decode_as_array
        self.store_kwargs(**kwargs)

    def gather_params(self):
        """
        Retrieves the current parameters of the encoding.
        
        This method provides a customizable interface for parameter extraction.
        By default, it wraps :meth:`get_params`, but subclasses can override
        it to implement encoding-specific parameter gathering logic.
        
        Returns
        -------
        params : dict
            Dictionary of current encoding parameters.
        
        See Also
        --------
        get_params : Base method for parameter retrieval from ParametrizableMixin.
        """

        return self.get_params()

    @abstractmethod
    def encode(self, solutions: Iterable) -> MatrixLike:
        """
        Encodes a list of problem solutions into the internal genotype representation.
        
        Transforms from phenotype (problem-specific) to genotype (internal algorithm)
        representation, typically producing numerical vectors or matrices.

        Parameters
        ----------
        solutions : Iterable
            Problem-specific solutions to be encoded. May be in any format defined
            by the problem (scalars, tuples, objects, etc.).

        Returns
        -------
        population : MatrixLike
            Encoded population as a numerical array. Shape should be 
            (n_solutions, solution_dimension).
        
        See Also
        --------
        decode : Inverse operation converting genotypes back to phenotypes.
        """

    @abstractmethod
    def decode(self, population: MatrixLike) -> Iterable:
        """
        Decodes internal genotype representations back into problem solutions.
        
        Transforms from genotype (internal algorithm) to phenotype (problem-specific)
        representation, reversing the :meth:`encode` operation.

        Parameters
        ----------
        population : MatrixLike
            Encoded population in genotype form (typically numerical array).

        Returns
        -------
        solutions : Iterable
            Problem-specific solutions matching the format expected by the objective function.
        
        See Also
        --------
        encode : Forward operation converting phenotypes to genotypes.
        """

    def get_state(self) -> dict:
        """
        Gets the current state of the encoding as a dictionary.
        
        Returns
        -------
        state : dict
            Dictionary containing encoding state information (typically empty for
            simple encodings, but may include parameters for complex ones).
        """
        return {}


class DefaultEncoding(Encoding):
    """
    Default encoding that uses the genotype directly as the solution.
    
    This encoding performs no transformation—encoded and decoded representations
    are identical. Use this when the problem already works with numerical vectors.

    Parameters
    ----------
    decode_as_array : bool, optional
        Whether to convert decoded solutions to numpy arrays. Default is True.
    
    Examples
    --------
    >>> encoding = DefaultEncoding()
    >>> solutions = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    >>> encoded = encoding.encode(solutions)  # Same as input
    >>> decoded = encoding.decode(encoded)    # Same as encoded
    """

    def __init__(self, decode_as_array: bool = True):
        """
        Constructor for the DefaultEncoding class.
        """
        super().__init__(decode_as_array=decode_as_array)

    def encode(self, solution: Iterable) -> MatrixLike:
        """
        Returns the solution unchanged (no transformation).
        
        Parameters
        ----------
        solution : Iterable
            Solution(s) to encode.
        
        Returns
        -------
        genotype : MatrixLike
            Same as input (no transformation applied).
        """
        return solution

    def decode(self, population: MatrixLike) -> Iterable:
        """
        Returns the population unchanged (no transformation).
        
        Parameters
        ----------
        population : MatrixLike
            Population in genotype form.
        
        Returns
        -------
        solutions : Iterable
            Same as input (no transformation applied).
        """
        return population


class EncodingFromLambda(Encoding):
    """
    Encoding that uses user-defined functions for phenotype-genotype transformations.
    
    This class allows quick prototyping of custom encodings without creating
    a new Encoding subclass.

    Parameters
    ----------
    encode_fn : Callable
        Function transforming phenotypes to genotypes. 
        Signature: (solutions: Iterable) -> MatrixLike
    decode_fn : Callable
        Function transforming genotypes back to phenotypes.
        Signature: (population: MatrixLike) -> Iterable
    **kwargs
        Additional encoding parameters.
    
    Examples
    --------
    >>> def encode_permutation(solutions):
    ...     # Convert permutation objects to integer arrays
    ...     return np.array([list(perm) for perm in solutions])
    >>> def decode_permutation(population):
    ...     # Convert integer arrays back to permutation objects
    ...     return [tuple(row) for row in population]
    >>> encoding = EncodingFromLambda(encode_permutation, decode_permutation)
    """

    def __init__(self, encode_fn: Callable, decode_fn: Callable, **kwargs):
        """
        Constructor for the EncodingFromLambda class.
        """
        super().__init__(**kwargs)
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def encode(self, solution: Iterable) -> MatrixLike:
        """
        Encodes solutions using the user-provided function.
        
        Parameters
        ----------
        solution : Iterable
            Solution(s) to encode.
        
        Returns
        -------
        genotype : MatrixLike
            Encoded genotype.
        """
        return self.encode_fn(solution)

    def decode(self, population: MatrixLike) -> Iterable:
        """
        Decodes genotypes using the user-provided function.
        
        Parameters
        ----------
        population : MatrixLike
            Population in genotype form.
        
        Returns
        -------
        solutions : Iterable
            Decoded solutions.
        """
        return self.decode_fn(population)
