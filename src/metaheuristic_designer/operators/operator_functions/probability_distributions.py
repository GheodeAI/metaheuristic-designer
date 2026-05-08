from abc import ABC, abstractmethod

from ...utils import RNGLike, TensorLike


class Distribution(ABC):
    @abstractmethod
    def sample(shape: tuple, random_state: RNGLike) -> TensorLike:
        """_summary_

        Parameters
        ----------
        shape : tuple
            _description_
        random_state : RNGLike
            _description_

        Returns
        -------
        TensorLike
            _description_
        """