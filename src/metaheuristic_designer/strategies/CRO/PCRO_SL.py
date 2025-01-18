from __future__ import annotations
from typing import Union, List
import random
from ...ParamScheduler import ParamScheduler
from .CRO_SL import CRO_SL


class PCRO_SL(CRO_SL):
    """
    Probabilistic Coral Reef Optimization with Substrate Layers.

    Published in:
    - Pérez-Aracil, Jorge, et al. "New Probabilistic, Dynamic Multi-Method Ensembles for Optimization Based on the CRO-SL." Mathematics 11.7 (2023): 1666.

    Original implementation in https://github.com/jperezaracil/PyCROSL/
    """

    def __init__(
        self,
        initializer: Initializer,
        operator_list: List[Operator],
        params: ParamScheduler | dict = None,
        name: str = "PCRO-SL",
    ):
        super().__init__(initializer, operator_list, params=params, name=name)
        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)
        self.operator.chosen_idx = self.operator_idx

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)
        self.operator.chosen_idx = self.operator_idx
