from __future__ import annotations
from typing import Iterable
from copy import copy
from ..operator import Operator


class CompositeOperator(Operator):
    """
    Operator class that utilizes a list of operators to modify individuals.

    Parameters
    ----------
    method: str
        Type of operator that will be applied.
    op_list: List[Operator]
        List of operators that will be used.
    params: ParamScheduler or dict, optional
        Dictionary of parameters to define the operator.
    name: str, optional
        Name that is associated with the operator.
    """

    def __init__(self, op_list: Iterable[Operator], name: str = None, encoding=None, random_state=None):
        """
        Constructor for the OperatorMeta class
        """

        if name is None:
            op_names = []
            for op in op_list:
                if not isinstance(op, Operator):
                    op_names.append("lambda_func")
                else:
                    op_names.append(op.name)

            joined_names = ", ".join(op_names)
            name = f"Sequence ({joined_names})"

        # We need to define the op_list before the constructor since it's used in the update method
        self.op_list = op_list

        super().__init__(name=name, encoding=encoding, random_state=random_state)

    def evolve(self, population, initializer=None):
        new_population = copy(population)

        for op in self.op_list:
            new_population = op.evolve(new_population, initializer)

        return new_population

    def update(self, progress):
        super().update(progress)

        for op in self.op_list:
            if isinstance(op, Operator):
                op.update(progress)

    def get_state(self) -> dict:
        data = super().get_state()

        data["op_list"] = []

        for op in self.op_list:
            if isinstance(op, Operator):
                data["op_list"].append(op.get_state())
            else:
                data["op_list"].append("lambda_func")

        return data
