from __future__ import annotations
from copy import copy
from ..constraint_handler import ConstraintHandler
from ..encodings import ParameterExtendingEncoding
from ..utils import MatrixLike, ScalarLike


class ExtendedConstraintHandler(ConstraintHandler):
    def __init__(self, solution_handler: ConstraintHandler, param_handler_dict: dict, encoding: ParameterExtendingEncoding, **kwargs):
        assert isinstance(encoding, ParameterExtendingEncoding), "An `ExtendedEncoding` instance must be used with this type of ConstraintHandler"

        self.solution_handler = solution_handler
        self.param_handler_dict = param_handler_dict
        self.encoding = encoding
        super().__init__(**kwargs)

    def repair_solution(self, genotype_matrix: MatrixLike) -> MatrixLike:
        solution_matrix = self.encoding.extract_solution(genotype_matrix)
        params = self.encoding.decode_params(genotype_matrix)

        solution_matrix_repaired = self.solution_handler.repair_solution(solution_matrix)

        param_fixed = copy(params)
        for param_name, _ in self.encoding.param_sizes:
            param_matrix = params[param_name]
            param_fixed[param_name] = self.param_handler_dict[param_name].repair_solution(param_matrix)

        # In repair_solution, before the encode call
        return self.encoding.encode(solution_matrix_repaired, param_fixed)

    def penalty(self, genotype_matrix: MatrixLike) -> ScalarLike:
        solution_matrix = self.encoding.extract_solution(genotype_matrix)
        params = self.encoding.decode_params(genotype_matrix)

        penalty = self.solution_handler.penalty(solution_matrix)
        for param_name, _ in self.encoding.param_sizes:
            param_matrix = params[param_name]
            penalty += self.param_handler_dict[param_name].penalty(param_matrix)

        return penalty
