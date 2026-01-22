from __future__ import annotations
from ..constraint_handler import ConstraintHandler

class ExtendedConstraintHandler(ConstraintHandler):
    def __init__(self, solution_handler: ConstraintHandler, param_handler_dict: dict, encoding: ExtendedEncoding):
        assert isinstance(encoding, ExtendedEncoding), "An `ExtendedEncoding` instance must be used with this type of ConstraintHandler"

        self.solution_handler = solution_handler
        self.param_handler_dict = param_handler_dict
        self.encoding = encoding
    
    def repair_solution(self, solution):
        solution_vec = self.encoding.decode(solution[None, :])
        param = self.encoding.decode_params(solution[None, :])
        
        solution_vec_fixed = self.solution_handler.repair_solution(solution_vec)
        param_fixed = copy(param)
        for param_name, _ in self.encoding.param_sizes:
            param_vec = param[param_name]
            param_fixed[param_name] = self.param_handler_dict[param_name].repair_solution(param_vec)

        return self.encoding.encode(solution_vec_fixed, param_fixed)

    def penalty(self, solution):
        solution_vec = self.encoding.decode(solution[None, :])[0]
        param = self.encoding.decode_params(solution[None, :])
        
        penalty = self.solution_handler.penalty(solution_vec)
        for param_name, _ in self.encoding.param_sizes:
            param_vec = param[param_name]
            penalty += self.param_handler_dict[param_name].penalty(param_vec)

        return penalty