import pytest

import pyevolcomp as pec
import pyevolcomp.Algorithms
import pyevolcomp.SearchMethods
import pyevolcomp.Operators
import pyevolcomp.Decoders
import pyevolcomp.benchmarks

def test_test():
    print("hi")

    params = {
        # General
        "stop_cond": "neval or time_limit or fit_target",
        "time_limit": 1.0,
        "cpu_time_limit": 100.0,
        "ngen": 1000,
        "neval": 6e5,
        "fit_target": 1e-3,

        "verbose": True,
        "v_timer": 0.5
    }

    strat = pec.Algorithms.HillClimb(pec.Operators.OperatorReal("RandNoise", {"method":"Cauchy", "F": 0.001}))
    alg = pec.SearchMethods.GeneralSearch(strat, params)

    objfunc = pec.benchmarks.Sphere(10, "min")
