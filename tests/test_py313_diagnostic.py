import unittest
import numpy as np

from namd_analysis.kinetics import build_rate_matrix, fit_master_equation, parse_edges, propagate

GROUPS = ["CBM", "BCF", "PCBM", "VBM"]


class Py313Diagnostic(unittest.TestCase):
    def test_print_blind_case(self):
        edges = parse_edges("CBM->BCF,BCF->VBM", GROUPS)
        K = build_rate_matrix([6.0, 1.0], edges, len(GROUPS))
        time = np.linspace(0.0, 1.5, 300)
        observed = propagate(K, np.array([1.0, 0.0, 0.0, 0.0]), time)
        fit = fit_master_equation(
            time, observed, GROUPS,
            parse_edges("CBM->BCF,BCF->VBM,PCBM->BCF", GROUPS),
        )
        for rate in fit.rates:
            print("DIAG1", rate.name, rate.identified, rate.unidentified_reason,
                  rate.null_space_participation, rate.relative_stderr,
                  rate.degenerate_with, rate.optimizer_bound)

    def test_print_three_state_case(self):
        groups = ["A", "B", "C"]
        time = np.linspace(0.0, 2.0, 401)
        K = build_rate_matrix([5.0, 1.0], parse_edges("A->B,B->C", groups), 3)
        observed = propagate(K, np.array([1.0, 0.0, 0.0]), time)
        fit = fit_master_equation(
            time, observed, groups, parse_edges("A->B,B->C,C->B", groups)
        )
        for rate in fit.rates:
            print("DIAG2", rate.name, rate.identified, rate.unidentified_reason,
                  rate.null_space_participation, rate.relative_stderr,
                  rate.degenerate_with, rate.optimizer_bound)


if __name__ == "__main__":
    unittest.main()
