"""l1_report.py: the LoTTE hypotheses as registered in section 16 and its amendment."""
import numpy as np

import l1_report as l1

KEYS = [(s, seed) for s in "AB" for seed in (123, 2024, 3407)]


def block(g_shared, g_local, end_shared=0.5, end_local=0.4, g_control=0.001):
    per_run = {}
    for i, (s, seed) in enumerate(KEYS):
        for arm, g, end in ((l1.D, g_shared[i], end_shared), (l1.B, g_local[i], end_local),
                            (l1.C, g_shared[i], end_shared), (l1.B0, g_local[i], end_local),
                            (l1.NO_SHIFT, g_control, end_shared)):
            per_run[(arm, s, seed)] = {"A": 0.05, "G": g, "bwt": 0.01, "end_ndcg_earlier": end}
    return per_run


def losses(d, control=np.zeros(100)):
    return {l1.D: d, l1.NO_SHIFT: control}


def test_retention_needs_five_of_six_paired_wins_and_a_higher_end_score():
    d = np.array([0.0] * 80 + [0.02] * 20)
    passing = l1.hypotheses(block([0.01] * 5 + [0.05], [0.03] * 6), losses(d))
    assert all(passing[h]["passes"] for h in ("H1", "H2", "H3", "H4", "H5", "H6"))
    assert not l1.hypotheses(block([0.004] * 6, [0.03] * 6), losses(d))["H2"]["passes"]
    four = l1.hypotheses(block([0.01] * 4 + [0.05] * 2, [0.03] * 6), losses(d))
    assert not four["H4"]["passes"]
    lower_end = l1.hypotheses(block([0.01] * 6, [0.03] * 6, 0.4, 0.5), losses(d))
    assert not lower_end["H4"]["passes"]
    assert not l1.hypotheses(block([0.01] * 6, [0.03] * 6), losses(d[:85]))["H3"]["passes"]


def test_regression_beyond_churn_needs_the_shifted_stream_above_the_control():
    d = np.array([0.0] * 80 + [0.02] * 20)
    as_much = l1.hypotheses(block([0.01] * 6, [0.03] * 6, g_control=0.02), losses(d))
    assert not as_much["H6"]["passes"]
    same_share = l1.hypotheses(block([0.01] * 6, [0.03] * 6), losses(d, d))
    assert not same_share["H6"]["passes"]
