"""l1_report.py: the LoTTE hypotheses as registered in section 16 and its amendment."""
import numpy as np

import l1_report as l1

KEYS = [(s, seed) for s in "AB" for seed in (123, 2024, 3407)]


def block(g_shared, g_local, end_shared=0.5, end_local=0.4):
    per_run = {}
    for i, (s, seed) in enumerate(KEYS):
        for arm, g, end in ((l1.D, g_shared[i], end_shared), (l1.B, g_local[i], end_local),
                            (l1.C, g_shared[i], end_shared), (l1.B0, g_local[i], end_local)):
            per_run[(arm, s, seed)] = {"A": 0.05, "G": g, "bwt": 0.01, "end_ndcg_earlier": end}
    return per_run


def test_retention_needs_five_of_six_paired_wins_and_a_higher_end_score():
    losses = np.array([0.0] * 80 + [0.02] * 20)
    passing = l1.hypotheses(block([0.01] * 5 + [0.05], [0.03] * 6), losses)
    assert passing["H4"]["passes"] and passing["H5"]["passes"] and passing["H3"]["passes"]
    assert passing["H1"]["passes"] and passing["H2"]["passes"]
    assert not l1.hypotheses(block([0.004] * 6, [0.03] * 6), losses)["H2"]["passes"]
    four = l1.hypotheses(block([0.01] * 4 + [0.05] * 2, [0.03] * 6), losses)
    assert not four["H4"]["passes"]
    lower_end = l1.hypotheses(block([0.01] * 6, [0.03] * 6, 0.4, 0.5), losses)
    assert not lower_end["H4"]["passes"]
    assert not l1.hypotheses(block([0.01] * 6, [0.03] * 6), losses[:85])["H3"]["passes"]
