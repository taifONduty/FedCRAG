"""The §9.5 scorer must count exits and bucket margins the way it claims."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_margins import exits_between, quartile_table  # noqa: E402


def rec(rel_ranks, margin10):
    return {"rel_ranks": rel_ranks, "margins": {"10": margin10},
            "top_ids": [], "ndcg10": 0.5}


def test_exits_count_relevant_docs_leaving_the_top_ten():
    prev = {"q1": rec({"a": 3, "b": 9}, 0.01),      # both in top 10
            "q2": rec({"c": 40}, 0.20),              # not at risk
            "q3": rec({"d": 1}, 0.05)}
    curr = {"q1": rec({"a": 2, "b": 15}, 0.02),      # b exits
            "q2": rec({"c": 5}, 0.20),
            "q3": rec({"d": 1}, 0.05)}                # stays
    out = dict((q, (n_in, n_out, m)) for q, n_in, n_out, m in exits_between(prev, curr))
    assert out["q1"] == (2, 1, 0.01)
    assert out["q3"] == (1, 0, 0.05)
    assert "q2" not in out


def test_quartile_table_puts_small_margins_in_q1_and_uses_loo_sign():
    rng = np.random.default_rng(0)
    n = 40
    # every query has one relevant doc in the top 10 at round 1; at round 2 the
    # smallest-margin half exits and the largest-margin half stays
    margins = np.sort(rng.uniform(0.001, 0.5, size=n))
    prev = {f"q{i}": rec({"d": 5}, float(m)) for i, m in enumerate(margins)}
    curr = {f"q{i}": rec({"d": 50 if i < n // 2 else 5}, float(m))
            for i, m in enumerate(margins)}
    rep = {"rounds": {"frozen": {"c": prev}, "round_1": {"c": curr}},
           "loo_alignment": {"round_1": {"c": -0.1}}}
    buckets, by_sign, rows = quartile_table(rep)
    assert buckets[0] == [10, 10] and buckets[1] == [10, 10]     # Q1, Q2 all exit
    assert buckets[2] == [0, 10] and buckets[3] == [0, 10]       # Q3, Q4 none exit
    assert by_sign["neg"] == [20, 40] and by_sign["nonneg"] == [0, 0]
    assert rows == [("c", "round_1", -0.1, 20, 40)]
