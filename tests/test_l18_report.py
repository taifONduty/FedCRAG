"""l18_report.py: block 18's hypotheses as registered in section 18."""
import l18_report as l18

KEYS = [(s, seed) for s in "AB" for seed in (123, 2024, 3407)]


def block(floor_g, floor_end, strong_g=(0.02,) * 6, hn_end=0.60, floorhn_end=0.60):
    per_run = {}
    for i, (s, seed) in enumerate(KEYS):
        for arm, g, end, a in ((l18.D, 0.03, 0.50, 0.10), (l18.FLOOR, floor_g[i], floor_end, 0.10),
                               (l18.HARD, 0.05, 0.60, 0.15), (l18.HN, 0.05, hn_end, 0.20),
                               (l18.FLOORHN, strong_g[i], floorhn_end, 0.20),
                               (l18.AVERAGE, 0.03, 0.50, 0.10)):
            per_run[(arm, s, seed)] = {"A": a, "G": g, "end_ndcg_earlier": end}
    return per_run


def test_the_floor_needs_five_of_six_lower_g_and_higher_end_scores():
    passing = l18.hypotheses(block([0.01] * 5 + [0.05], 0.55))
    assert passing["M1-M3 floor vs D"]["passes"]
    assert passing["T1 hard vs D"]["passes"] and passing["T2 hn vs D"]["passes"]
    assert passing["M4 floorhn vs hn"]["passes"]
    assert not passing["A1 average vs D"]["passes"]
    assert not l18.hypotheses(block([0.01] * 4 + [0.05] * 2, 0.55))["M1-M3 floor vs D"]["passes"]
    assert not l18.hypotheses(block([0.01] * 6, 0.45))["M1-M3 floor vs D"]["passes"]
    assert not l18.hypotheses(block([0.01] * 6, 0.50))["M1-M3 floor vs D"]["passes"]


def test_the_floor_on_the_strong_recipe_may_not_lose_more_than_the_tolerance_of_quality():
    within = l18.hypotheses(block([0.01] * 6, 0.55, floorhn_end=0.596))
    assert within["M4 floorhn vs hn"]["passes"]
    below = l18.hypotheses(block([0.01] * 6, 0.55, floorhn_end=0.594))
    assert not below["M4 floorhn vs hn"]["passes"]
    assert not l18.hypotheses(block([0.01] * 6, 0.55, strong_g=(0.06,) * 6))[
        "M4 floorhn vs hn"]["passes"]
