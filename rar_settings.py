"""The rank-anchored replay configuration chosen by the rule of registration section 17, or
none if no configuration beats replay on the development criterion. Given the section 17 out
dir and a family as well, the same rule over that family's runs against 17's replay run: the
one-sided floor of 17.3, the hard-negative replay and server averaging of 17.4, or the exact
running blend of 17.5.

usage: python rar_settings.py <development out dir> [<17 out dir> [floor|hard|average|blend]]
"""
import json
import os
import sys

import numpy as np

import t1_report as t1
from l1_settings import _record, earlier_guard_ndcg

BASELINE = "rar-fedavg-replay-A-s123"
CANDIDATES = {(lam, retention): f"rar-anchor-lam{lam}-{retention}-A-s123"
              for retention in ("random", "fragile") for lam in (0.5, 2.0)}
FAMILIES = {"floor": {(lam, "random"): f"floor-lam{lam}-A-s123" for lam in (0.5, 2.0)},
            "hard": {(lam, "random"): f"hard-lam{lam}-A-s123" for lam in (0.5, 2.0)},
            "average": {(k, "tail"): f"average-k{k}-A-s123" for k in (2, 4, 8)},
            "blend": {(b, "exact"): f"blend-b{b}-A-s123" for b in (0.25, 0.5)}}
LABELS = {"average": ("window", "average"), "blend": ("blend", "average")}
TOLERANCE = 0.005


def measures(record):
    return earlier_guard_ndcg(record), float(np.mean(t1.recompute(record)[0]))


def choose(candidates, baseline):
    """Among candidates (guard score, A) whose A is at least the baseline's minus TOLERANCE,
    the highest guard score, a tie going to the smaller lambda and then to random retention;
    None unless that score is higher than the baseline's."""
    eligible = {k: v for k, v in candidates.items() if v[1] >= baseline[1] - TOLERANCE}
    if not eligible:
        return None
    best = max(sorted(eligible, key=lambda k: (k[0], k[1] != "random")),
               key=lambda k: eligible[k][0])
    return best if eligible[best][0] > baseline[0] else None


def main(out_dir, baseline_dir=None, family="floor"):
    names = CANDIDATES if baseline_dir is None else FAMILIES[family]
    labels = LABELS.get(family, ("lambda_anchor", "retention")) if baseline_dir else (
        "lambda_anchor", "retention")
    baseline = measures(_record(os.path.join(baseline_dir or out_dir, BASELINE)))
    candidates = {k: measures(_record(os.path.join(out_dir, name))) for k, name in names.items()}
    chosen = choose(candidates, baseline)
    print(json.dumps({"chosen": None if chosen is None else dict(zip(labels, chosen)),
                      "replay": {"earlier_guard_ndcg": baseline[0], "A": baseline[1]},
                      "candidates": {f"{lam}-{ret}": {"earlier_guard_ndcg": g, "A": a}
                                     for (lam, ret), (g, a) in candidates.items()}},
                     indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
