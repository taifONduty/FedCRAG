"""The rank-anchored replay configuration chosen by the rule of registration section 17, or
none if no configuration beats replay on the development criterion.

usage: python rar_settings.py <development out dir>
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


def main(out_dir):
    baseline = measures(_record(os.path.join(out_dir, BASELINE)))
    candidates = {k: measures(_record(os.path.join(out_dir, name)))
                  for k, name in CANDIDATES.items()}
    chosen = choose(candidates, baseline)
    print(json.dumps({"chosen": None if chosen is None
                      else {"lambda_anchor": chosen[0], "retention": chosen[1]},
                      "replay": {"earlier_guard_ndcg": baseline[0], "A": baseline[1]},
                      "candidates": {f"{lam}-{ret}": {"earlier_guard_ndcg": g, "A": a}
                                     for (lam, ret), (g, a) in candidates.items()}},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
