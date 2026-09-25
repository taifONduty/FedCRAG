"""Seed churn (the amended registration section 16): how far two seeds of the same arm and
order differ per query at the same stage, beside the arm's regression and loss share.

For two interchangeable models the expected positive part of a score difference is half the
mean absolute difference, and the share of queries losing at least 0.010 is half the share
differing by that much, so both numbers are on the scale of G and of the loss share.

usage: python seed_churn.py <out dir> <run prefix>
"""
import itertools
import sys
from collections import defaultdict

import numpy as np

import t1_report as t1


def churn(a, b):
    """Half the mean absolute per-query difference between two seeds' acquisition references,
    and half the share of queries that differ by at least t1.LOSS."""
    diffs = []
    for c in a["clients"]:
        for e in map(str, a["order"][c]):
            sa = t1.scores(a["references"][c][e]["scores"])
            sb = t1.scores(b["references"][c][e]["scores"])
            diffs += [abs(sa[q] - sb[q]) for q in sa]
    diffs = np.array(diffs)
    return float(np.mean(diffs) / 2), float(np.mean(diffs >= t1.LOSS) / 2)


def main(out_dir, prefix):
    runs = t1.load(out_dir, prefix)
    groups = defaultdict(dict)
    for (arm, schedule, seed), record in runs.items():
        groups[(arm, schedule)][seed] = record
    print("| arm | order | seed churn | seed loss share | G | loss share |")
    print("|---|---|---:|---:|---:|---:|")
    for (arm, schedule), by_seed in sorted(groups.items()):
        pairs = [churn(by_seed[x], by_seed[y]) for x, y in itertools.combinations(sorted(by_seed), 2)]
        if not pairs:
            continue
        summaries = [t1.run_summary(r, *t1.recompute(r)) for r in by_seed.values()]
        print(f"| {arm} | {schedule} | {np.mean([p[0] for p in pairs]):.4f} |"
              f" {np.mean([p[1] for p in pairs]):.4f} |"
              f" {np.mean([s['G'] for s in summaries]):.4f} |"
              f" {np.mean([s['query_loss_rate'] for s in summaries]):.4f} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
