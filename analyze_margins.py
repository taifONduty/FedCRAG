"""Score the registered §9.5 prediction from margin-replay outputs.

Prediction (committed before data): relevant-document EXITS from the top 10
between consecutive rounds concentrate in queries whose PRE-round margin at
cutoff 10 sits in the lowest quartile, and are more frequent in rounds where
the client's leave-one-out alignment is negative.

Falsifier: exit rates that do not depend on the pre-round margin quartile.
Then "depth-graded damage" demotes from mechanism to descriptive observation.

Usage: python analyze_margins.py replay/*.json.gz
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def round_labels(rep):
    labels = [k for k in rep["rounds"] if k.startswith("round_")]
    return sorted(labels, key=lambda s: int(s.split("_")[1]))


def exits_between(prev, curr):
    """Relevant docs in top 10 at prev round and outside it at curr round."""
    out = []
    for qid, rec in prev.items():
        in_prev = {d for d, r in rec["rel_ranks"].items() if r <= 10}
        if not in_prev:
            continue
        curr_ranks = curr.get(qid, {}).get("rel_ranks", {})
        exited = {d for d in in_prev if curr_ranks.get(d, 10**9) > 10}
        out.append((qid, len(in_prev), len(exited), rec["margins"].get("10")))
    return out


def quartile_table(rep):
    """Exit rate by pre-round margin@10 quartile, pooled over rounds/clients."""
    labels = round_labels(rep)
    seq = ["frozen"] + labels
    buckets = defaultdict(lambda: [0, 0])          # quartile -> [exits, at-risk]
    by_sign = defaultdict(lambda: [0, 0])          # loo sign -> [exits, at-risk]
    per_client_rows = []
    for client in rep["rounds"]["frozen"]:
        for a, b in zip(seq[:-1], seq[1:]):
            prev = rep["rounds"][a].get(client, {})
            curr = rep["rounds"][b].get(client, {})
            ex = exits_between(prev, curr)
            margins = [m for _, _, _, m in ex if m is not None]
            if len(margins) < 8:
                continue
            qs = np.quantile(margins, [0.25, 0.5, 0.75])
            loo = rep["loo_alignment"].get(b, {}).get(client)
            sign = "neg" if (loo is not None and loo < 0) else "nonneg"
            for _, n_in, n_out, m in ex:
                if m is None:
                    continue
                q = int(np.searchsorted(qs, m, side="right"))   # 0..3
                buckets[q][0] += n_out; buckets[q][1] += n_in
                by_sign[sign][0] += n_out; by_sign[sign][1] += n_in
            per_client_rows.append((client, b, loo,
                                    sum(n_out for _, _, n_out, _ in ex),
                                    sum(n_in for _, n_in, _, _ in ex)))
    return buckets, by_sign, per_client_rows


def main(paths):
    for path in paths:
        rep = load(path)
        fid = rep["fidelity"]
        print(f"\n== {rep['run']}  (weight_by={rep['weight_by']}; fidelity worst "
              f"|diff| {fid['worst_abs_diff']:.2e}, {'PASS' if fid['passed'] else 'FAIL'})")
        buckets, by_sign, rows = quartile_table(rep)
        print("  top-10 EXIT rate by pre-round margin@10 quartile (Q1 = smallest margins):")
        for q in range(4):
            ex, at = buckets[q]
            rate = ex / at if at else float("nan")
            print(f"    Q{q+1}: {ex:5d} / {at:5d} = {rate:.3f}")
        q1 = buckets[0][0] / buckets[0][1] if buckets[0][1] else float("nan")
        q4 = buckets[3][0] / buckets[3][1] if buckets[3][1] else float("nan")
        print(f"  Q1/Q4 exit-rate ratio: {q1 / q4 if q4 else float('nan'):.2f}"
              "  (prediction: >> 1; ~1 falsifies the margin mechanism)")
        print("  exit rate by client leave-one-out alignment sign:")
        for sign in ("neg", "nonneg"):
            ex, at = by_sign[sign]
            print(f"    {sign:6s}: {ex:5d} / {at:5d} = {(ex / at if at else float('nan')):.3f}")
        print("  per client/round: loo alignment, exits / at-risk")
        for client, label, loo, ex, at in rows:
            print(f"    {client:9s} {label:8s} loo={loo if loo is None else round(loo, 3)!s:>7} "
                  f"exits {ex:4d}/{at:4d}")


if __name__ == "__main__":
    main(sys.argv[1:])
