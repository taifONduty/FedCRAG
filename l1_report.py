"""Block L1 on LoTTE (registration section 16 and its amendments): hypotheses H1 to H6 and the
reported comparisons, recomputed from the per-query records with t1_report's functions.

usage: python l1_report.py <L1 out dir> <report dir>
"""
import os
import sys

import numpy as np

import t1_report as t1

B, B0, C, D, E, F = ("local-replay", "local", "fedavg", "fedavg-replay",
                     "fedavg-replay-distill", "fedavg-replay-accept")
NO_SHIFT = "iid-fedavg-replay"
ARMS = [B0, B, C, D, E, F, NO_SHIFT]
REQUIRED = {B0, B, C, D, E, NO_SHIFT}
PAIRS = [(D, B), (C, B0), (D, C), (E, D), (F, D), (D, NO_SHIFT)]
RUNS_PER_ARM = 6
LOSS_SHARE = 0.10


def lower_in(per_run, shared, local, measure="G"):
    keys = [k for k in per_run if k[0] == shared]
    return sum(per_run[k][measure] < per_run[(local,) + k[1:]][measure] for k in keys)


def retention(per_run, shared, local):
    """H4 and H5: lower G in at least five of the six paired runs and a higher mean nDCG@10 on
    the earlier experience at the end of the stream."""
    wins = lower_in(per_run, shared, local)
    ends = [float(np.mean(t1.arm_values(per_run, arm, "end_ndcg_earlier")))
            for arm in (shared, local)]
    return {"lower G": wins, "end nDCG": ends,
            "passes": wins >= RUNS_PER_ARM - 1 and ends[0] > ends[1]}


def beyond_churn(per_run, losses):
    """H6: G under D above the no-shift control in at least five of the six paired runs, and
    a higher pooled share of losing pairs."""
    wins = lower_in(per_run, NO_SHIFT, D)
    shares = [float(np.mean(losses[arm] >= t1.LOSS)) for arm in (D, NO_SHIFT)]
    return {"control G lower": wins, "loss share": shares,
            "passes": wins >= RUNS_PER_ARM - 1 and shares[0] > shares[1]}


def hypotheses(per_run, losses):
    """``losses``: each arm's per-query losses on earlier experiences, pooled over its runs."""
    bwt = float(np.mean(t1.arm_values(per_run, D, "bwt")))
    share = float(np.mean(losses[D] >= t1.LOSS))
    return {"H1": t1.criterion(t1.arm_values(per_run, D, "A"), t1.A_THRESHOLD),
            "H2": t1.criterion(t1.arm_values(per_run, D, "G"), t1.G_THRESHOLD),
            "H3": {"bwt": bwt, "loss share": share, "passes": bwt > 0 and share >= LOSS_SHARE},
            "H4": retention(per_run, D, B),
            "H5": retention(per_run, C, B0),
            "H6": beyond_churn(per_run, losses)}


def main(out_dir, report_dir):
    per_run, problems, cell_rows, losses = {}, [], [], {}
    for key, r in sorted(t1.load(out_dir, prefix="l1").items()):
        acq, hist = t1.recompute(r)
        problems += t1.sanity(key, r, acq, hist)
        per_run[key] = t1.run_summary(r, acq, hist)
        losses.setdefault(key[0], []).extend(h["losses"] for h in hist)
        cell_rows += [{"arm": key[0], "schedule": key[1], "seed": key[2],
                       **{k: v for k, v in h.items() if k != "losses"}} for h in hist]
    counts = {arm: sum(k[0] == arm for k in per_run) for arm in ARMS}
    arms = [arm for arm in ARMS if counts[arm]]
    if any(counts[arm] != RUNS_PER_ARM for arm in arms) or not REQUIRED <= set(arms):
        raise SystemExit(f"the block is incomplete: {counts}")
    result = hypotheses(per_run, {arm: np.concatenate(v) for arm, v in losses.items()})
    lines = ["# Block L1 on LoTTE: recomputed report", "",
             f"Sanity problems: {len(problems)}", *[f"- {p}" for p in problems], "",
             "## Hypotheses (registration 16)", "",
             *[f"- {name}: {value}" for name, value in result.items()], "",
             *t1.arms_section(per_run, arms, "All arms (mean over the six runs)"),
             *t1.runs_section(per_run, arms),
             *t1.paired_section(per_run, [p for p in PAIRS if p[0] in arms and p[1] in arms]),
             *t1.client_age_section(cell_rows, arms)]
    os.makedirs(report_dir, exist_ok=True)
    t1.write_cells(cell_rows, os.path.join(report_dir, "cells.csv"))
    with open(os.path.join(report_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
