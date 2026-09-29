"""Block 18 on LoTTE (registration section 18): the floor's hypotheses M1 to M3 against block
L1's replay runs, the strong recipes' T1 and T2, the floor on the strong recipe M4, and server
averaging A1, recomputed from the per-query records with t1_report's functions.

usage: python l18_report.py <L1 out dir> <block 18 dir> <report dir>
"""
import os
import sys

import numpy as np

import t1_report as t1

D, FLOOR, HARD, HN, FLOORHN, AVERAGE = ("D", "floor", "hard", "hn", "floorhn", "average")
ARMS = [D, FLOOR, HARD, HN, FLOORHN, AVERAGE]
PAIRS = [(FLOOR, D), (HARD, D), (HN, D), (FLOORHN, HN), (AVERAGE, D)]
RUNS_PER_ARM = 6
TOLERANCE = 0.005


def wins(per_run, arm, base, measure, lower):
    """The paired runs in which ``arm`` is strictly lower (or strictly higher) than ``base``."""
    keys = [k for k in per_run if k[0] == arm]
    sign = -1 if lower else 1
    return sum(sign * (per_run[k][measure] - per_run[(base,) + k[1:]][measure]) > 0 for k in keys)


def mean(per_run, arm, measure):
    return float(np.mean(t1.arm_values(per_run, arm, measure)))


def hypotheses(per_run):
    need = RUNS_PER_ARM - 1
    floor = {"M1 lower G": wins(per_run, FLOOR, D, "G", True),
             "M2 higher end nDCG": wins(per_run, FLOOR, D, "end_ndcg_earlier", False),
             "M3 A": [mean(per_run, FLOOR, "A"), mean(per_run, D, "A")]}
    floor["passes"] = (floor["M1 lower G"] >= need and floor["M2 higher end nDCG"] >= need
                       and floor["M3 A"][0] >= floor["M3 A"][1] - TOLERANCE)
    result = {"M1-M3 floor vs D": floor}
    for name, arm in (("T1 hard vs D", HARD), ("T2 hn vs D", HN)):
        thesis = {"higher end nDCG": wins(per_run, arm, D, "end_ndcg_earlier", False),
                  "higher G": wins(per_run, arm, D, "G", False)}
        thesis["passes"] = thesis["higher end nDCG"] >= need and thesis["higher G"] >= need
        result[name] = thesis
    strong = {"lower G": wins(per_run, FLOORHN, HN, "G", True),
              "end nDCG": [mean(per_run, FLOORHN, "end_ndcg_earlier"),
                           mean(per_run, HN, "end_ndcg_earlier")],
              "A": [mean(per_run, FLOORHN, "A"), mean(per_run, HN, "A")]}
    strong["passes"] = (strong["lower G"] >= need
                        and strong["end nDCG"][0] >= strong["end nDCG"][1] - TOLERANCE
                        and strong["A"][0] >= strong["A"][1] - TOLERANCE)
    result["M4 floorhn vs hn"] = strong
    average = {"lower G": wins(per_run, AVERAGE, D, "G", True),
               "higher end nDCG": wins(per_run, AVERAGE, D, "end_ndcg_earlier", False)}
    average["passes"] = average["lower G"] >= need and average["higher end nDCG"] >= need
    result["A1 average vs D"] = average
    return result


def main(l1_dir, block_dir, report_dir):
    runs = {(D,) + k[1:]: r for k, r in t1.load(l1_dir, prefix="l1").items()
            if k[0] == "fedavg-replay"}
    runs.update(t1.load(block_dir, prefix="l18"))
    counts = {arm: sum(k[0] == arm for k in runs) for arm in ARMS}
    if any(n != RUNS_PER_ARM for n in counts.values()):
        raise SystemExit(f"the block is incomplete: {counts}")
    per_run, problems, cell_rows, losses = {}, [], [], {}
    for key, r in sorted(runs.items()):
        acq, hist = t1.recompute(r)
        problems += t1.sanity(key, r, acq, hist)
        per_run[key] = t1.run_summary(r, acq, hist)
        losses.setdefault(key[0], []).extend(h["losses"] for h in hist)
        cell_rows += [{"arm": key[0], "schedule": key[1], "seed": key[2],
                       **{k: v for k, v in h.items() if k != "losses"}} for h in hist]
    pooled = {arm: np.concatenate(v) for arm, v in losses.items()}
    shares = [f"- {arm}: losing at least 0.010 {np.mean(pooled[arm] >= t1.LOSS):.1%}, at least"
              f" 0.1 {np.mean(pooled[arm] >= t1.SEVERE_LOSS):.1%}" for arm in ARMS]
    lines = ["# Block 18 on LoTTE: recomputed report", "",
             f"Sanity problems: {len(problems)}", *[f"- {p}" for p in problems], "",
             "## Hypotheses (registration 18)", "",
             *[f"- {name}: {value}" for name, value in hypotheses(per_run).items()], "",
             "## Pooled shares of losing pairs", "", *shares, "",
             *t1.arms_section(per_run, ARMS, "All arms (mean over the six runs)"),
             *t1.runs_section(per_run, ARMS), *t1.paired_section(per_run, PAIRS),
             *t1.client_age_section(cell_rows, ARMS)]
    os.makedirs(report_dir, exist_ok=True)
    t1.write_cells(cell_rows, os.path.join(report_dir, "cells.csv"))
    with open(os.path.join(report_dir, "report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
