"""Block T1 pilot report, recomputed from the per-query records without importing the
production aggregation code (regression.py, continual_driver.summarise, t1_gate.py).

usage: python t1_report.py <T1 out dir> <report dir>
"""
import csv
import glob
import json
import os
import sys

import numpy as np

A_THRESHOLD, G_THRESHOLD, LOSS, SEVERE_LOSS = 0.020, 0.010, 0.010, 0.1
ARMS = ["frozen", "local", "fedavg", "fedavg-replay", "fedavg-replay-distill"]
TRAINED_ARMS = ARMS[1:]
PAIRS = [("fedavg-replay-distill", "fedavg-replay"), ("fedavg-replay", "local"),
         ("fedavg-replay", "fedavg"), ("fedavg", "local")]
TOL = 1e-12


def load(out_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "pilot-*", "continual_*.json"))):
        name = os.path.basename(os.path.dirname(path))
        rest, seed = name.rsplit("-s", 1)
        arm, schedule = rest[len("pilot-"):].rsplit("-", 1)
        with open(path) as f:
            record = json.load(f)
        with open(os.path.join(os.path.dirname(path), "wall_seconds")) as f:
            record["wall_seconds"] = int(f.read())
        runs[(arm, schedule, int(seed))] = record
    return runs


def scores(block, split="test"):
    return block[split]["per_query"]["ndcg@10"]


def mean_score(block, split="test"):
    return float(np.mean(list(scores(block, split).values())))


def recompute(r, split="test"):
    """Acquisition cells, and historical (client, experience, evaluation position) cells
    with the per-query losses behind them."""
    T, acq, hist = r["experiences_per_client"], [], []
    for c in r["clients"]:
        for t, e in enumerate(map(str, r["order"][c])):
            frozen = scores(r["frozen"][c][e], split)
            first = scores(r["matrix"][str(t)][c][e], split)
            acq.append(np.mean([first[q] - frozen[q] for q in frozen]))
            ref = scores(r["references"][c][e]["scores"], split)
            for v in range(t + 1, T):
                cur = scores(r["matrix"][str(v)][c][e], split)
                diff = np.array([cur[q] - ref[q] for q in ref])
                path = [scores(r["matrix"][str(u)][c][e], split) for u in range(t, v + 1)]
                hist.append({
                    "client": c, "experience": e, "position": t, "eval_position": v,
                    "age": v - t,
                    "regression": float(np.mean(np.maximum(-diff, 0))),
                    "improvement": float(np.mean(np.maximum(diff, 0))),
                    "bwt": float(diff.mean()),
                    "peak_forgetting": float(np.mean(
                        [max(p[q] for p in path) - cur[q] for q in cur])),
                    "reference_ndcg": float(np.mean(list(ref.values()))),
                    "current_ndcg": float(np.mean(list(cur.values()))),
                    "losses": -diff,
                    "final": v == T - 1})
    return acq, hist


def sanity(key, r, acq, hist):
    problems = []
    T = r["experiences_per_client"]
    for c in r["clients"]:
        for t, e in enumerate(map(str, r["order"][c])):
            ref = r["references"][c][e]
            if ref["position"] != t:
                problems.append(f"{key} {c}:{e} reference position {ref['position']} != {t}")
            for split in ("guard", "test"):
                if scores(ref["scores"], split) != scores(r["matrix"][str(t)][c][e], split):
                    problems.append(f"{key} {c}:{e} {split} reference != scores at its position")
                ids = set(scores(r["frozen"][c][e], split))
                if any(set(scores(r["matrix"][str(v)][c][e], split)) != ids
                       for v in range(t, T)):
                    problems.append(f"{key} {c}:{e} {split} query set differs over time")
    for h in hist:
        if abs(h["bwt"] - (h["improvement"] - h["regression"])) > TOL:
            problems.append(f"{key} signed change != improvement - regression at "
                            f"{h['client']}:{h['experience']}@{h['eval_position']}")
    if key[0] == "frozen" and any(h["regression"] or h["improvement"] for h in hist):
        problems.append(f"{key} frozen arm changed its scores")
    A, G = np.mean(acq), np.mean([h["regression"] for h in hist])
    if abs(A - r["summary"]["A"]) > 1e-9 or abs(G - r["summary"]["G"]) > 1e-9:
        problems.append(f"{key} recomputed A, G {A}, {G} != stored "
                        f"{r['summary']['A']}, {r['summary']['G']}")
    return problems


def run_summary(r, acq, hist):
    guard_acq, guard_hist = recompute(r, "guard")
    final = [h for h in hist if h["final"]]
    losses = np.concatenate([h["losses"] for h in hist])
    end = str(r["experiences_per_client"] - 1)
    worst = max(hist, key=lambda h: h["regression"])
    return {
        "A": np.mean(acq), "G": np.mean([h["regression"] for h in hist]),
        "A_guard": np.mean(guard_acq),
        "G_guard": np.mean([h["regression"] for h in guard_hist]),
        "bwt": np.mean([h["bwt"] for h in hist]),
        "peak_forgetting": np.mean([h["peak_forgetting"] for h in hist]),
        "reference_ndcg": np.mean([h["reference_ndcg"] for h in hist]),
        "current_ndcg_earlier": np.mean([h["current_ndcg"] for h in hist]),
        "end_ndcg_earlier": np.mean([h["current_ndcg"] for h in final]),
        "end_ndcg_all": np.mean([mean_score(r["matrix"][end][c][str(e)])
                                 for c in r["clients"] for e in r["order"][c]]),
        "pool_ndcg": np.mean([r["eval_pool"][c]["ndcg@10"] for c in r["clients"]]),
        "query_loss_rate": float(np.mean(losses >= LOSS)),
        "severe_loss_rate": float(np.mean(losses >= SEVERE_LOSS)),
        "cells_over": int(sum(h["regression"] >= G_THRESHOLD for h in hist)),
        "cells": len(hist),
        "worst": f"c{worst['client']} e{worst['experience']} @{worst['eval_position']}"
                 f" ({fmt(worst['regression'])})",
        "wall": r["wall_seconds"]}


def criterion(values, threshold):
    mean = float(np.mean(values))
    over_half = int(sum(v >= threshold / 2 for v in values))
    return {"mean": mean, "runs_over_half": over_half,
            "runs_over_full": int(sum(v >= threshold for v in values)),
            "passes": mean >= threshold and over_half >= len(values) - 1}


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def arm_values(per_run, arm, measure):
    return [per_run[k][measure] for k in sorted(per_run) if k[0] == arm]


def gate_section(per_run):
    a_d, g_d = arm_values(per_run, "fedavg-replay", "A"), arm_values(per_run, "fedavg-replay", "G")
    ad, gd = np.mean(a_d), np.mean(g_d)
    ae = np.mean(arm_values(per_run, "fedavg-replay-distill", "A"))
    ge = np.mean(arm_values(per_run, "fedavg-replay-distill", "G"))
    g2, g1 = criterion(a_d, A_THRESHOLD), criterion(g_d, G_THRESHOLD)
    outcome = ("recipe or stream at fault" if not g2["passes"] else
               "replay controls the regression" if not g1["passes"] else "method arms")
    rows = [f"| {label} | {threshold} | {fmt(c['mean'])} | {c['runs_over_half']}/{len(a_d)} |"
            f" {c['runs_over_full']}/{len(a_d)} | {c['passes']} |"
            for label, threshold, c in [("G2: A under replay", A_THRESHOLD, g2),
                                        ("G1: G under replay", G_THRESHOLD, g1)]]
    return ["## Gate (recomputed)", "",
            "| criterion | threshold | mean | runs >= half | runs >= full | passes |",
            "|---|---:|---:|---:|---:|---|", *rows, "",
            f"G3 (reported): A replay {ad:.6f} vs A local "
            f"{np.mean(arm_values(per_run, 'local', 'A')):.6f}",
            f"Outcome: {outcome}",
            f"Distillation criterion: G_E {ge:.6f} < 0.5 G_D {0.5 * gd:.6f}: {ge < 0.5 * gd};"
            f" A_E {ae:.6f} >= A_D - 0.005 {ad - 0.005:.6f}: {ae >= ad - 0.005}", ""]


def arms_section(per_run):
    cols = [("A", "A"), ("G", "G"), ("bwt", "BWT"), ("peak_forgetting", "peak forg."),
            ("reference_ndcg", "ref nDCG"), ("current_ndcg_earlier", "nDCG earlier"),
            ("end_ndcg_earlier", "end nDCG earlier"), ("end_ndcg_all", "end nDCG all"),
            ("pool_ndcg", "pool nDCG"), ("query_loss_rate", "q loss>=.01"),
            ("severe_loss_rate", "q loss>=.1"), ("wall", "wall s")]
    lines = ["## All arms (mean over runs; frozen is one seed per schedule)", "",
             "| arm | runs | " + " | ".join(c[1] for c in cols) + " |",
             "|---|---:|" + "---:|" * len(cols)]
    for arm in ARMS:
        means = {k: np.mean(arm_values(per_run, arm, k)) for k, _ in cols}
        lines.append(f"| {arm} | {len(arm_values(per_run, arm, 'A'))} | " + " | ".join(
            f"{means[k]:.0f}" if k == "wall" else fmt(means[k]) for k, _ in cols) + " |")
    return lines + ["", "nDCG earlier: mean over every (cell, later evaluation) of the current"
                    " score on an earlier experience's test queries. End nDCG earlier: the same"
                    " at the end of the stream only. End nDCG all: all four experiences at the"
                    " end. Pool nDCG: final model on the client's eval pool.", ""]


def runs_section(per_run):
    lines = ["## Six runs", "", "| arm | schedule | seed | A | G | A guard | G guard |"
             " end nDCG earlier | cells G>=.01 | worst cell |",
             "|---|---|---:|" + "---:|" * 6 + "---|"]
    for k in sorted(per_run, key=lambda k: (ARMS.index(k[0]), k[1], k[2])):
        p = per_run[k]
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {fmt(p['A'])} | {fmt(p['G'])} |"
                     f" {fmt(p['A_guard'])} | {fmt(p['G_guard'])} | {fmt(p['end_ndcg_earlier'])} |"
                     f" {p['cells_over']}/{p['cells']} | {p['worst']} |")
    return lines


def paired_section(per_run):
    measures = ("A", "G", "end_ndcg_earlier", "end_ndcg_all", "pool_ndcg")
    lines = ["", "## Paired differences, same schedule and seed", "",
             "| pair | dA | dG | d end nDCG earlier | d end nDCG all | d pool nDCG |"
             " runs with dG<0 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x, y in PAIRS:
        keys = [k for k in per_run if k[0] == x]
        diffs = {m: [per_run[k][m] - per_run[(y,) + k[1:]][m] for k in keys] for m in measures}
        lines.append(f"| {x} - {y} | " + " | ".join(fmt(np.mean(v)) for v in diffs.values())
                     + f" | {sum(v < 0 for v in diffs['G'])}/{len(keys)} |")
    return lines


def client_age_section(cell_rows):
    lines = ["", "## Regression by client and experience age (test, mean over the six runs)", ""]
    for arm in TRAINED_ARMS:
        rows = [c for c in cell_rows if c["arm"] == arm]
        ages = sorted({c["age"] for c in rows})
        lines += [f"### {arm}", "", "| client | " + " | ".join(f"age {a}" for a in ages)
                  + " | all | end nDCG earlier |", "|---|" + "---:|" * (len(ages) + 2)]
        for c in sorted({c["client"] for c in rows}, key=int):
            cr = [x for x in rows if x["client"] == c]
            lines.append(f"| {c} | " + " | ".join(
                fmt(np.mean([x["regression"] for x in cr if x["age"] == a])) for a in ages)
                + f" | {fmt(np.mean([x['regression'] for x in cr]))} |"
                f" {fmt(np.mean([x['current_ndcg'] for x in cr if x['final']]))} |")
        lines.append("")
    return lines


def write_cells(cell_rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0]))
        w.writeheader()
        w.writerows(cell_rows)


def main(out_dir, report_dir):
    per_run, problems, cell_rows = {}, [], []
    for key, r in sorted(load(out_dir).items()):
        acq, hist = recompute(r)
        problems += sanity(key, r, acq, hist)
        per_run[key] = run_summary(r, acq, hist)
        cell_rows += [{"arm": key[0], "schedule": key[1], "seed": key[2],
                       **{k: v for k, v in h.items() if k != "losses"}} for h in hist]
    lines = ["# Block T1 pilot: recomputed report", "",
             "Recomputed from per-query records without the production aggregation code.",
             f"Sanity problems: {len(problems)}", *[f"- {p}" for p in problems], "",
             *gate_section(per_run), *arms_section(per_run), *runs_section(per_run),
             *paired_section(per_run), *client_age_section(cell_rows)]
    os.makedirs(report_dir, exist_ok=True)
    write_cells(cell_rows, os.path.join(report_dir, "cells.csv"))
    with open(os.path.join(report_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
