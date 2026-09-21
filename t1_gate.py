"""The registered decision of block T1 (registration section 14) from validated run
summaries: A (acquisition) and G (positive-part regression) per run."""
import argparse
import glob
import json

import numpy as np

SEEDS = (123, 2024, 3407)
SCHEDULES = ("A", "B")
A_THRESHOLD, G_THRESHOLD = 0.020, 0.010


def criterion(values, threshold):
    """Passes if the mean clears the threshold and at least five of the six runs clear
    half of it."""
    values = [float(v) for v in values]
    return {"mean": float(np.mean(values)), "threshold": threshold,
            "runs_over_half": int(sum(v >= threshold / 2 for v in values)),
            "passes": bool(np.mean(values) >= threshold
                           and sum(v >= threshold / 2 for v in values) >= 5)}


def _by_arm(runs, arm, key):
    rows = {(r["seed"], r["schedule"]): r[key] for r in runs if r["arm"] == arm}
    expected = [(s, sch) for sch in SCHEDULES for s in SEEDS]
    if sorted(rows) != sorted(expected):
        raise ValueError(f"arm {arm}: expected the six seed-by-schedule runs, "
                         f"found {sorted(rows)}")
    return [rows[k] for k in expected]


def decide(runs):
    a_d, g_d = _by_arm(runs, "fedavg-replay", "A"), _by_arm(runs, "fedavg-replay", "G")
    a_b = _by_arm(runs, "local", "A")
    g2, g1 = criterion(a_d, A_THRESHOLD), criterion(g_d, G_THRESHOLD)
    g3 = {"A_D": float(np.mean(a_d)), "A_B": float(np.mean(a_b)),
          "passes": bool(np.mean(a_d) >= np.mean(a_b))}
    decision = {"G2": g2, "G1": g1, "G3": g3}
    if not g2["passes"]:
        decision["outcome"] = "recipe or stream at fault"
    elif not g1["passes"]:
        decision["outcome"] = "replay controls the regression"
    else:
        decision["outcome"] = "method arms"
    try:
        a_e, g_e = (_by_arm(runs, "fedavg-replay-distill", "A"),
                    _by_arm(runs, "fedavg-replay-distill", "G"))
    except ValueError:
        return decision
    decision["distillation_controls_regression"] = bool(
        np.mean(g_e) < 0.5 * np.mean(g_d) and np.mean(a_e) >= np.mean(a_d) - 0.005)
    decision["E"] = {"A": float(np.mean(a_e)), "G": float(np.mean(g_e))}
    return decision


def load_runs(out_dir):
    """One row per validated pilot run directory named pilot-<arm>-<schedule>-s<seed>."""
    rows = []
    for path in sorted(glob.glob(f"{out_dir}/pilot-*/continual_*.json")):
        directory = path.rsplit("/", 2)[-2]
        if not glob.glob(f"{out_dir}/{directory}/.validated"):
            continue
        # pilot-<arm>-<schedule>-s<seed>; only the arm contains hyphens, so read from the right
        parts = directory.split("-")
        arm, schedule, seed = "-".join(parts[1:-2]), parts[-2], parts[-1]
        with open(path) as handle:
            summary = json.load(handle)["summary"]
        rows.append({"arm": arm, "schedule": schedule, "seed": int(seed[1:]),
                     "A": summary["A"], "G": summary["G"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    print(json.dumps(decide(load_runs(args.out_dir)), indent=1))


if __name__ == "__main__":
    main()
