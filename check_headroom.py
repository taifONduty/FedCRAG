# Headroom gate (ROADMAP Decision D1 / W3' plan): a backbone passes iff
# independent fine-tuning beats the frozen model on EVERY slice — otherwise
# "forgetting" on that slice partly measures recipe-induced degradation, not
# lost gains (audit finding, 2026-08-17).
# Usage: python check_headroom.py results/controls_<model>_seed<seed>.json [metric]
import json
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: python check_headroom.py <controls_json> [metric=ndcg@10]")
        sys.exit(2)
    metric = sys.argv[2] if len(sys.argv) > 2 else "ndcg@10"
    with open(sys.argv[1]) as f:
        d = json.load(f)
    frozen, independent = d["frozen"], d["independent"]
    joint = d.get("joint", {})
    print(f"model={d.get('model')} seed={d.get('seed')} metric={metric} "
          f"commit={d.get('commit', 'n/a')}")
    print(f"{'slice':<12} {'frozen':>8} {'indep':>8} {'delta':>8} "
          f"{'joint':>8}  gate")
    failures = []
    for s in d["slices"]:
        fz, ind = frozen[s][metric], independent[s][metric]
        jt = joint.get(s, {}).get(metric, float("nan"))
        delta = ind - fz
        ok = delta > 0
        if not ok:
            failures.append(s)
        print(f"{s:<12} {fz:>8.4f} {ind:>8.4f} {delta:>+8.4f} "
              f"{jt:>8.4f}  {'PASS' if ok else 'FAIL'}")
    if failures:
        print(f"GATE FAIL: independent <= frozen on {failures} — this backbone "
              f"lacks fine-tuning headroom there; do not use it as primary.")
        sys.exit(1)
    print("GATE PASS: independent > frozen on all slices.")
    sys.exit(0)


if __name__ == "__main__":
    main()
