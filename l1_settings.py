"""Settings of the LoTTE block chosen by the rules of registration section 15: arm E's lambda
and whether arm F enters.

usage: python l1_settings.py <T1 out dir> <development study out dir>
"""
import datetime
import glob
import json
import os
import sys

import numpy as np

F_DEADLINE = datetime.datetime(2026, 9, 30, 23, 59, tzinfo=datetime.timezone.utc)
DISTILL_RUNS = {0.1: ("dev", "dev-fedavg-replay-distill-lam0.1-A-s123"),
                0.25: ("dev", "dev-fedavg-replay-distill-lam0.25-A-s123"),
                2.0: ("t1", "pilot-fedavg-replay-distill-A-s123")}
ACCEPT_RUN = "dev-fedavg-replay-accept-A-s123"


def _record(run_dir):
    if not os.path.exists(os.path.join(run_dir, ".validated")):
        raise SystemExit(f"{run_dir} has not validated")
    with open(glob.glob(os.path.join(run_dir, "continual_*.json"))[0]) as handle:
        return json.load(handle)


def earlier_guard_ndcg(record):
    """Mean nDCG@10 on the guard queries of every experience but the last, after the last."""
    end = str(record["experiences_per_client"] - 1)
    return float(np.mean([record["matrix"][end][c][str(e)]["guard"]["ndcg@10"]
                          for c in record["clients"] for e in record["order"][c][:-1]]))


def choose_lambda(scores):
    """The highest score; a tie goes to the smaller lambda."""
    return max(sorted(scores), key=lambda lam: scores[lam])


def accept_enters(run_dir):
    marker = os.path.join(run_dir, ".validated")
    if not os.path.exists(marker):
        return False, None
    at = datetime.datetime.fromtimestamp(os.path.getmtime(marker), datetime.timezone.utc)
    return at <= F_DEADLINE, at.isoformat()


def main(t1_dir, dev_dir):
    roots = {"t1": t1_dir, "dev": dev_dir}
    scores = {lam: earlier_guard_ndcg(_record(os.path.join(roots[where], name)))
              for lam, (where, name) in DISTILL_RUNS.items()}
    enters, validated_at = accept_enters(os.path.join(dev_dir, ACCEPT_RUN))
    print(json.dumps({"lambda": choose_lambda(scores),
                      "earlier_guard_ndcg": {str(k): v for k, v in scores.items()},
                      "accept": enters, "accept_validated_at": validated_at}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
