"""False negatives behind hard-negative replay's regression (registration section 17.8). For a
run whose retained queries trained against their references' floor sets, the sets are rebuilt
from the saved reference states, and each earlier test query at each later evaluation is
flagged when one of its relevant passages had served as a negative for a retained query by
then. The losses of flagged and unflagged queries are compared in that run and in a control
run on the same stream.

usage: python false_negatives.py --run <run dir> --control <run dir> --data_root <dir>
           --out <json>
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import anchors
import continual_driver as driver
import experiences
from aggregation_schemes import state_dict_sha256

THRESHOLDS = (0.010, 0.1)


def load(run_dir):
    (path,) = glob.glob(os.path.join(run_dir, "continual_*.json"))
    with open(path) as handle:
        return json.load(handle)


def reference_state(run_dir, record, position):
    """The global state at the end of ``position``, checked against its recorded hash."""
    last = [r for r in record["rounds"] if r["position"] == position][-1]
    payload = torch.load(os.path.join(run_dir, last["state_file"]), weights_only=True)
    if state_dict_sha256(payload["global"]) != last["hashes"]["global"]:
        raise SystemExit(f"{last['state_file']} does not match its recorded hash")
    return payload["global"]


def used_negatives(record, sets, client, T):
    """For each evaluation position v, the passages that served as negatives for the client's
    retained queries in the rounds of positions 1..v."""
    used, out = set(), {}
    for v in range(1, T):
        for r in record["rounds"]:
            if r["position"] == v:
                used |= {p for q in r["memory"][client]["replay"] for p in sets[q][1:]}
        out[v] = set(used)
    return out


def losses(record, client, e, position, v):
    order = record["order"][client]
    assert order[position] == e
    ref = record["references"][client][str(e)]["scores"]["test"]["per_query"]["ndcg@10"]
    cur = record["matrix"][str(v)][client][str(e)]["test"]["per_query"]["ndcg@10"]
    return {q: max(0.0, ref[q] - cur[q]) for q in ref}


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "mean_loss": float(values.mean()) if values.size else None,
            **{f"share_loss_ge_{t}": float((values >= t).mean()) if values.size else None
               for t in THRESHOLDS}}


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run whose retained queries used floor sets")
    ap.add_argument("--control", required=True, help="a run on the same stream and order")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run, control = load(args.run), load(args.control)
    if run["manifest_sha256"] != control["manifest_sha256"] or run["order"] != control["order"]:
        raise SystemExit("the control run is not on the same stream and order")
    with open(run["manifest_path"]) as handle:
        manifest = json.load(handle)
    clients, T, k = run["clients"], run["experiences_per_client"], run["args"]["anchor_k"]
    corpora = experiences.client_corpora(manifest, args.data_root, clients)
    name, batch = run["args"]["model"], run["args"]["eval_batch_size"]
    model_path, q_prefix, d_prefix, fp16 = driver.resolve_local(name)
    model, _ = driver.new_model(name, model_path, run["args"]["lora_rank"], fp16,
                                "trainable-ab", grad_ckpt=False)
    pairs = {"flagged": {"run": [], "control": []}, "unflagged": {"run": [], "control": []}}
    for c in clients:
        order = run["order"][c]
        cells = {e: experiences.materialise(manifest, args.data_root, c, e, corpora[c])
                 for e in order}
        sets = {}
        for p in range(T - 1):
            state = reference_state(args.run, run, p)
            encoded = driver.encode_corpus(model, state, corpora[c], d_prefix, batch)
            cell = cells[order[p]]
            qids = list(cell["train_q"])
            q_emb = driver._encode(model, state, [q_prefix + cell["train_q"][q] for q in qids],
                                   batch)
            made = anchors.rank_anchors(q_emb, qids, cell["train_qrels"], encoded[1],
                                        encoded[0], k)
            sets.update({q: a["floor_pids"] for q, a in made.items()})
        negatives = used_negatives(run, sets, c, T)
        for p in range(T - 1):
            e = order[p]
            qrels = cells[e]["test_qrels"]
            for v in range(p + 1, T):
                lost = {"run": losses(run, c, e, p, v), "control": losses(control, c, e, p, v)}
                for q in lost["run"]:
                    kind = ("flagged" if any(rel > 0 and pid in negatives[v]
                                             for pid, rel in qrels[q].items()) else "unflagged")
                    for which in ("run", "control"):
                        pairs[kind][which].append(lost[which][q])
    result = {"run": args.run, "control": args.control, "anchor_k": k,
              "commit": driver.get_git_commit(),
              **{kind: {which: summary(v) for which, v in by.items()}
                 for kind, by in pairs.items()}}
    driver.dump_json(result, args.out)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
