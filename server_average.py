"""Server averaging (registration sections 17.4 and 17.5). At the end of each experience the
server deploys the average of the global models of that experience's last K rounds instead of
the last one (17.4), or, with --blend b, the exact running blend b * previous deployment +
(1 - b) * this experience's last global model, averaged as LoRA updates B A rather than as
separate factors (17.5). Training is unchanged, so a finished run's saved round states are
evaluated again under that deployment with the driver's own evaluation; K = 1 must reproduce
the run.

usage: python server_average.py --source <run dir> (--window K | --blend b) --data_root <dir>
           --out <dir>
"""
import argparse
import glob
import json
import os

import torch

import continual_driver as driver
import experiences
import t1_report
from aggregation_schemes import state_dict_sha256

REPRODUCTION_TOLERANCE = 1e-4


def average(states):
    """The element-wise mean of adapter states, in each tensor's own dtype."""
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype)
            for k in states[0]}


def stack(weighted):
    """One LoRA state whose update B A equals the weighted sum of the given states' updates,
    for a model of rank (states x rank) with the same alpha / r: the A factors stacked along
    the rank, the B factors scaled by their weights and stacked alongside."""
    out = {}
    for key in weighted[0][1]:
        if "lora_A" not in key and "lora_B" not in key:
            raise ValueError(f"cannot stack the non-LoRA tensor {key}")
        dim = 0 if "lora_A" in key else 1
        out[key] = torch.cat([s[key] * (w if dim == 1 else 1.0) for w, s in weighted], dim=dim)
    return out


def blend_weights(position, b):
    """The weight of each experience's last global model in the deployment at ``position``
    under d_0 = s_0, d_t = b d_(t-1) + (1 - b) s_t."""
    return [b ** position if s == 0 else b ** (position - s) * (1 - b)
            for s in range(position + 1)]


def round_states(source_dir, record, position, window):
    """The global states of the last ``window`` rounds at ``position``, each checked against
    the hash the run recorded for it."""
    rounds = [r for r in record["rounds"] if r["position"] == position]
    states = []
    for r in rounds[len(rounds) - window:]:
        payload = torch.load(os.path.join(source_dir, r["state_file"]), weights_only=True)
        if state_dict_sha256(payload["global"]) != r["hashes"]["global"]:
            raise SystemExit(f"{r['state_file']} does not match its recorded hash")
        states.append(payload["global"])
    return states


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the directory of a validated federated run")
    rule = ap.add_mutually_exclusive_group(required=True)
    rule.add_argument("--window", type=int)
    rule.add_argument("--blend", type=float)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    (path,) = glob.glob(os.path.join(args.source, "continual_*.json"))
    with open(path) as handle:
        source = json.load(handle)
    T, R, clients = (source["experiences_per_client"], source["rounds_per_experience"],
                     source["clients"])
    if source["arm"] in driver.LOCAL_ARMS + ("frozen",) or (
            args.window is not None and not 1 <= args.window <= R) or (
            args.blend is not None and not 0 < args.blend < 1):
        raise SystemExit(f"cannot average {source['arm']} with window {args.window} "
                         f"or blend {args.blend} over {R} rounds")
    if driver._sha256_file(source["manifest_path"]) != source["manifest_sha256"]:
        raise SystemExit("the manifest does not match the run's recorded digest")
    with open(source["manifest_path"]) as handle:
        manifest = json.load(handle)
    corpora = experiences.client_corpora(manifest, args.data_root, clients)
    cells = {c: {e: experiences.materialise(manifest, args.data_root, c, e, corpora[c])
                 for e in range(T)} for c in clients}
    name, batch = source["args"]["model"], source["args"]["eval_batch_size"]
    model_path, q_prefix, d_prefix, fp16 = driver.resolve_local(name)
    rank = source["args"]["lora_rank"]
    model, _ = driver.new_model(name, model_path, rank, fp16, "trainable-ab", grad_ckpt=False)
    last = [round_states(args.source, source, t, 1)[0] for t in range(T)] if args.blend else []
    out = {"arm": source["arm"] + "-average", "window": args.window, "blend": args.blend,
           "seed": source["seed"],
           "source_record": path, "source_sha256": driver._sha256_file(path),
           "manifest_path": source["manifest_path"], "manifest_sha256": source["manifest_sha256"],
           "clients": clients, "order": source["order"], "experiences_per_client": T,
           "rounds_per_experience": R, "commit": driver.get_git_commit(), "args": vars(args),
           "frozen": source["frozen"], "matrix": {}, "references": {c: {} for c in clients},
           "eval_pool": {}, "summary": {}}
    for t in range(T):
        if args.blend:
            state = stack(list(zip(blend_weights(t, args.blend), last[:t + 1])))
            model, _ = driver.new_model(name, model_path, rank * (t + 1), fp16, "trainable-ab",
                                        grad_ckpt=False)
        else:
            state = average(round_states(args.source, source, t, args.window))
        out["matrix"][str(t)] = {}
        for c in clients:
            order = source["order"][c]
            scored = driver.evaluate(model, state, corpora[c], cells[c],
                                     [order[s] for s in range(t + 1)], q_prefix, d_prefix, batch)
            out["matrix"][str(t)][c] = scored
            out["references"][c][str(order[t])] = {
                "position": t, "sha256": state_dict_sha256(state), "scores": scored[str(order[t])]}
    out["summary"] = driver.summarise(out)
    problems = t1_report.sanity(("average", args.window), out, *t1_report.recompute(out))
    if args.window == 1:
        out["reproduction"] = {k: out["summary"][k] - source["summary"][k] for k in ("A", "G")}
        problems += [f"window 1 misses the run's {k} by {d}" for k, d in out["reproduction"].items()
                     if abs(d) > REPRODUCTION_TOLERANCE]
    os.makedirs(args.out, exist_ok=True)
    tag = f"_average{args.window}" if args.window else f"_blend{args.blend}"
    jpath = os.path.join(args.out, os.path.basename(path)[:-5] + tag + ".json")
    driver.dump_json(out, jpath)
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
