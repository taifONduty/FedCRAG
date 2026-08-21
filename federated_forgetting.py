import os
import json
import argparse
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from peft import (LoraConfig, TaskType,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics, check_lora_targets, LORA_TARGETS,
                            amp_enabled, get_git_commit)


def get_adapter_state(model):
    inner = model[0].auto_model
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(inner).items()}


def set_adapter_state(model, state):
    inner = model[0].auto_model
    set_peft_model_state_dict(inner, state)


def _module_pairs(state):
    mods = {}
    for k, v in state.items():
        m = re.match(r"(.*)\.lora_(A|B)\.weight$", k)
        if m:
            mods.setdefault(m.group(1), {})[m.group(2)] = v.float()
    return {n: (ab["A"], ab["B"]) for n, ab in mods.items()
            if "A" in ab and "B" in ab}


def _stack_ip(s1, s2):
    # <sum ci Bi Ai, sum cj Bj Aj>_F via trace identity — r x r ops only,
    # never materializing d_out x d_in products (mechanism_suite.py convention)
    tot = 0.0
    for c1, B1, A1 in s1:
        for c2, B2, A2 in s2:
            tot += c1 * c2 * torch.sum((B1.T @ B2) * (A2 @ A1.T)).item()
    return tot


def maxmin_weights(client_states, broadcast_state):
    # Conflict-aware aggregation weights: cosine Gram of weight-space client
    # updates (dW_k = B_k A_k - B_g A_g), then the LP
    #   w* = argmax_{w in simplex} min_i (G w)_i.
    # Returns None (caller falls back to uniform) if the LP fails.
    from scipy.optimize import linprog
    prev = _module_pairs(broadcast_state)
    stacks = []
    for st in client_states:
        mp = _module_pairs(st)
        stacks.append({n: [(1.0, B, A)]
                       + ([(-1.0, prev[n][1], prev[n][0])] if n in prev else [])
                       for n, (A, B) in mp.items()})
    K = len(stacks)
    G = np.zeros((K, K))
    for a in range(K):
        for b in range(a, K):
            ip = sum(_stack_ip(stacks[a][n], stacks[b][n])
                     for n in stacks[a] if n in stacks[b])
            G[a, b] = G[b, a] = ip
    norms = np.sqrt(np.clip(np.diag(G), 1e-24, None))
    Gc = G / np.outer(norms, norms)
    c = np.zeros(K + 1); c[-1] = -1.0
    res = linprog(c, A_ub=np.hstack([-Gc, np.ones((K, 1))]), b_ub=np.zeros(K),
                  A_eq=[[1.0] * K + [0.0]], b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(None, None)], method="highs")
    if not res.success:
        print("  WARNING: max-min LP failed; falling back to uniform weights")
        return None
    return [float(x) for x in res.x[:K]]


def fedavg(states, weights=None):
    if weights is None:
        weights = [1.0] * len(states)
    total = sum(weights)
    if total <= 0:
        print("  WARNING: all FedAvg weights are zero (no client trained?); "
              "falling back to uniform averaging")
        weights = [1.0] * len(states)
        total = float(len(states))
    weights = [w / total for w in weights]
    avg = {}
    for key in states[0]:
        avg[key] = sum(w * s[key].float() for w, s in zip(weights, states))
    return avg


def make_examples(data, q_prefix, d_prefix):
    ex = []
    for qid, rels in data["train_qrels"].items():
        if qid not in data["train_q"]:
            continue
        for did, rel in rels.items():
            if rel > 0 and did in data["corpus"]:
                ex.append(InputExample(texts=[q_prefix + data["train_q"][qid],
                                              d_prefix + doc_text(data["corpus"][did])]))
    return ex


def new_model(model_name, model_path, lora_rank, fp16, grad_ckpt=True):
    model = SentenceTransformer(model_path, trust_remote_code=True)
    if fp16:
        model.half()
    check_lora_targets(model, model_name)
    cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                     r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1,
                     target_modules=LORA_TARGETS)
    model.add_adapter(cfg)
    if grad_ckpt:
        model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def client_train(model, global_state, data, q_prefix, d_prefix,
                 epochs, batch_size, lr, name, max_steps=0):
    # max_steps > 0 caps local optimization steps per round (standard FL
    # practice: fixed local work per round rather than full epochs). Without
    # a cap, the largest client (nfcorpus, ~110k pairs) trains a full 3.4k-step
    # epoch EVERY round, which dominates wall-clock and makes R=15 x 3 seeds
    # x 3 weightings infeasible on one GPU. Aggregation weights (n_k) still
    # use example counts, so client size asymmetry is preserved where the
    # weighting arms need it. Convention introduced 2026-08-18; the May pilot
    # (bge-m3, R=5) trained full epochs per round.
    set_adapter_state(model, global_state)
    examples = make_examples(data, q_prefix, d_prefix)
    num_steps = 0
    if not examples:
        print(f"  WARNING: client '{name}' has 0 training pairs -> "
              f"NO local training this round (state = incoming global state)")
    else:
        if len(examples) < batch_size:
            bs = len(examples)
            print(f"  WARNING: client '{name}' has only {len(examples)} train "
                  f"pairs (< batch_size {batch_size}); shrinking batch_size to {bs}")
            loader = DataLoader(examples, shuffle=True, batch_size=bs,
                                drop_last=False)
        else:
            # avoids in-batch false negatives when a query has several positives
            loader = NoDuplicatesDataLoader(examples, batch_size=batch_size)
        if max_steps > 0 and epochs != 1:
            raise ValueError(
                "max_steps_per_round requires local_epochs=1: sentence-"
                "transformers silently drops steps_per_epoch when epochs > 1 "
                "(fit_mixin), which would un-cap the round unnoticed.")
        steps_per_epoch = (min(len(loader), max_steps) if max_steps > 0
                           else len(loader))
        loss_fn = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss_fn)], epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  optimizer_params={"lr": lr},
                  warmup_steps=max(1, int(0.1 * steps_per_epoch)),
                  show_progress_bar=False, use_amp=amp_enabled())
        num_steps = steps_per_epoch * epochs
    return get_adapter_state(model), len(examples), num_steps


def eval_global(model, global_state, data, slices, q_prefix, d_prefix,
                metrics, batch_size):
    set_adapter_state(model, global_state)
    scores = {}
    for s in slices:
        corpus = data[s]["corpus"]
        cids = list(corpus.keys())
        qids = [q for q in data[s]["eval_q"]
                if q in data[s]["eval_qrels"] and data[s]["eval_qrels"][q]]
        if not qids:
            scores[s] = {m: float("nan") for m in metrics}
            continue
        c_emb = model.encode([d_prefix + doc_text(corpus[c]) for c in cids],
                             batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([q_prefix + data[s]["eval_q"][q] for q in qids],
                             batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        scores[s] = evaluate_metrics(cids, c_emb, qids, q_emb,
                                     data[s]["eval_qrels"], metrics)
    return scores


def print_scores(label, scores, slices, metrics):
    for s in slices:
        print(f"  [{label}] {s}: "
              + " ".join(f"{m}:{scores[s][m]:.4f}" for m in metrics))


def dump_json(out, jpath):
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, jpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=["nfcorpus", "fiqa", "scifact", "arguana"])
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--metrics", nargs="+", default=["ndcg@10", "recall@10", "recall@100"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=128,
                    help="encode batch for evaluation only (no gradients; "
                         "affects speed, not results)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--weight_by", choices=["examples", "corpus", "maxmin"],
                    default="examples",
                    help="FedAvg weighting basis when --weighted: 'examples' = "
                         "local training-pair count (canonical FedAvg n_k), "
                         "'corpus' = client corpus size (the original F4 run), "
                         "'maxmin' = conflict-aware per-round LP over the "
                         "cosine Gram of client weight-space updates (the "
                         "FedSpan repair; falls back to uniform on LP failure)")
    ap.add_argument("--save_states", action="store_true",
                    help="save per-client + global adapter states each round "
                         "(needed for mechanism diagnostics, e.g. principal "
                         "angles between client updates)")
    ap.add_argument("--max_steps_per_round", type=int, default=0,
                    help="cap on local optimization steps per client per round "
                         "(0 = full epoch, the May-pilot convention)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable gradient checkpointing (faster when VRAM "
                         "allows; no effect on results)")
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}
    model_path, q_prefix, d_prefix, fp16 = resolve_local(args.model)

    model = new_model(args.model, model_path, args.lora_rank, fp16,
                      grad_ckpt=not args.no_grad_ckpt)
    global_state = get_adapter_state(model)

    # tag must encode EVERY arm-distinguishing option: the May-2026 pilot lost
    # its unweighted per-round data to a filename collision, and weighted runs
    # with different --weight_by would collide again without the basis here.
    basis = f"weighted-{args.weight_by}" if args.weighted else "unweighted"
    tag = f"{basis}_r{args.num_rounds}"
    model_safe = args.model.replace("/", "_")
    jpath = os.path.join(args.out,
                         f"federated_{model_safe}_seed{args.seed}_{tag}.json")

    out = {"seed": args.seed, "slices": args.slices, "model": args.model,
           "metrics": args.metrics, "num_rounds": args.num_rounds,
           "weighted": args.weighted,
           "weight_by": args.weight_by if args.weighted else None,
           "split_fallback": [s for s in args.slices
                              if data[s].get("split_fallback")],
           "commit": get_git_commit(), "use_amp": amp_enabled(),
           "args": vars(args), "clients": {}, "R_matrix": {}, "BWT": None}
    R = out["R_matrix"]

    R["frozen"] = eval_global(model, global_state, data, args.slices,
                              q_prefix, d_prefix, args.metrics,
                              args.eval_batch_size)
    print_scores("frozen", R["frozen"], args.slices, args.metrics)
    dump_json(out, jpath)

    for rnd in range(args.num_rounds):
        print(f"  --- round {rnd+1}/{args.num_rounds} ---")
        states, n_examples, client_stats = [], [], {}
        for s in args.slices:
            st, n_ex, n_steps = client_train(model, global_state, data[s],
                                             q_prefix, d_prefix,
                                             args.local_epochs,
                                             args.batch_size, args.lr, name=s,
                                             max_steps=args.max_steps_per_round)
            states.append(st)
            n_examples.append(n_ex)
            client_stats[s] = {"num_examples": n_ex, "num_steps": n_steps}
        if args.weighted:
            if args.weight_by == "maxmin":
                weights = maxmin_weights(states, global_state)
            elif args.weight_by == "corpus":
                weights = [len(data[s]["corpus"]) for s in args.slices]
            else:
                weights = n_examples
        else:
            weights = None
        if args.weighted and args.weight_by == "maxmin":
            wn = weights if weights is not None else [1.0 / len(states)] * len(states)
            out.setdefault("round_weights", {})[f"round_{rnd+1}"] = [round(w, 4) for w in wn]
            print(f"  maxmin weights: {[round(w, 3) for w in wn]}")
        global_state = fedavg(states, weights=weights)
        label = f"round_{rnd+1}"
        out["clients"][label] = client_stats
        if args.save_states:
            spath = os.path.join(
                args.out,
                f"states_{model_safe}_seed{args.seed}_{tag}_round{rnd+1}.pt")
            torch.save({"clients": dict(zip(args.slices, states)),
                        "global": global_state}, spath)
            print(f"  saved adapter states -> {spath}")
        R[label] = eval_global(model, global_state, data, args.slices,
                               q_prefix, d_prefix, args.metrics,
                               args.eval_batch_size)
        print_scores(label, R[label], args.slices, args.metrics)
        dump_json(out, jpath)
        torch.cuda.empty_cache()

    bwt = {}
    anchor = "round_1"
    final = f"round_{args.num_rounds}"
    for m in args.metrics:
        terms = [R[final][s][m] - R[anchor][s][m] for s in args.slices]
        bwt[m] = float(np.mean(terms))
        print(f">>> Federated BWT[{m}] (round1->final) = {bwt[m]:+.4f}")

    out["BWT"] = bwt
    dump_json(out, jpath)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
