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


# Gram/LP/QP/delta-space weighting machinery lives in aggregation_schemes.py
# (pure math, unit-tested in tests/test_aggregation.py). NOTE: the corrected
# trace identity there fixes a cross-pairing in the campaign-era inline copy;
# measured effect on the shipped maxmin arms: weight shift <= 0.0025, gamma*
# shift <= 0.0023 (results/gram_bug_audit.json) — below the seed noise floor.
from aggregation_schemes import (afl_update, apply_delta_weights,
                                 fednova_delta_weights, maxmin_weights,
                                 mgda_weights, qffl_delta_weights, update_gram)


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


def estimate_client_losses(model, global_state, data_by_slice, slices,
                           q_prefix, d_prefix, sample, batch_size, rng_seed):
    """Per-client training loss at the BROADCAST point w^t (the evaluation
    point q-FedAvg and AFL prescribe: F_k(w^t), before local training).

    MNRL-consistent estimator: for a deterministic sample of up to ``sample``
    training pairs per client, loss = in-batch-negatives cross-entropy over
    cosine scores at the MNRL default scale of 20. No gradients; batched with
    the eval batch size. The sample is fixed per (seed, round) so all schemes
    see identical losses.
    """
    set_adapter_state(model, global_state)
    out = []
    for s in slices:
        examples = make_examples(data_by_slice[s], q_prefix, d_prefix)
        if not examples:
            out.append(0.0)
            continue
        rng = np.random.RandomState(rng_seed)
        idx = rng.permutation(len(examples))[:sample]
        losses = []
        with torch.no_grad():
            for lo in range(0, len(idx), batch_size):
                chunk = [examples[i] for i in idx[lo:lo + batch_size]]
                q = model.encode([e.texts[0] for e in chunk],
                                 batch_size=batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
                d = model.encode([e.texts[1] for e in chunk],
                                 batch_size=batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
                scores = 20.0 * (q @ d.T)
                target = torch.arange(len(chunk), device=scores.device)
                ce = torch.nn.functional.cross_entropy(scores, target,
                                                       reduction="sum")
                losses.append(float(ce))
        out.append(sum(losses) / max(len(idx), 1))
    return out


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
    ap.add_argument("--weight_by",
                    choices=["examples", "corpus", "maxmin",
                             "qffl", "afl", "mgda", "fednova"],
                    default="examples",
                    help="FedAvg weighting basis when --weighted: 'examples' = "
                         "local training-pair count (canonical FedAvg n_k), "
                         "'corpus' = client corpus size (the original F4 run), "
                         "'maxmin' = conflict-aware per-round LP over the "
                         "cosine Gram of client weight-space updates (the "
                         "FedSpan repair; falls back to uniform on LP failure); "
                         "E2 external baselines: 'qffl' = q-FedAvg "
                         "(arXiv:1905.10497, loss^q with h_k normalization), "
                         "'afl' = agnostic-FL multiplicative-weights ascent "
                         "(arXiv:1902.00146), 'mgda' = min-norm weights on the "
                         "raw update Gram (arXiv:1810.04650), 'fednova' = "
                         "tau-normalized averaging (arXiv:2007.07481)")
    ap.add_argument("--qffl_q", type=float, default=1.0,
                    help="q-FedAvg fairness exponent q (only with "
                         "--weight_by qffl)")
    ap.add_argument("--afl_eta", type=float, default=0.1,
                    help="AFL mixture-weight ascent step size (only with "
                         "--weight_by afl)")
    ap.add_argument("--loss_sample", type=int, default=2048,
                    help="max training pairs per client for the broadcast-"
                         "point loss estimate (qffl/afl only; deterministic "
                         "per round)")
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

    ADAPTIVE = ("maxmin", "qffl", "afl", "mgda", "fednova")
    afl_lam = [1.0 / len(args.slices)] * len(args.slices)

    for rnd in range(args.num_rounds):
        print(f"  --- round {rnd+1}/{args.num_rounds} ---")
        label = f"round_{rnd+1}"
        # qffl/afl evaluate F_k at the broadcast point, BEFORE local training
        losses = None
        if args.weighted and args.weight_by in ("qffl", "afl"):
            losses = estimate_client_losses(
                model, global_state, data, args.slices, q_prefix, d_prefix,
                args.loss_sample, args.eval_batch_size,
                rng_seed=args.seed * 1000 + rnd)
            out.setdefault("client_losses", {})[label] = \
                dict(zip(args.slices, [round(x, 5) for x in losses]))
            print(f"  broadcast-point losses: "
                  f"{[round(x, 3) for x in losses]}")
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
        # --- aggregation dispatch ---------------------------------------
        # simplex schemes feed fedavg; delta-space schemes (v need not sum
        # to 1) feed apply_delta_weights. n_k/corpus/uniform unchanged.
        delta_v = None
        if not args.weighted:
            weights = None
        elif args.weight_by == "maxmin":
            weights = maxmin_weights(states, global_state)
        elif args.weight_by == "mgda":
            weights = mgda_weights(states, global_state)
        elif args.weight_by == "afl":
            afl_lam = afl_update(afl_lam, losses, args.afl_eta)
            weights = afl_lam
        elif args.weight_by == "qffl":
            sq_norms = np.diag(update_gram(states, global_state)).tolist()
            delta_v = qffl_delta_weights(losses, sq_norms, q=args.qffl_q,
                                         L=1.0 / args.lr)
        elif args.weight_by == "fednova":
            delta_v = fednova_delta_weights(
                n_examples, [client_stats[s]["num_steps"]
                             for s in args.slices])
        elif args.weight_by == "corpus":
            weights = [len(data[s]["corpus"]) for s in args.slices]
        else:
            weights = n_examples
        if args.weighted and args.weight_by in ADAPTIVE:
            wn = (delta_v if delta_v is not None else
                  (weights if weights is not None
                   else [1.0 / len(states)] * len(states)))
            out.setdefault("round_weights", {})[label] = \
                [round(w, 5) for w in wn]
            if delta_v is not None:
                out["weight_space"] = "delta"
            print(f"  {args.weight_by} weights: {[round(w, 3) for w in wn]}")
        if delta_v is not None:
            global_state = apply_delta_weights(global_state, states, delta_v)
        else:
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
