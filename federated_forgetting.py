import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from peft import (LoraConfig, TaskType,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics, check_lora_targets, LORA_TARGETS)


def get_adapter_state(model):
    inner = model[0].auto_model
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(inner).items()}


def set_adapter_state(model, state):
    inner = model[0].auto_model
    set_peft_model_state_dict(inner, state)


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


def new_model(model_name, model_path, lora_rank, fp16):
    model = SentenceTransformer(model_path, trust_remote_code=True)
    if fp16:
        model.half()
    check_lora_targets(model, model_name)
    cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                     r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1,
                     target_modules=LORA_TARGETS)
    model.add_adapter(cfg)
    model[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def client_train(model, global_state, data, q_prefix, d_prefix,
                 epochs, batch_size, lr, name):
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
        loss_fn = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss_fn)], epochs=epochs,
                  optimizer_params={"lr": lr},
                  warmup_steps=max(1, int(0.1 * len(loader))),
                  show_progress_bar=False, use_amp=True)
        num_steps = len(loader) * epochs
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
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--weight_by", choices=["examples", "corpus"], default="examples",
                    help="FedAvg weighting basis when --weighted: 'examples' = "
                         "local training-pair count (canonical FedAvg n_k), "
                         "'corpus' = client corpus size (the original F4 run)")
    ap.add_argument("--save_states", action="store_true",
                    help="save per-client + global adapter states each round "
                         "(needed for mechanism diagnostics, e.g. principal "
                         "angles between client updates)")
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}
    model_path, q_prefix, d_prefix, fp16 = resolve_local(args.model)

    model = new_model(args.model, model_path, args.lora_rank, fp16)
    global_state = get_adapter_state(model)

    tag = "weighted" if args.weighted else "unweighted"
    model_safe = args.model.replace("/", "_")
    jpath = os.path.join(args.out,
                         f"federated_{model_safe}_seed{args.seed}_{tag}.json")

    out = {"seed": args.seed, "slices": args.slices, "model": args.model,
           "metrics": args.metrics, "num_rounds": args.num_rounds,
           "weighted": args.weighted,
           "weight_by": args.weight_by if args.weighted else None,
           "split_fallback": [s for s in args.slices
                              if data[s].get("split_fallback")],
           "args": vars(args), "clients": {}, "R_matrix": {}, "BWT": None}
    R = out["R_matrix"]

    R["frozen"] = eval_global(model, global_state, data, args.slices,
                              q_prefix, d_prefix, args.metrics, args.batch_size)
    print_scores("frozen", R["frozen"], args.slices, args.metrics)
    dump_json(out, jpath)

    for rnd in range(args.num_rounds):
        print(f"  --- round {rnd+1}/{args.num_rounds} ---")
        states, n_examples, client_stats = [], [], {}
        for s in args.slices:
            st, n_ex, n_steps = client_train(model, global_state, data[s],
                                             q_prefix, d_prefix,
                                             args.local_epochs,
                                             args.batch_size, args.lr, name=s)
            states.append(st)
            n_examples.append(n_ex)
            client_stats[s] = {"num_examples": n_ex, "num_steps": n_steps}
        if args.weighted:
            weights = ([len(data[s]["corpus"]) for s in args.slices]
                       if args.weight_by == "corpus" else n_examples)
        else:
            weights = None
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
                               args.batch_size)
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
