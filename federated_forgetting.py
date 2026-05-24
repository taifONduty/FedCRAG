import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from peft import (LoraConfig, TaskType,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics)


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


def new_model(model_path, lora_rank, fp16):
    model = SentenceTransformer(model_path, trust_remote_code=True)
    if fp16:
        model.half()
    cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                     r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1,
                     target_modules=["query", "key", "value", "dense"])
    model.add_adapter(cfg)
    return model


def client_train(global_state, model_path, lora_rank, fp16, data,
                 q_prefix, d_prefix, epochs, batch_size, lr):
    model = new_model(model_path, lora_rank, fp16)
    set_adapter_state(model, global_state)
    examples = make_examples(data, q_prefix, d_prefix)
    if examples and len(examples) >= batch_size:
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
        loss_fn = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss_fn)], epochs=epochs,
                  optimizer_params={"lr": lr},
                  warmup_steps=max(1, int(0.1 * len(loader))),
                  show_progress_bar=False)
    state = get_adapter_state(model)
    del model; torch.cuda.empty_cache()
    return state


def eval_global(global_state, model_path, lora_rank, fp16, data, slices,
                q_prefix, d_prefix, metrics, batch_size):
    model = new_model(model_path, lora_rank, fp16)
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
    del model; torch.cuda.empty_cache()
    return scores


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
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}
    model_path, q_prefix, d_prefix, fp16 = resolve_local(args.model)
    primary = args.metrics[0]

    init = new_model(model_path, args.lora_rank, fp16)
    global_state = get_adapter_state(init)
    del init; torch.cuda.empty_cache()

    R = {}
    R["frozen"] = eval_global(global_state, model_path, args.lora_rank, fp16,
                              data, args.slices, q_prefix, d_prefix,
                              args.metrics, args.batch_size)
    print("  [frozen] " + " ".join(f"{s}:{R['frozen'][s][primary]:.4f}" for s in args.slices))

    for rnd in range(args.num_rounds):
        print(f"  --- round {rnd+1}/{args.num_rounds} ---")
        states, sizes = [], []
        for s in args.slices:
            st = client_train(global_state, model_path, args.lora_rank, fp16,
                              data[s], q_prefix, d_prefix,
                              args.local_epochs, args.batch_size, args.lr)
            states.append(st)
            sizes.append(len(data[s]["corpus"]))
        weights = sizes if args.weighted else None
        global_state = fedavg(states, weights=weights)
        label = f"round_{rnd+1}"
        R[label] = eval_global(global_state, model_path, args.lora_rank, fp16,
                               data, args.slices, q_prefix, d_prefix,
                               args.metrics, args.batch_size)
        print(f"  [{label}] " + " ".join(f"{s}:{R[label][s][primary]:.4f}" for s in args.slices))

    bwt = {}
    anchor = "round_1"
    final = f"round_{args.num_rounds}"
    for m in args.metrics:
        terms = [R[final][s][m] - R[anchor][s][m] for s in args.slices]
        bwt[m] = float(np.mean(terms))
        print(f">>> Federated BWT[{m}] (round1->final) = {bwt[m]:+.4f}")

    out = {"seed": args.seed, "slices": args.slices, "model": args.model,
           "metrics": args.metrics, "num_rounds": args.num_rounds,
           "weighted": args.weighted, "R_matrix": R, "BWT": bwt}
    jpath = os.path.join(args.out, f"federated_{args.model.replace('/','_')}_seed{args.seed}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
