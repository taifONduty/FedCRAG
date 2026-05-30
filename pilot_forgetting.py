import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, InputExample
from peft import LoraConfig, TaskType
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics)
from sentence_transformers import SentenceTransformer


def build_eval(model, data, q_prefix, d_prefix, metrics, batch_size):
    corpus = data["corpus"]
    cids = list(corpus.keys())
    qids = [q for q in data["eval_q"] if q in data["eval_qrels"] and data["eval_qrels"][q]]
    if not qids:
        return {m: float("nan") for m in metrics}
    c_emb = model.encode([d_prefix + doc_text(corpus[c]) for c in cids],
                         batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    q_emb = model.encode([q_prefix + data["eval_q"][q] for q in qids],
                         batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    return evaluate_metrics(cids, c_emb, qids, q_emb, data["eval_qrels"], metrics)


def train_on_slice(model, data, q_prefix, d_prefix, epochs, batch_size, lr):
    examples = []
    for qid, rels in data["train_qrels"].items():
        if qid not in data["train_q"]:
            continue
        for did, rel in rels.items():
            if rel > 0 and did in data["corpus"]:
                examples.append(InputExample(
                    texts=[q_prefix + data["train_q"][qid],
                           d_prefix + doc_text(data["corpus"][did])]))
    if len(examples) < batch_size:
        print(f"  warning: only {len(examples)} train pairs")
        if not examples:
            return
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              optimizer_params={"lr": lr},
              warmup_steps=max(1, int(0.1 * len(loader))),
              show_progress_bar=True, use_amp=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=["nfcorpus", "fiqa", "scifact", "arguana"])
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--metrics", nargs="+", default=["ndcg@10", "recall@10", "recall@100"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    order = args.slices[:]
    random.shuffle(order)
    print(f"seed {args.seed} | order: {order}")

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}

    path, q_prefix, d_prefix, fp16 = resolve_local(args.model)
    model = SentenceTransformer(path, trust_remote_code=True)
    if fp16:
        model.half()
    peft_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                          r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                          lora_dropout=0.1,
                          target_modules=["query", "key", "value", "dense"])
    model.add_adapter(peft_cfg)
    model[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    primary = args.metrics[0]
    R = {}

    def eval_all(label):
        R[label] = {}
        for s in args.slices:
            R[label][s] = build_eval(model, data[s], q_prefix, d_prefix,
                                     args.metrics, args.batch_size)
        print(f"  [{label}] " +
              " ".join(f"{s}:{R[label][s][primary]:.4f}" for s in args.slices))

    print("=== frozen baseline ===")
    eval_all("frozen")
    torch.cuda.empty_cache()

    for i, s in enumerate(order):
        print(f"=== stage {i}: train on '{s}' ===")
        train_on_slice(model, data[s], q_prefix, d_prefix,
                       args.epochs, args.batch_size, args.lr)
        eval_all(f"after_{i}_{s}")

    forget = {}
    for m in args.metrics:
        terms = []
        for i, s in enumerate(order[:-1]):
            peak = max(R[f"after_{j}_{order[j]}"][s][m]
                       for j in range(i, len(order))
                       if not np.isnan(R[f"after_{j}_{order[j]}"][s][m]))
            final = R[f"after_{len(order)-1}_{order[-1]}"][s][m]
            terms.append(peak - final)
        forget[m] = float(np.mean(terms)) if terms else float("nan")
        print(f">>> Forgetting[{m}] = {forget[m]:+.4f}")

    bwt = {}
    for m in args.metrics:
        terms = []
        final_label = f"after_{len(order)-1}_{order[-1]}"
        for i, s in enumerate(order[:-1]):
            learned = R[f"after_{i}_{s}"][s][m]
            final = R[final_label][s][m]
            terms.append(final - learned)
        bwt[m] = float(np.mean(terms)) if terms else float("nan")
        print(f">>> BWT[{m}] = {bwt[m]:+.4f}")

    out = {"seed": args.seed, "order": order, "slices": args.slices,
           "model": args.model, "metrics": args.metrics,
           "R_matrix": R, "forgetting": forget, "BWT": bwt}
    jpath = os.path.join(args.out, f"pilot_{args.model.replace('/','_')}_seed{args.seed}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
