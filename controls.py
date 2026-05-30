import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from peft import LoraConfig, TaskType
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics)


def fresh_model(name, lora_rank):
    path, q_prefix, d_prefix, fp16 = resolve_local(name)
    model = SentenceTransformer(path, trust_remote_code=True)
    if fp16:
        model.half()
    cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                     r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1,
                     target_modules=["query", "key", "value", "dense"])
    model.add_adapter(cfg)
    model[0].auto_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, q_prefix, d_prefix


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


def train(model, examples, epochs, batch_size, lr):
    if len(examples) < batch_size:
        if not examples:
            return
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss_fn)], epochs=epochs,
              optimizer_params={"lr": lr},
              warmup_steps=max(1, int(0.1 * len(loader))),
              show_progress_bar=True, use_amp=True)


def evaluate(model, data, q_prefix, d_prefix, metrics, batch_size):
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

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}

    print("=== frozen ===")
    path, q_prefix, d_prefix, fp16 = resolve_local(args.model)
    fm = SentenceTransformer(path, trust_remote_code=True)
    if fp16:
        fm.half()
    frozen = {s: evaluate(fm, data[s], q_prefix, d_prefix, args.metrics, args.batch_size)
              for s in args.slices}
    for s in args.slices:
        print(f"  {s}: " + " ".join(f"{m}:{frozen[s][m]:.4f}" for m in args.metrics))
    del fm; torch.cuda.empty_cache()

    print("=== independent (per-slice ceiling) ===")
    independent = {}
    for s in args.slices:
        model, qp, dp = fresh_model(args.model, args.lora_rank)
        train(model, make_examples(data[s], qp, dp), args.epochs, args.batch_size, args.lr)
        independent[s] = evaluate(model, data[s], qp, dp, args.metrics, args.batch_size)
        print(f"  {s}: " + " ".join(f"{m}:{independent[s][m]:.4f}" for m in args.metrics))
        del model; torch.cuda.empty_cache()

    print("=== joint oracle (multi-task ceiling) ===")
    model, qp, dp = fresh_model(args.model, args.lora_rank)
    all_ex = []
    for s in args.slices:
        all_ex.extend(make_examples(data[s], qp, dp))
    train(model, all_ex, args.epochs, args.batch_size, args.lr)
    joint = {s: evaluate(model, data[s], qp, dp, args.metrics, args.batch_size)
             for s in args.slices}
    for s in args.slices:
        print(f"  {s}: " + " ".join(f"{m}:{joint[s][m]:.4f}" for m in args.metrics))
    del model; torch.cuda.empty_cache()

    out = {"seed": args.seed, "slices": args.slices, "model": args.model,
           "metrics": args.metrics, "frozen": frozen,
           "independent": independent, "joint": joint}
    jpath = os.path.join(args.out, f"controls_{args.model.replace('/','_')}_seed{args.seed}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
