import os
import json
import argparse
import time
import numpy as np
import torch
from fedcrag_common import (LOCAL_MODELS, API_MODELS, MODEL_SETS, resolve_local,
                            load_slice, doc_text, evaluate_metrics,
                            load_local_model, encode_texts, APIEmbedder)


def benchmark_local(model_name, slice_data, slices, metrics, out_dir,
                    corpus_batch, query_batch):
    path, q_prefix, d_prefix, fp16 = resolve_local(model_name)
    print(f"=== local: {model_name} ({path}) ===")
    model, q_prefix, d_prefix, fp16 = load_local_model(model_name)
    bs = 8 if fp16 else corpus_batch
    cache_dir = os.path.join(out_dir, "emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    res = {}
    for s in slices:
        corpus, queries, qrels = slice_data[s]
        cids = list(corpus.keys())
        safe = model_name.replace("/", "_")
        cache = os.path.join(cache_dir, f"{safe}__{s}.npy")
        c_emb = encode_texts(model, [doc_text(corpus[c]) for c in cids],
                             d_prefix, bs, cache)
        qids = [q for q in queries if q in qrels and qrels[q]]
        q_emb = encode_texts(model, [queries[q] for q in qids], q_prefix, query_batch)
        res[s] = evaluate_metrics(cids, c_emb, qids, q_emb, qrels, metrics)
        print("  " + s + ": " + " ".join(f"{m}:{res[s][m]:.4f}" for m in metrics))
    del model; torch.cuda.empty_cache()
    return res


def benchmark_api(model_name, slice_data, slices, metrics, out_dir,
                  corpus_batch, query_batch, api_key, base_url):
    print(f"=== api: {model_name} ({API_MODELS[model_name]['model']}) ===")
    embedder = APIEmbedder(model_name, api_key=api_key, base_url=base_url)
    cache_dir = os.path.join(out_dir, "emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    res = {}
    for s in slices:
        corpus, queries, qrels = slice_data[s]
        cids = list(corpus.keys())
        safe = model_name.replace("/", "_")
        cache = os.path.join(cache_dir, f"{safe}__{s}.npy")
        if os.path.exists(cache):
            c_emb = np.load(cache)
        else:
            c_emb = embedder.encode([doc_text(corpus[c]) for c in cids],
                                    batch_size=corpus_batch)
            np.save(cache, c_emb)
        qids = [q for q in queries if q in qrels and qrels[q]]
        q_emb = embedder.encode([queries[q] for q in qids], batch_size=query_batch)
        res[s] = evaluate_metrics(cids, c_emb, qids, q_emb, qrels, metrics)
        print("  " + s + ": " + " ".join(f"{m}:{res[s][m]:.4f}" for m in metrics))
    return res


def print_tables(results, models, slices, metrics):
    for metric in metrics:
        print("\n" + "=" * 80)
        print("  " + metric.upper())
        print("=" * 80)
        header = f"{'Model':<26}" + "".join(f"{s[:10]:>11}" for s in slices) + f"{'AVG':>9}"
        print(header)
        print("-" * len(header))
        for mdl in models:
            if mdl not in results:
                continue
            vals = [results[mdl][s][metric] for s in slices]
            avg = float(np.nanmean(vals))
            print(f"{mdl:<26}" + "".join(f"{v:>11.4f}" for v in vals) + f"{avg:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_models", nargs="*", default=[])
    ap.add_argument("--api_models", nargs="*", default=[])
    ap.add_argument("--slices", nargs="+", default=["nfcorpus", "fiqa", "scifact", "arguana"])
    ap.add_argument("--metrics", nargs="+", default=["ndcg@10", "recall@10", "recall@100"])
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results/benchmark")
    ap.add_argument("--corpus_batch", type=int, default=64)
    ap.add_argument("--query_batch", type=int, default=64)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--base_url", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    local_models = args.local_models
    if len(local_models) == 1 and local_models[0] in MODEL_SETS:
        local_models = MODEL_SETS[local_models[0]]
        print(f"expanded set -> {local_models}")

    print("loading slices...")
    slice_data = {}
    for s in args.slices:
        corpus, queries, qrels = load_slice(s, args.data_root)
        slice_data[s] = (corpus, queries, qrels)
        print(f"  {s}: {len(corpus)} docs, {sum(1 for q in qrels if qrels[q])} judged queries")

    results = {}
    for m in local_models:
        t0 = time.time()
        results[m] = benchmark_local(m, slice_data, args.slices, args.metrics,
                                     args.out, args.corpus_batch, args.query_batch)
        print(f"  ({time.time()-t0:.0f}s)")
    for m in args.api_models:
        t0 = time.time()
        results[m] = benchmark_api(m, slice_data, args.slices, args.metrics,
                                   args.out, args.corpus_batch, args.query_batch,
                                   args.api_key, args.base_url)
        print(f"  ({time.time()-t0:.0f}s)")

    with open(os.path.join(args.out, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print_tables(results, local_models + args.api_models, args.slices, args.metrics)


if __name__ == "__main__":
    main()
