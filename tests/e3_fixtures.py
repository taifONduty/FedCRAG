"""Synthetic client payloads shaped like the slices E3 actually shards.

The NFCorpus-shaped fixture reproduces the property the sharding decision
rests on: many positives per training query (BEIR NFCorpus train has 110,575
pairs over 2,590 queries) and a heavy integer tail, so the positive-count
distribution carries the exact ties the tie-break randomization needs.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_DOCS = 3633
N_TRAIN_QUERIES = 2590
N_EVAL_QUERIES = 323
MAX_POSITIVES = 724


def _corpus(n_docs=N_DOCS):
    return {f"d{i}": {"title": f"title {i}", "text": f"body of document {i}"}
            for i in range(n_docs)}


def nfcorpus_shaped_payload(seed=0, n_train=N_TRAIN_QUERIES,
                            n_eval=N_EVAL_QUERIES, n_docs=N_DOCS,
                            missing_docs=0):
    """A payload with NFCorpus's query/pair ratio and heavy positive tail.

    ``missing_docs`` qrel entries point at document ids absent from the
    corpus, exercising the silent-drop accounting of ``make_examples``.
    """
    rng = random.Random(seed)
    corpus = _corpus(n_docs)
    doc_ids = sorted(corpus)
    train_q, train_qrels = {}, {}
    for i in range(n_train):
        qid = f"TRAIN-{i:05d}"
        count = max(1, min(MAX_POSITIVES, int(rng.lognormvariate(3.0, 1.0))))
        picks = rng.sample(doc_ids, min(count, len(doc_ids)))
        train_q[qid] = f"train query {i}"
        train_qrels[qid] = {did: 1 for did in picks}
    for i in range(missing_docs):
        qid = f"TRAIN-{i:05d}"
        train_qrels[qid] = dict(train_qrels[qid])
        train_qrels[qid][f"absent-{i}"] = 1
    eval_q, eval_qrels = {}, {}
    for i in range(n_eval):
        qid = f"EVAL-{i:05d}"
        picks = rng.sample(doc_ids, rng.randint(1, 5))
        eval_q[qid] = f"eval query {i}"
        eval_qrels[qid] = {did: 1 for did in picks}
    return {"corpus": corpus, "train_q": train_q, "train_qrels": train_qrels,
            "eval_q": eval_q, "eval_qrels": eval_qrels,
            "split_fallback": False}


def small_payload(n_train=8, n_eval=6, positives=4, n_docs=40):
    """A degenerate payload for the guard tests (too small to shard)."""
    corpus = _corpus(n_docs)
    doc_ids = sorted(corpus)
    train_q = {f"T{i}": f"q{i}" for i in range(n_train)}
    train_qrels = {f"T{i}": {doc_ids[(i + j) % n_docs]: 1
                             for j in range(positives)}
                   for i in range(n_train)}
    eval_q = {f"E{i}": f"e{i}" for i in range(n_eval)}
    eval_qrels = {f"E{i}": {doc_ids[i]: 1} for i in range(n_eval)}
    return {"corpus": corpus, "train_q": train_q, "train_qrels": train_qrels,
            "eval_q": eval_q, "eval_qrels": eval_qrels,
            "split_fallback": False}


def payload_pairs(payload):
    """Training pairs exactly as ``make_examples`` counts them."""
    total = 0
    for qid, rels in payload["train_qrels"].items():
        if qid not in payload["train_q"]:
            continue
        total += sum(1 for did, rel in rels.items()
                     if rel > 0 and did in payload["corpus"])
    return total


def scoreable_eval_queries(payload):
    return {qid for qid in payload["eval_q"]
            if payload["eval_qrels"].get(qid)}
