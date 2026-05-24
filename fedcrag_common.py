import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
import pytrec_eval


LOCAL_MODELS = {
    "bge-small":      ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", "", False),
    "bge-base":       ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", "", False),
    "bge-large":      ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", "", False),
    "bge-m3":         ("BAAI/bge-m3", "", "", False),
    "e5-base":        ("intfloat/e5-base-v2", "query: ", "passage: ", False),
    "e5-large":       ("intfloat/e5-large-v2", "query: ", "passage: ", False),
    "me5-large":      ("intfloat/multilingual-e5-large-instruct", "Instruct: Given a query, retrieve relevant passages\nQuery: ", "", False),
    "gte-large":      ("Alibaba-NLP/gte-large-en-v1.5", "", "", False),
    "gte-mbase":      ("Alibaba-NLP/gte-multilingual-base", "", "", False),
    "mxbai-large":    ("mixedbread-ai/mxbai-embed-large-v1", "Represent this sentence for searching relevant passages: ", "", False),
    "nomic-v1.5":     ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: ", False),
    "snowflake-l":    ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", "", False),
    "embeddinggemma": ("google/embeddinggemma-300m", "task: search result | query: ", "title: none | text: ", False),
    "jina-v3":        ("jinaai/jina-embeddings-v3", "", "", False),
    "stella-1.5b":    ("NovaSearch/stella_en_1.5B_v5", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", False),
    "gte-qwen2-1.5b": ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", True),
    "qwen3-0.6b":     ("Qwen/Qwen3-Embedding-0.6B", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", False),
    "qwen3-4b":       ("Qwen/Qwen3-Embedding-4B", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", True),
    "qwen3-8b":       ("Qwen/Qwen3-Embedding-8B", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", True),
    "e5-mistral":     ("intfloat/e5-mistral-7b-instruct", "Instruct: Given a query, retrieve relevant passages\nQuery: ", "", True),
    "gte-qwen2-7b":   ("Alibaba-NLP/gte-Qwen2-7B-instruct", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", True),
    "sfr-2r":         ("Salesforce/SFR-Embedding-2_R", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", True),
    "nv-embed-v2":    ("nvidia/NV-Embed-v2", "Instruct: Given a question, retrieve passages that answer it\nQuery: ", "", True),
    "linq-mistral":   ("Linq-AI-Research/Linq-Embed-Mistral", "Instruct: Given a query, retrieve relevant passages\nQuery: ", "", True),
    "gritlm-7b":      ("GritLM/GritLM-7B", "<|user|>\nGiven a query, retrieve relevant passages\n<|embed|>\n", "<|embed|>\n", True),
}


API_MODELS = {
    "openai-3-large":   {"provider": "openai",     "model": "text-embedding-3-large", "dim": 3072},
    "openai-3-small":   {"provider": "openai",     "model": "text-embedding-3-small", "dim": 1536},
    "cohere-v4":        {"provider": "cohere",      "model": "embed-v4.0",             "dim": 1536},
    "voyage-3.5":       {"provider": "voyage",      "model": "voyage-3.5",             "dim": 1024},
    "gemini-001":       {"provider": "openrouter",  "model": "google/gemini-embedding-001", "dim": 3072},
}


MODEL_SETS = {
    "train_tier": ["bge-small", "bge-base", "bge-large", "bge-m3"],
    "compare":    ["bge-m3", "e5-large", "gte-large", "mxbai-large", "snowflake-l", "qwen3-0.6b", "gte-qwen2-1.5b"],
    "sota_local": ["bge-m3", "e5-large", "gte-qwen2-1.5b", "qwen3-4b", "e5-mistral", "qwen3-8b", "nv-embed-v2", "gte-qwen2-7b"],
    "leaderboard_top": ["qwen3-8b", "qwen3-4b", "gte-qwen2-7b", "e5-mistral", "me5-large", "bge-m3"],
}


def resolve_local(name):
    if name in LOCAL_MODELS:
        path, qp, dp, fp16 = LOCAL_MODELS[name]
        return path, qp, dp, fp16
    return name, "", "", False


def load_slice(name, data_root):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    path = beir_util.download_and_unzip(url, data_root)
    corpus, queries, qrels = GenericDataLoader(path).load(split="test")
    return corpus, queries, qrels


def load_slice_with_train(name, data_root):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    path = beir_util.download_and_unzip(url, data_root)
    corpus, test_q, test_qrels = GenericDataLoader(path).load(split="test")
    try:
        _, train_q, train_qrels = GenericDataLoader(path).load(split="train")
    except Exception:
        qids = sorted(test_qrels.keys())
        half = len(qids) // 2
        tr, ev = set(qids[:half]), set(qids[half:])
        train_q = {q: test_q[q] for q in tr if q in test_q}
        train_qrels = {q: test_qrels[q] for q in tr}
        test_q = {q: test_q[q] for q in ev if q in test_q}
        test_qrels = {q: test_qrels[q] for q in ev}
    return {"corpus": corpus, "train_q": train_q, "train_qrels": train_qrels,
            "eval_q": test_q, "eval_qrels": test_qrels}


def doc_text(d):
    return (d.get("title", "") + " " + d.get("text", "")).strip()


def parse_metric(m):
    name, k = m.split("@")
    k = int(k)
    mapping = {"ndcg": "ndcg_cut", "recall": "recall", "map": "map_cut",
               "precision": "P", "mrr": "recip_rank", "success": "success"}
    return mapping.get(name, name), k


def required_depth(metrics, floor=100):
    depth = floor
    for m in metrics:
        _, k = parse_metric(m)
        depth = max(depth, k)
    return depth


def evaluate_metrics(cids, c_emb, qids, q_emb, qrels, metrics):
    depth = required_depth(metrics)
    sims = q_emb @ c_emb.T
    topk = np.argsort(-sims, axis=1)[:, :min(depth, sims.shape[1])]
    run = {qids[i]: {cids[j]: float(sims[i, j]) for j in topk[i]}
           for i in range(len(qids))}
    measures = set()
    for m in metrics:
        base, k = parse_metric(m)
        if base in ("ndcg_cut", "map_cut", "P", "recall", "success"):
            measures.add(f"{base}.{k}")
        else:
            measures.add(base)
    ev = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = ev.evaluate(run)
    out = {}
    for m in metrics:
        base, k = parse_metric(m)
        key = f"{base}_{k}" if base in ("ndcg_cut", "map_cut", "P", "recall", "success") else base
        vals = [scores[q][key] for q in scores if key in scores[q]]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_local_model(name):
    path, q_prefix, d_prefix, fp16 = resolve_local(name)
    model = SentenceTransformer(path, trust_remote_code=True)
    if fp16:
        model.half()
    return model, q_prefix, d_prefix, fp16


def encode_texts(model, texts, prefix, batch_size, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            return emb
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    if cache_path:
        np.save(cache_path, emb)
    return emb


class APIEmbedder:
    def __init__(self, model_key, api_key=None, base_url=None):
        cfg = API_MODELS[model_key]
        self.provider = cfg["provider"]
        self.model = cfg["model"]
        self.dim = cfg["dim"]
        from openai import OpenAI
        if self.provider == "openrouter":
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
                base_url=base_url or "https://openrouter.ai/api/v1")
        elif self.provider == "openai":
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "voyage":
            self.client = OpenAI(api_key=api_key or os.environ.get("VOYAGE_API_KEY"),
                                 base_url="https://api.voyageai.com/v1")
        elif self.provider == "cohere":
            self.client = OpenAI(api_key=api_key or os.environ.get("COHERE_API_KEY"),
                                 base_url="https://api.cohere.ai/compatibility/v1")
        else:
            raise ValueError(f"unknown provider {self.provider}")

    def encode(self, texts, batch_size=64, **kwargs):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            out.extend([d.embedding for d in resp.data])
        arr = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
