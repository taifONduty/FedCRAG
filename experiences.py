"""A constructed semantic-shift stream over MS MARCO and MS-Shift: feasibility, manifests
and the per-(client, experience) data the trainer consumes."""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TFIDF = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
SVD_DIM = 50
KMEANS_RESTARTS = 10
RERUNS = 5


def _digest(items):
    return hashlib.sha256("\n".join(str(x) for x in items).encode("utf-8")).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ clustering

def cluster_queries(texts, k, seed):
    """k-means on unit-normalised TF-IDF features reduced by SVD (cosine geometry),
    fitted on ``texts`` only."""
    vectorizer = TfidfVectorizer(**TFIDF)
    features = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(SVD_DIM, features.shape[1] - 1), random_state=seed)
    reduced = normalize(svd.fit_transform(features))
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_RESTARTS, random_state=seed).fit(reduced)
    record = {"tfidf": dict(TFIDF), "svd_dim": int(svd.n_components), "unit_rows": True,
              "k": k, "seed": seed,
              "n_init": KMEANS_RESTARTS,
              "vocabulary_sha256": _digest(sorted(vectorizer.vocabulary_)),
              "centroid_sha256": hashlib.sha256(
                  np.round(kmeans.cluster_centers_, 8).tobytes()).hexdigest()}
    return {"assignment": kmeans.labels_.tolist(), "record": record,
            "_vectorizer": vectorizer, "_svd": svd, "_kmeans": kmeans}


def assign_queries(model, texts):
    """Assign held-out texts to the frozen centroids."""
    reduced = normalize(model["_svd"].transform(model["_vectorizer"].transform(texts)))
    return model["_kmeans"].predict(reduced).tolist()


def cluster_with_minimum(texts, k, seed, minimum):
    """Rerun the clustering with the next seed until every sub-cluster holds at least
    ``minimum`` queries; fail after RERUNS reruns."""
    for attempt in range(RERUNS + 1):
        model = cluster_queries(texts, k, seed + attempt)
        sizes = np.bincount(model["assignment"], minlength=k)
        if sizes.min() >= minimum:
            return model
    raise ValueError(f"a sub-cluster stayed below {minimum} queries after {RERUNS} reruns "
                     f"(sizes {sizes.tolist()} at seed {seed + RERUNS})")


def choose_counts(table, train_cap, guard_cap, test_cap):
    """The largest uniform counts every cell supports: guard is a tenth of the training
    share, test comes from the official evaluation queries."""
    min_train = min(cell["train_eligible"] for cell in table.values())
    min_eval = min(cell["eval_eligible"] for cell in table.values())
    guard = min(guard_cap, min_train // 11)
    return {"train": min(train_cap, min_train - guard), "guard": guard,
            "test": min(test_cap, min_eval)}


# ------------------------------------------------------------------- loading

def _read_tsv(path, columns):
    with open(path, newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            yield row[:columns]


def load_queries(path):
    return {qid: text for qid, text in _read_tsv(path, 2)}


def load_train_qrels(path):
    qrels = {}
    for qid, _, pid, rel in _read_tsv(path, 4):
        if int(rel) > 0:
            qrels.setdefault(qid, {})[pid] = int(rel)
    return qrels


def load_msshift(root):
    """MS-Shift training queries: {query text: topic cluster}."""
    labels = {}
    path = os.path.join(root, "ms-marco-shift", "TRAIN", "queries_clustering.tsv")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            labels[row["query"].strip()] = int(row["topic_cluster"])
    return labels


def has_official_eval(root, topic):
    return os.path.exists(os.path.join(root, "ms-marco-shift", "EVAL", "queries",
                                       f"queries_{topic}.tsv"))


def load_eval(root, topic):
    """MS-Shift's official evaluation queries and qrels for one topic cluster."""
    base = os.path.join(root, "ms-marco-shift", "EVAL")
    queries = load_queries(os.path.join(base, "queries", f"queries_{topic}.tsv"))
    with open(os.path.join(base, "qrel", f"qrel_{topic}.json")) as handle:
        qrels = json.load(handle)
    return queries, {q: rels for q, rels in qrels.items() if q in queries}


HOLDOUT_FRACTION = 0.15


def eval_queries(manifest, root, client):
    """The evaluation pool of one client: the official queries of its topic, or, for a
    topic without official evaluation queries, its held-out training queries."""
    entry = manifest["clients"][client]
    pool = set(entry["eval_pool"])
    if entry["eval_source"] == "official":
        queries, qrels = load_eval(root, entry["topic"])
    else:
        queries = load_queries(os.path.join(root, "msmarco-passage", "queries.train.tsv"))
        qrels = load_train_qrels(os.path.join(root, "msmarco-passage", "qrels.train.tsv"))
    return ({q: queries[q] for q in pool}, {q: qrels[q] for q in pool})


def collection_ids(root):
    """Every passage id of the collection, streamed, as a sorted integer array."""
    ids = []
    with open(os.path.join(root, "msmarco-passage", "collection.tsv"), newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            ids.append(int(row[0]))
    return np.sort(np.asarray(ids, dtype=np.int64))


def load_passages(root, wanted):
    wanted = set(wanted)
    passages = {}
    with open(os.path.join(root, "msmarco-passage", "collection.tsv"), newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row[0] in wanted:
                passages[row[0]] = row[1]
    missing = wanted - set(passages)
    if missing:
        raise ValueError(f"{len(missing)} passages of the manifest are not in the collection")
    return passages


def _training_side(root):
    """Topic-labelled MS MARCO training queries that have judgements: {qid: (text, topic)}."""
    labels = load_msshift(root)
    queries = load_queries(os.path.join(root, "msmarco-passage", "queries.train.tsv"))
    qrels = load_train_qrels(os.path.join(root, "msmarco-passage", "qrels.train.tsv"))
    side = {}
    for qid in sorted(queries, key=int):
        text = queries[qid].strip()
        if text in labels and qid in qrels and qid not in side:
            side[qid] = (text, labels[text])
    return side, qrels


# ----------------------------------------------------------------- manifests

def _pseudo_clients(topic, qids, texts, eval_ids, eval_texts, pseudo_clients, seed):
    """Split one topic into ``pseudo_clients`` clients by a first clustering, fitted on
    the training text; client id = 10 * topic + index."""
    if pseudo_clients == 1:
        return {topic: (qids, eval_ids)}
    model = cluster_queries(texts, pseudo_clients, seed)
    eval_assignment = assign_queries(model, eval_texts)
    return {10 * topic + i: ([q for q, a in zip(qids, model["assignment"]) if a == i],
                             [q for q, a in zip(eval_ids, eval_assignment) if a == i])
            for i in range(pseudo_clients)}


def _cells(root, clients, k, seed, minimum, pseudo_clients=1):
    """Per client: the clustering model (every sub-cluster at least ``minimum`` training
    queries), its training-side queries by sub-cluster and its evaluation queries by
    sub-cluster."""
    side, train_qrels = _training_side(root)
    cells = {}
    for topic in clients:
        topic_ids = sorted((q for q, (_, t) in side.items() if t == topic), key=int)
        if has_official_eval(root, topic):
            source = "official"
            eval_q, eval_qrels = load_eval(root, topic)
            topic_eval = sorted((q for q in eval_q if q in eval_qrels), key=int)
        else:
            source = "train-holdout"
            rng = np.random.default_rng([seed, topic, 99])
            held = rng.choice(len(topic_ids), size=int(HOLDOUT_FRACTION * len(topic_ids)),
                              replace=False)
            topic_eval = sorted((topic_ids[i] for i in held), key=int)
            eval_q = {q: side[q][0] for q in topic_eval}
            eval_qrels = {q: train_qrels[q] for q in topic_eval}
            topic_ids = [q for q in topic_ids if q not in set(topic_eval)]
        split = _pseudo_clients(topic, topic_ids, [side[q][0] for q in topic_ids],
                                topic_eval, [eval_q[q] for q in topic_eval],
                                pseudo_clients, seed)
        for client, (qids, eval_ids) in split.items():
            model = cluster_with_minimum([side[q][0] for q in qids], k, seed, minimum)
            eval_assignment = assign_queries(model, [eval_q[q] for q in eval_ids])
            cells[client] = {
                "topic": topic, "model": model, "eval_source": source, "eval_pool": eval_ids,
                "train": {e: [q for q, a in zip(qids, model["assignment"]) if a == e]
                          for e in range(k)},
                "eval": {e: [q for q, a in zip(eval_ids, eval_assignment) if a == e]
                         for e in range(k)},
                "eval_qrels": eval_qrels, "eval_q": eval_q}
    return cells, side, train_qrels


def feasibility(root, clients, k, seed, pseudo_clients=1):
    cells, _, train_qrels = _cells(root, clients, k, seed, 1, pseudo_clients)
    table = {}
    for topic, cell in cells.items():
        for e in range(k):
            relevant = {p for q in cell["train"][e] for p in train_qrels[q]}
            relevant |= {p for q in cell["eval"][e] for p in cell["eval_qrels"][q]}
            table[(topic, e)] = {"train_eligible": len(cell["train"][e]),
                                 "eval_eligible": len(cell["eval"][e]),
                                 "relevant_passages": len(relevant)}
    return table


def build_manifest(root, clients, experiences_per_client, schedule, counts, corpus_size,
                   hard_k, seed, retriever, source_digests=None, pseudo_clients=1):
    """Sample the splits, assemble each client's fixed corpus and record every list with
    its digest. ``retriever(texts, k)`` returns hard-distractor passage ids per text."""
    k = experiences_per_client
    minimum = counts["train"] + counts["guard"]
    cells, side, train_qrels = _cells(root, clients, k, seed, minimum, pseudo_clients)
    universe = collection_ids(root)
    manifest = {"version": 1, "seed": seed, "counts": dict(counts), "corpus_size": corpus_size,
                "hard_k": hard_k, "experiences_per_client": k, "pseudo_clients": pseudo_clients,
                "source_sha256": source_digests or {}, "clients": {}}
    for topic in sorted(cells):
        cell = cells[topic]
        client = {"topic": cell["topic"], "order": list(schedule[topic]),
                  "eval_source": cell["eval_source"], "eval_pool": list(cell["eval_pool"]),
                  "clustering": cell["model"]["record"], "experiences": {}}
        relevant, queries_for_hard = set(), []
        for e in range(k):
            rng = np.random.default_rng([seed, topic, e])
            train_pool = cell["train"][e]
            picked = rng.choice(len(train_pool), size=minimum, replace=False)
            train_ids = sorted((train_pool[i] for i in picked[:counts["train"]]), key=int)
            guard_ids = sorted((train_pool[i] for i in picked[counts["train"]:]), key=int)
            eval_pool = cell["eval"][e]
            if len(eval_pool) < counts["test"]:
                raise ValueError(f"client {topic} experience {e} has {len(eval_pool)} "
                                 f"evaluation queries, fewer than {counts['test']}")
            picked = rng.choice(len(eval_pool), size=counts["test"], replace=False)
            test_ids = sorted((eval_pool[i] for i in picked), key=int)
            qrels = {q: train_qrels[q] for q in train_ids + guard_ids}
            qrels.update({q: cell["eval_qrels"][q] for q in test_ids})
            client["experiences"][str(e)] = {"train": train_ids, "guard": guard_ids,
                                             "test": test_ids, "qrels": qrels}
            relevant |= {p for rels in qrels.values() for p in rels}
            queries_for_hard += [side[q][0] for q in train_ids + guard_ids]
            queries_for_hard += [cell["eval_q"][q] for q in test_ids]
        relevant |= {p for rels in cell["eval_qrels"].values() for p in rels}
        hard = {p for hits in retriever(queries_for_hard, hard_k) for p in hits} - relevant
        chosen = relevant | hard
        if len(chosen) > corpus_size:
            raise ValueError(f"client {topic}: {len(chosen)} relevant and hard passages "
                             f"exceed the corpus size {corpus_size}")
        rng = np.random.default_rng([seed, topic, k])
        candidates = universe[~np.isin(universe, np.asarray(sorted(int(p) for p in chosen)))]
        fill = rng.choice(candidates, size=corpus_size - len(chosen), replace=False)
        corpus = (sorted(relevant, key=int) + sorted(hard, key=int)
                  + [str(p) for p in np.sort(fill)])
        client["corpus"] = corpus
        client["corpus_parts"] = {"relevant": len(relevant), "hard": len(hard),
                                  "random": int(len(fill))}
        client["digests"] = _client_digests(client)
        manifest["clients"][str(topic)] = client
    return manifest


def _client_digests(client):
    cells = client["experiences"]
    return {"corpus": _digest(client["corpus"]),
            "experiences": _digest(
                f"{e}:{split}:{','.join(cells[e][split])}"
                for e in sorted(cells, key=int) for split in ("train", "guard", "test")),
            "qrels": _digest(json.dumps({e: cells[e]["qrels"] for e in sorted(cells, key=int)},
                                        sort_keys=True))}


def verify_manifest(manifest):
    for name, client in manifest["clients"].items():
        for key, value in _client_digests(client).items():
            if client["digests"].get(key) != value:
                raise ValueError(f"client {name}: {key} digest does not match its list")
        cells = client["experiences"]
        seen = set()
        for e in cells.values():
            ids = set(e["train"]) | set(e["guard"]) | set(e["test"])
            if len(ids) != len(e["train"]) + len(e["guard"]) + len(e["test"]) or seen & ids:
                raise ValueError(f"client {name}: splits overlap")
            seen |= ids
        if len(set(client["corpus"])) != len(client["corpus"]):
            raise ValueError(f"client {name}: corpus has repeated passages")


def client_corpus(manifest, root, client):
    ids = manifest["clients"][client]["corpus"]
    passages = load_passages(root, ids)
    return {pid: {"text": passages[pid]} for pid in ids}


def materialise(manifest, root, client, experience, corpus=None):
    """The data dict of one (client, experience) cell in the shape ``client_train`` and
    the evaluation consume."""
    cell = manifest["clients"][client]["experiences"][str(experience)]
    train_queries = load_queries(os.path.join(root, "msmarco-passage", "queries.train.tsv"))
    eval_q, _ = eval_queries(manifest, root, client)
    data = {"corpus": corpus if corpus is not None else client_corpus(manifest, root, client)}
    for split, source in (("train", train_queries), ("guard", train_queries), ("test", eval_q)):
        data[f"{split}_q"] = {q: source[q] for q in cell[split]}
        data[f"{split}_qrels"] = {q: cell["qrels"][q] for q in cell[split]}
    return data


# ------------------------------------------------------------------------ CLI

def bm25_retriever(root):
    """Hard distractors from BM25 over the full collection (needs the memory of the GPU
    machine, not the laptop)."""
    import bm25s
    ids, texts = [], []
    with open(os.path.join(root, "msmarco-passage", "collection.tsv"), newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            ids.append(row[0])
            texts.append(row[1])
    index = bm25s.BM25()
    index.index(bm25s.tokenize(texts, stopwords="en"))

    def retrieve(query_texts, k):
        hits, _ = index.retrieve(bm25s.tokenize(query_texts, stopwords="en"), k=k)
        return [[ids[j] for j in row] for row in hits]
    return retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["feasibility", "build", "verify"])
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--clients", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--experiences", type=int, default=4)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--schedule", help="JSON {client: [experience order]} (build)")
    ap.add_argument("--counts", help="JSON {train, guard, test} (build)")
    ap.add_argument("--corpus_size", type=int, default=60000)
    ap.add_argument("--hard_k", type=int, default=10)
    ap.add_argument("--pseudo_clients", type=int, default=1,
                    help="split each topic into this many clients (calibration stream)")
    ap.add_argument("--out", help="manifest path (build) or manifest to verify")
    args = ap.parse_args()
    if args.command == "feasibility":
        table = feasibility(args.data_root, args.clients, args.experiences, args.seed,
                            args.pseudo_clients)
        print("client\texperience\ttrain_eligible\teval_eligible\trelevant_passages")
        for (topic, e), cell in sorted(table.items()):
            print(f"{topic}\t{e}\t{cell['train_eligible']}\t{cell['eval_eligible']}"
                  f"\t{cell['relevant_passages']}")
        return
    if args.command == "verify":
        with open(args.out) as handle:
            verify_manifest(json.load(handle))
        print("manifest verified")
        return
    schedule = {int(c): tuple(order) for c, order in json.loads(args.schedule).items()}
    sources = {name: _file_digest(os.path.join(args.data_root, name)) for name in (
        "msmarco-passage/collection.tsv", "msmarco-passage/queries.train.tsv",
        "msmarco-passage/qrels.train.tsv", "ms-marco-shift/TRAIN/queries_clustering.tsv")}
    manifest = build_manifest(args.data_root, args.clients, args.experiences, schedule,
                              json.loads(args.counts), args.corpus_size, args.hard_k,
                              args.seed, bm25_retriever(args.data_root), sources,
                              args.pseudo_clients)
    verify_manifest(manifest)
    with open(args.out, "w") as handle:
        json.dump(manifest, handle)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
