"""LoTTE as a continual stream for the confirmation block (registration section 16). The five
topics are the clients; each topic's dev forums and test forums, which share no forum and no
passage, are its two experiences. Manifests have the shape experiences.py builds, so the
driver and the validator read them unchanged."""
import argparse
import copy
import json
import os

import numpy as np

from experiences import _client_digests, _digest, _file_digest, verify_manifest

TOPICS = ("lifestyle", "recreation", "science", "technology", "writing")
SPLITS = ("dev", "test")
KINDS = ("search", "forum")
HARD_K = (5, 3)


class CorpusOverflow(ValueError):
    pass


def _split_dir(root, topic, split):
    return os.path.join(root, "lotte", "lotte", topic, split)


def _collection_rows(root, topic):
    for split in SPLITS:
        with open(os.path.join(_split_dir(root, topic, split), "collection.tsv")) as handle:
            for line in handle:
                pid, text = line.rstrip("\n").split("\t", 1)
                yield f"{split}-{pid}", text


def load_queries(root, topic):
    """Every search and forum query of a topic: its text, its answer passages and its
    experience (0 for the dev forums, 1 for the test forums)."""
    texts, answers, experience = {}, {}, {}
    for e, split in enumerate(SPLITS):
        for kind in KINDS:
            path = os.path.join(_split_dir(root, topic, split), f"qas.{kind}.jsonl")
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    q = f"{split}-{kind}-{row['qid']}"
                    texts[q] = row["query"]
                    answers[q] = [f"{split}-{p}" for p in row["answer_pids"]]
                    experience[q] = e
    return texts, answers, experience


def answer_votes(root, topic):
    votes = {}
    for split in SPLITS:
        with open(os.path.join(_split_dir(root, topic, split), "metadata.jsonl")) as handle:
            for line in handle:
                row = json.loads(line)
                votes.update({f"{split}-{p}": s for p, s in zip(row["post_ids"], row["scores"])})
    return votes


def top_answer(answers, votes):
    """The highest-voted answer; a tie goes to the smallest passage number."""
    return max(answers, key=lambda p: (votes[p], -int(p.split("-")[1])))


def load_passages(root, topic, wanted):
    wanted = set(wanted)
    passages = {pid: text for pid, text in _collection_rows(root, topic) if pid in wanted}
    if len(passages) != len(wanted):
        raise ValueError(f"{len(wanted) - len(passages)} passages are not in LoTTE {topic}")
    return passages


def bm25_retriever(root, topic):
    """BM25 over a topic's two collections, built once and saved beside them."""
    import bm25s
    index_dir = os.path.join(root, "lotte", f"bm25s_{topic}")
    ids_path = os.path.join(index_dir, "ids.json")
    if os.path.exists(ids_path):
        index = bm25s.BM25.load(index_dir)
        with open(ids_path) as handle:
            ids = json.load(handle)
    else:
        ids, texts = zip(*_collection_rows(root, topic))
        index = bm25s.BM25()
        index.index(bm25s.tokenize(list(texts), stopwords="en"))
        index.save(index_dir)
        with open(ids_path, "w") as handle:
            json.dump(list(ids), handle)
    try:
        index.activate_numba_scorer()
    except ImportError:
        pass

    def retrieve(query_texts, k):
        hits, _ = index.retrieve(bm25s.tokenize(query_texts, stopwords="en"), k=k,
                                 n_threads=-1)
        return [[ids[j] for j in row] for row in hits]
    return retrieve


def _client(root, c, topic, counts, corpus_size, hard_k, seed, rows_for):
    texts, answers, experience = load_queries(root, topic)
    votes = answer_votes(root, topic)
    size = counts["train"] + counts["guard"] + counts["test"]
    client = {"topic": topic, "order": [], "eval_source": "none", "eval_pool": [],
              "experiences": {}}
    relevant, picked_all = set(), []
    for e in range(len(SPLITS)):
        pool = sorted(q for q in texts if experience[q] == e)
        if len(pool) < size:
            raise ValueError(f"{topic} experience {e} has {len(pool)} queries, fewer than {size}")
        picked = [pool[i] for i in np.random.default_rng([seed, c, e]).choice(
            len(pool), size=size, replace=False)]
        train = sorted(picked[:counts["train"]])
        guard = sorted(picked[counts["train"]:counts["train"] + counts["guard"]])
        test = sorted(picked[counts["train"] + counts["guard"]:])
        qrels = {q: {top_answer(answers[q], votes): 1} for q in train}
        qrels.update({q: {p: 1 for p in answers[q]} for q in guard + test})
        client["experiences"][str(e)] = {"train": train, "guard": guard, "test": test,
                                         "qrels": qrels}
        relevant |= {p for q in picked for p in answers[q]}
        picked_all += picked
    rows = rows_for(topic, [texts[q] for q in picked_all])
    hard = {p for row in rows for p in row[:hard_k]} - relevant
    chosen = relevant | hard
    if len(chosen) > corpus_size:
        raise CorpusOverflow(f"{topic}: {len(chosen)} answer and hard passages exceed the "
                             f"{corpus_size}-passage corpus")
    rest = sorted({pid for pid, _ in _collection_rows(root, topic)} - chosen)
    drawn = np.random.default_rng([seed, c, len(SPLITS)]).choice(
        len(rest), size=corpus_size - len(chosen), replace=False)
    client["corpus"] = sorted(relevant) + sorted(hard) + sorted(rest[i] for i in drawn)
    client["corpus_parts"] = {"relevant": len(relevant), "hard": len(hard),
                              "random": len(drawn)}
    guard_ids = {q for cell in client["experiences"].values() for q in cell["guard"]}
    client["guard_hits"] = {q: row for q, row in zip(picked_all, rows) if q in guard_ids}
    client["digests"] = _client_digests(client)
    return client


def build_manifests(root, schedules, counts, corpus_size, seed, retriever_for,
                    topics=TOPICS):
    """One build for every schedule. Hard passages are each selected query's top-5 BM25
    passages, or its top-3 for every client if any client's corpus would overflow."""
    cache = {}

    def rows_for(topic, texts):
        key = (topic, _digest(texts))
        if key not in cache:
            cache[key] = retriever_for(topic)(texts, HARD_K[0])
        return cache[key]

    for hard_k in HARD_K:
        try:
            clients = {str(c): _client(root, c, topic, counts, corpus_size, hard_k, seed,
                                       rows_for)
                       for c, topic in enumerate(topics)}
            break
        except CorpusOverflow:
            if hard_k == HARD_K[-1]:
                raise
    base = {"version": 1, "source": "lotte", "seed": seed, "counts": dict(counts),
            "corpus_size": corpus_size, "hard_k": hard_k, "experiences_per_client": len(SPLITS),
            "clients": clients}
    manifests = {}
    for name, schedule in schedules.items():
        manifest = copy.deepcopy(base)
        for c, client in manifest["clients"].items():
            client["order"] = list(schedule[c])
        manifests[name] = manifest
    return manifests


def no_shift_manifest(root, manifest, seed):
    """The no-shift control of the amended section 16: each client's selected queries,
    re-split at random into two experiences of the same sizes, with the same corpus and the
    same judgement rule, so the second experience brings no change of forums."""
    control = copy.deepcopy(manifest)
    control["control"] = "no-shift"
    counts = manifest["counts"]
    size = counts["train"] + counts["guard"] + counts["test"]
    for c, client in control["clients"].items():
        _, answers, _ = load_queries(root, client["topic"])
        votes = answer_votes(root, client["topic"])
        pool = sorted(q for cell in client["experiences"].values()
                      for split in ("train", "guard", "test") for q in cell[split])
        order = np.random.default_rng([seed, int(c), len(SPLITS) + 1]).permutation(len(pool))
        for e in range(len(SPLITS)):
            picked = [pool[i] for i in order[e * size:(e + 1) * size]]
            train = sorted(picked[:counts["train"]])
            guard = sorted(picked[counts["train"]:counts["train"] + counts["guard"]])
            test = sorted(picked[counts["train"] + counts["guard"]:])
            qrels = {q: {top_answer(answers[q], votes): 1} for q in train}
            qrels.update({q: {p: 1 for p in answers[q]} for q in guard + test})
            client["experiences"][str(e)] = {"train": train, "guard": guard, "test": test,
                                             "qrels": qrels}
        client["guard_hits"] = {}
        client["digests"] = _client_digests(client)
    return control


def guard_hits(manifest):
    """Each guard query's top-5 BM25 passages, per client, for arm F's check pool."""
    return {c: client["guard_hits"] for c, client in manifest["clients"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "guard-hits", "no-shift"])
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--schedules", help="JSON {name: {client: [experience order]}} (build)")
    ap.add_argument("--counts", help="JSON {train, guard, test} (build)")
    ap.add_argument("--corpus_size", type=int, default=60000)
    ap.add_argument("--manifest", help="source manifest (guard-hits, no-shift)")
    ap.add_argument("--out", required=True, help="output directory (build) or hits file")
    args = ap.parse_args()
    if args.command == "no-shift":
        with open(args.manifest) as handle:
            control = no_shift_manifest(args.data_root, json.load(handle), args.seed)
        verify_manifest(control)
        with open(args.out, "w") as handle:
            json.dump(control, handle)
        return
    if args.command == "guard-hits":
        with open(args.manifest) as handle:
            hits = guard_hits(json.load(handle))
        with open(args.out, "w") as handle:
            json.dump(hits, handle)
        return
    manifests = build_manifests(args.data_root, json.loads(args.schedules),
                                json.loads(args.counts), args.corpus_size, args.seed,
                                lambda topic: bm25_retriever(args.data_root, topic))
    source = {"lotte.tar.gz": _file_digest(os.path.join(args.data_root, "lotte",
                                                        "lotte.tar.gz"))}
    os.makedirs(args.out, exist_ok=True)
    for name, manifest in manifests.items():
        manifest["source_sha256"] = source
        verify_manifest(manifest)
        path = os.path.join(args.out, f"lotte_{name}.json")
        with open(path, "w") as handle:
            json.dump(manifest, handle)
        print(f"wrote {path}: hard_k {manifest['hard_k']}, corpus parts "
              f"{ {c: m['corpus_parts'] for c, m in manifest['clients'].items()} }")


if __name__ == "__main__":
    main()
