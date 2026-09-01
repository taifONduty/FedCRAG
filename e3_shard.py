"""Split one federated client into near-duplicate sub-silos at query level.

The E3 clone federation needs several clients drawn from one distribution.
The partition unit is the **training query**, never the (query, document)
pair: BEIR NFCorpus train carries 110,575 pairs over 2,590 queries, so a
pair-level cut would place every query in every shard and leave the shards
sharing their whole query population. Queries are kept atomic and the shards
are balanced on the pair count ``make_examples`` will actually emit, because
that count is both the ``--weight_by examples`` basis and the quantity a step
cap divides.

Balancing is longest-processing-time: descending weight, ties broken by a
seeded random key, each query to the currently lightest bin. The seed enters
through the tie-break rather than through a pre-shuffle, so the split depends
on neither the stability of the sort nor the payload's key order. Measured on
a 2,590-query / 84,867-pair NFCorpus-shaped payload with 188 distinct integer
positive counts: five seeds give five distinct partitions, every one of them
28,289 / 28,289 / 28,289 pairs (max/min ratio 1.000000) and 108 / 108 / 107
eval queries, with shard-0 re-overlap across seeds 0.36-0.37 against 1/3 by
chance -- so a second shard seed is close to an independent partition.

The corpus is shared between a parent's shards by object identity, so every
sub-silo retrieves over the same index its parent did: the task is unchanged,
a gold document cannot be evicted from one shard's index, and the
size-weighted mean of the per-shard scores reconstructs the monolithic score
exactly. Eval queries are partitioned instead, which is what makes the
sub-silos' scores distinguishable at all.

Nothing here mutates a parent payload. Shards get new section dicts holding
the parent's own value objects.
"""
import hashlib
import json
import os
import random
from collections import namedtuple

SHARDER_ALGORITHM = "lpt-desc-weight-random-tiebreak-greedy-lightest-bin/v2"
MANIFEST_SCHEMA_VERSION = "e3-shard/1"

#: A silo scoring fewer queries than this has a per-client nDCG too noisy to
#: read, and ``eval_global`` emits a silent NaN column at zero.
MIN_EVAL_QUERIES = 50

#: ``make_examples`` silently drops a positive whose document is missing from
#: the corpus. Dropping is legitimate (ArguAna: 703 train queries, 701 pairs)
#: but a shard losing a materially different share of its pairs than its
#: siblings is not that shard's sibling any more.
MAX_DROP_RATE = 0.01

#: Absolute slack allowed between a shard's drop rate and its parent's. The
#: spec proposed an exact 1e-9 agreement between shards; that is unreachable
#: whenever any pair is dropped at all. Which queries carry a dangling qrel is
#: independent of the partition, so the per-shard counts are a random split of
#: a fixed pool and their rates differ by sampling noise alone -- three
#: NFCorpus-sized shards splitting 100 dangling positives spread their rates
#: by ~4e-4 with nothing wrong. The check that remains meaningful is against
#: that noise, so the gate is this absolute slack plus DROP_RATE_SIGMA
#: binomial standard errors of the parent rate.
DROP_RATE_TOL = 1e-3
DROP_RATE_SIGMA = 4.0

SECTIONS = ("corpus", "train_q", "train_qrels", "eval_q", "eval_qrels")

ShardedFederation = namedtuple(
    "ShardedFederation", ["data", "slices", "step_caps", "manifest"])


class ShardingError(ValueError):
    """A sharded federation that is not the one the design asked for.

    Every condition raising this is checked before the first gradient step;
    none of them is recoverable by degrading the split.
    """


# --- primitives ------------------------------------------------------------

def parse_shard_spec(tokens, slices):
    """``["nfcorpus:3"]`` -> ``{"nfcorpus": 3}``, checked against --slices."""
    spec = {}
    for token in tokens:
        parent, sep, count = str(token).partition(":")
        if not sep or not count.isdigit() or not parent:
            raise ShardingError(
                f"--shard_spec entries must look like PARENT:N; got '{token}'")
        n_shards = int(count)
        if n_shards < 1:
            raise ShardingError(
                f"--shard_spec '{token}' must request at least 1 shard")
        if parent in spec:
            raise ShardingError(f"--shard_spec parent repeated: {parent}")
        if parent not in slices:
            raise ShardingError(
                f"--shard_spec parent '{parent}' is not in --slices")
        spec[parent] = n_shards
    return spec


def conserved_step_caps(parent_cap, n_shards):
    """Per-shard caps summing exactly to the parent's, largest first.

    Splitting a client must not multiply its optimization work: three shards
    each running the parent's 500-step cap would train 1,500 steps per round
    against the monolith's 500, changing local work and aggregation mass at
    the same time.
    """
    if n_shards < 1:
        raise ShardingError("n_shards must be at least 1")
    if parent_cap < 0:
        raise ShardingError("parent step cap must be nonnegative")
    base, remainder = divmod(parent_cap, n_shards)
    return [base + (1 if i < remainder else 0) for i in range(n_shards)]


def train_pair_weights(payload):
    """Per-query training-pair count, counted exactly as ``make_examples``."""
    corpus = payload["corpus"]
    train_q = payload["train_q"]
    weights = {}
    for qid, rels in payload["train_qrels"].items():
        if qid not in train_q:
            continue
        weights[qid] = sum(1 for did, rel in rels.items()
                           if rel > 0 and did in corpus)
    return weights


def dropped_train_pairs(payload):
    """Positives ``make_examples`` drops silently: document not in corpus."""
    corpus = payload["corpus"]
    train_q = payload["train_q"]
    return sum(1
               for qid, rels in payload["train_qrels"].items()
               if qid in train_q
               for did, rel in rels.items()
               if rel > 0 and did not in corpus)


def dropped_eval_qrels(payload):
    corpus = payload["corpus"]
    return sum(1
               for qid, rels in payload["eval_qrels"].items()
               if qid in payload["eval_q"]
               for did, rel in rels.items()
               if rel > 0 and did not in corpus)


def scoreable_eval_queries(payload):
    """Queries ``eval_global`` will actually score."""
    return {qid for qid in payload["eval_q"] if payload["eval_qrels"].get(qid)}


def _rng(shard_seed, parent, kind):
    digest = hashlib.sha256(
        f"{shard_seed}|{parent}|{kind}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def _assign(ids, weight, rng, n_shards):
    """LPT with a seeded random tie-break, greedy to the lightest bin."""
    ordered = sorted(ids)
    keys = {qid: rng.random() for qid in ordered}
    loads = [0] * n_shards
    assignment = {}
    for qid in sorted(ordered, key=lambda q: (-weight[q], keys[q])):
        target = min(range(n_shards), key=lambda i: (loads[i], i))
        assignment[qid] = target
        loads[target] += weight[qid]
    return assignment


def _restrict(section, qids):
    return {qid: value for qid, value in section.items() if qid in qids}


def shard_payload(payload, parent, n_shards, shard_seed):
    """Partition one payload into ``n_shards`` sub-silos.

    Returns ``[(client_name, payload), ...]`` in shard-index order. Train and
    eval queries are partitioned independently; the corpus is passed through
    by identity. Every key of every partitioned section lands in exactly one
    shard, so the union of the shards reconstructs the parent byte for byte.
    Queries carrying no in-corpus positive have weight 0 and ride along in
    the balance-neutral tail rather than being dropped, which is what keeps
    that reconstruction exact.
    """
    if n_shards < 2:
        raise ShardingError(
            f"shard_payload needs at least 2 shards; got {n_shards}")
    weights = train_pair_weights(payload)
    train_ids = set(payload["train_q"]) | set(payload["train_qrels"])
    train_weight = {qid: weights.get(qid, 0) for qid in train_ids}
    scoreable = scoreable_eval_queries(payload)
    eval_ids = set(payload["eval_q"]) | set(payload["eval_qrels"])
    eval_weight = {qid: (1 if qid in scoreable else 0) for qid in eval_ids}

    train_assign = _assign(train_ids, train_weight,
                           _rng(shard_seed, parent, "train"), n_shards)
    eval_assign = _assign(eval_ids, eval_weight,
                          _rng(shard_seed, parent, "eval"), n_shards)

    shards = []
    for index in range(n_shards):
        train_keep = {qid for qid, i in train_assign.items() if i == index}
        eval_keep = {qid for qid, i in eval_assign.items() if i == index}
        shards.append((f"{parent}-s{index}", {
            "corpus": payload["corpus"],
            "train_q": _restrict(payload["train_q"], train_keep),
            "train_qrels": _restrict(payload["train_qrels"], train_keep),
            "eval_q": _restrict(payload["eval_q"], eval_keep),
            "eval_qrels": _restrict(payload["eval_qrels"], eval_keep),
            "split_fallback": payload["split_fallback"],
        }))
    return shards


# --- guards ----------------------------------------------------------------

def assert_drop_rates(parent, rates, totals=None, max_rate=MAX_DROP_RATE,
                      tol=DROP_RATE_TOL, sigma=DROP_RATE_SIGMA):
    """Every shard of one parent loses the same small share of its pairs.

    ``totals`` is the per-shard pair count including dropped pairs; supplying
    it widens the gate by the binomial standard error a random partition of a
    fixed pool of dangling qrels produces anyway.
    """
    if not rates:
        return
    if max(rates) > max_rate:
        raise ShardingError(
            f"parent '{parent}': shard training-pair drop rate "
            f"{max(rates):.6g} exceeds {max_rate:.6g}")
    if totals and sum(totals) > 0:
        pooled = sum(rate * total for rate, total in zip(rates, totals))
        pooled /= sum(totals)
    else:
        pooled = sum(rates) / len(rates)
    for index, rate in enumerate(rates):
        slack = tol
        if totals and totals[index] > 0:
            slack += sigma * (pooled * (1.0 - pooled) / totals[index]) ** 0.5
        if abs(rate - pooled) > slack:
            raise ShardingError(
                f"parent '{parent}': shard {index} training-pair drop rate "
                f"{rate:.6g} departs from the parent rate {pooled:.6g} by "
                f"more than {slack:.6g}; the shards are not clones of one "
                "distribution")


def _assert_train_eval_disjoint(name, payload):
    overlap = set(payload["train_q"]) & set(payload["eval_q"])
    if overlap:
        raise ShardingError(
            f"client '{name}' has a train/eval query overlap of "
            f"{len(overlap)} queries (e.g. {sorted(overlap)[0]}): it would "
            "train on its own test set")


def _assert_gold_docs_present(name, payload):
    corpus = payload["corpus"]
    for qid, rels in payload["eval_qrels"].items():
        if qid not in payload["eval_q"]:
            continue
        for did, rel in rels.items():
            if rel > 0 and did not in corpus:
                raise ShardingError(
                    f"client '{name}': gold document '{did}' for eval query "
                    f"'{qid}' is absent from its retrieval corpus")


def _assert_partition(parent, payload, shards):
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        union = set()
        for name, shard in shards:
            keys = set(shard[section])
            clash = union & keys
            if clash:
                raise ShardingError(
                    f"parent '{parent}': {section} key '{sorted(clash)[0]}' "
                    f"appears in more than one shard (at '{name}')")
            union |= keys
        if union != set(payload[section]):
            missing = len(set(payload[section]) - union)
            extra = len(union - set(payload[section]))
            raise ShardingError(
                f"parent '{parent}': shard union of {section} is not the "
                f"parent ({missing} missing, {extra} unexpected); the split "
                "is a subsample, not a partition")
    for name, shard in shards:
        for other, peer in shards:
            cross = set(shard["train_q"]) & set(peer["eval_q"])
            if cross:
                raise ShardingError(
                    f"parent '{parent}': shard '{name}' trains on eval query "
                    f"'{sorted(cross)[0]}' of shard '{other}'")
        if shard["corpus"] is not payload["corpus"]:
            raise ShardingError(
                f"parent '{parent}': shard '{name}' does not share the "
                "parent corpus object; per-shard indexes change the task and "
                "can evict gold documents")


def _assert_local_work(parent, shards, caps, batch_size, min_eval_queries):
    pair_counts, epoch_steps = [], []
    for (name, shard), cap in zip(shards, caps):
        pairs = sum(train_pair_weights(shard).values())
        if pairs < batch_size:
            raise ShardingError(
                f"client '{name}' has {pairs} training pairs, fewer than "
                f"batch_size {batch_size}: its in-batch negative pool, and "
                "therefore its loss, would differ from its siblings'")
        steps = pairs // batch_size
        # Only a genuinely sharded parent owes this. The guard exists so that
        # sibling shards of one parent do the same amount of UNIQUE-data work
        # per round: a shard whose cap exceeds its epoch length recycles its
        # pairs while its siblings do not, which is what would make the shards
        # non-comparable. An unsharded parent (n_shards == 1) has no siblings,
        # and epoch recycling against a fixed step cap is exactly the regime
        # every E1 run already used -- ArguAna carries 701 in-corpus pairs, 21
        # steps per epoch at batch_size 32, against a 500-step cap. Applying
        # the guard there rejected the pre-registered E3 command line.
        if len(shards) > 1 and steps < cap:
            raise ShardingError(
                f"shard '{name}' cannot reach its step cap {cap} "
                f"({steps} steps per epoch at batch_size {batch_size}); it "
                "would recycle pairs its sibling shards do not")
        scoreable = len(scoreable_eval_queries(shard))
        if scoreable < min_eval_queries:
            raise ShardingError(
                f"client '{name}' has {scoreable} scoreable eval queries, "
                f"below the declared minimum {min_eval_queries}")
        pair_counts.append(pairs)
        epoch_steps.append(steps)
    if len(shards) > 1 and min(epoch_steps) < max(caps):
        raise ShardingError(
            f"parent '{parent}': equal local work is unachievable "
            f"(min {min(epoch_steps)} steps per epoch < max cap {max(caps)})")


# --- manifest --------------------------------------------------------------

def sharder_source_sha256():
    """Content identity of this module: a split names the code that made it."""
    digest = hashlib.sha256()
    with open(os.path.abspath(__file__), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _section_sha256(section, values):
    digest = hashlib.sha256()
    digest.update(section.encode("utf-8") + b"\0")
    for key in sorted(values, key=str):
        digest.update(str(key).encode("utf-8") + b"\0")
        digest.update(_canonical(values[key]) + b"\0")
    return digest.hexdigest()


def _ids_sha256(ids):
    return hashlib.sha256(_canonical(sorted(ids))).hexdigest()


def manifest_sha256(manifest):
    """Hash of the manifest with the hash field itself removed."""
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def write_manifest(manifest, path):
    """Write and fsync the manifest, then fsync its directory."""
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _shard_record(name, index, payload, cap, batch_size, fingerprint,
                  corpus_digest):
    pairs = sum(train_pair_weights(payload).values())
    dropped = dropped_train_pairs(payload)
    total = pairs + dropped
    return {
        "client": name,
        "shard_index": index,
        "n_train_queries": len(payload["train_q"]),
        "n_train_pairs": pairs,
        "n_eval_queries": len(payload["eval_q"]),
        "n_eval_queries_scoreable": len(scoreable_eval_queries(payload)),
        "n_corpus": len(payload["corpus"]),
        "step_cap": cap,
        "steps_per_epoch_uncapped": pairs // batch_size,
        "data_coverage_per_round": round(cap * batch_size / pairs, 10)
                                   if pairs else 0.0,
        "dropped_train_pairs_missing_doc": dropped,
        "dropped_train_pair_rate": (round(dropped / total, 12)
                                    if total else 0.0),
        "dropped_eval_qrels_missing_doc": dropped_eval_qrels(payload),
        "train_query_ids_sha256": _ids_sha256(payload["train_q"]),
        "eval_query_ids_sha256": _ids_sha256(payload["eval_q"]),
        "corpus_sha256": corpus_digest,
        "client_data_sha256": fingerprint(payload),
    }


# --- federation ------------------------------------------------------------

def build_sharded_federation(data, slices, shard_spec, shard_seed, batch_size,
                             parent_max_steps_per_round, conserve_shard_steps,
                             fingerprint, min_eval_queries=MIN_EVAL_QUERIES,
                             max_drop_rate=MAX_DROP_RATE,
                             drop_rate_tol=DROP_RATE_TOL):
    """Expand parent slices into sub-silo clients, checked and documented.

    ``fingerprint`` is the driver's own payload content hash, injected rather
    than reimplemented so the manifest records the same identity the run
    provenance does. Returns the new ``data`` dict, the expanded client list
    in ``--slices`` order, the per-client step caps and the manifest. Nothing
    in ``data`` is mutated.
    """
    if len(set(slices)) != len(slices):
        raise ShardingError("--slices names must be distinct before sharding")
    missing = [name for name in slices if name not in data]
    if missing:
        raise ShardingError(
            "slices without a loaded payload: " + ", ".join(missing))
    if (parent_max_steps_per_round > 0 and not conserve_shard_steps
            and any(count > 1 for count in shard_spec.values())):
        raise ShardingError(
            "sharding a capped client requires conserve_shard_steps: without "
            f"it each shard runs the parent's own {parent_max_steps_per_round}"
            "-step cap, multiplying that distribution's per-round "
            "optimization work by the number of shards")

    new_data, expanded, step_caps, parents = {}, [], {}, []
    for parent in slices:
        payload = data[parent]
        n_shards = int(shard_spec.get(parent, 1))
        caps = (conserved_step_caps(parent_max_steps_per_round, n_shards)
                if conserve_shard_steps
                else [parent_max_steps_per_round] * n_shards)
        _assert_train_eval_disjoint(parent, payload)
        _assert_gold_docs_present(parent, payload)
        parent_digest = fingerprint(payload)

        if n_shards == 1:
            clients = [(parent, payload)]
        else:
            clients = shard_payload(payload, parent, n_shards, shard_seed)
            _assert_partition(parent, payload, clients)
            rebuilt = {"corpus": payload["corpus"],
                       "split_fallback": payload["split_fallback"]}
            for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
                merged = {}
                for _, shard in clients:
                    merged.update(shard[section])
                rebuilt[section] = merged
            if fingerprint(rebuilt) != parent_digest:
                raise ShardingError(
                    f"parent '{parent}': the union of its shards does not "
                    "reproduce the parent payload fingerprint")

        for name, shard in clients:
            if name in new_data or (name != parent and name in data):
                raise ShardingError(
                    f"shard name '{name}' would collide with another client")
            _assert_train_eval_disjoint(name, shard)
            _assert_gold_docs_present(name, shard)
            new_data[name] = shard
            expanded.append(name)
        for (name, _), cap in zip(clients, caps):
            step_caps[name] = cap
        _assert_local_work(parent, clients, caps, batch_size,
                           min_eval_queries)
        if sum(caps) != parent_max_steps_per_round and conserve_shard_steps:
            raise ShardingError(
                f"parent '{parent}': per-shard caps {caps} do not sum to the "
                f"parent cap {parent_max_steps_per_round}")

        corpus_digest = _section_sha256("corpus", payload["corpus"])
        records = [_shard_record(name, index, shard, cap, batch_size,
                                 fingerprint, corpus_digest)
                   for index, ((name, shard), cap)
                   in enumerate(zip(clients, caps))]
        if n_shards > 1:
            assert_drop_rates(
                parent, [r["dropped_train_pair_rate"] for r in records],
                totals=[r["n_train_pairs"]
                        + r["dropped_train_pairs_missing_doc"]
                        for r in records],
                max_rate=max_drop_rate, tol=drop_rate_tol)
        if n_shards == 1 and new_data[parent] is not payload:
            raise ShardingError(
                f"unsharded client '{parent}' was not passed through intact")
        parents.append({
            "parent": parent,
            "n_shards": n_shards,
            "parent_data_sha256": parent_digest,
            "parent_n_train_queries": len(payload["train_q"]),
            "parent_n_train_pairs": sum(train_pair_weights(payload).values()),
            "parent_n_eval_queries": len(payload["eval_q"]),
            "parent_n_eval_queries_scoreable": len(
                scoreable_eval_queries(payload)),
            "parent_n_corpus": len(payload["corpus"]),
            "parent_step_cap": parent_max_steps_per_round,
            "split_fallback": bool(payload["split_fallback"]),
            "dropped_train_pairs_missing_doc": dropped_train_pairs(payload),
            "corpus_sha256": corpus_digest,
            "shards": records,
        })

    for parent in slices:
        if fingerprint(data[parent]) != [
                entry for entry in parents
                if entry["parent"] == parent][0]["parent_data_sha256"]:
            raise ShardingError(
                f"parent payload '{parent}' changed during sharding")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "shard_seed": int(shard_seed),
        "sharder_algorithm": SHARDER_ALGORITHM,
        "sharder_source_sha256": sharder_source_sha256(),
        "batch_size": int(batch_size),
        "parent_max_steps_per_round": int(parent_max_steps_per_round),
        "conserve_shard_steps": bool(conserve_shard_steps),
        "min_eval_queries": int(min_eval_queries),
        "parents": parents,
        "expanded_slices": list(expanded),
        "step_caps": dict(step_caps),
        "assertions_passed": [
            "unique_slice_names", "payloads_match_slices", "disjoint_train",
            "disjoint_eval", "train_eval_disjoint", "union_equals_parent",
            "parent_fingerprint_reconstructed", "shared_corpus_identity",
            "gold_docs_present", "dropped_rate_symmetric",
            "min_pairs_ge_batch", "equal_effective_bs",
            "eval_queries_scoreable", "step_budget_conserved",
            "singleton_unmodified", "parent_payload_unmutated",
        ],
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return ShardedFederation(data=new_data, slices=expanded,
                             step_caps=step_caps, manifest=manifest)


def partition_seed_sensitivity(payload, parent, n_shards, seeds):
    """How many DISTINCT partitions a set of shard seeds actually produces.

    The seed enters sharding only through the LPT tie-break, so a payload whose
    per-query training-pair counts are all distinct is seed-inert: every seed
    returns the same partition. A partition-robustness arm run on such a
    payload is not a robustness check, it is the same run repeated at full GPU
    price. Call this before committing those runs and report the number.

    Returns ``(distinct, signatures)`` where ``signatures`` maps each seed to
    the partition it produced, so an inert payload is visible rather than
    inferred.
    """
    signatures = {}
    for seed in seeds:
        shards = shard_payload(payload, parent, n_shards, int(seed))
        signatures[int(seed)] = tuple(
            _ids_sha256(sorted(shard["train_q"])) for _, shard in shards)
    return len(set(signatures.values())), signatures
