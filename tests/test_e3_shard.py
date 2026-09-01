"""E3 clone-federation sharding: partition, eval design, budget, provenance."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e3_fixtures import (nfcorpus_shaped_payload, payload_pairs,  # noqa: E402
                         scoreable_eval_queries, small_payload)
from e3_shard import (MANIFEST_SCHEMA_VERSION, SHARDER_ALGORITHM,  # noqa: E402
                      ShardingError, assert_drop_rates,
                      build_sharded_federation,
                      conserved_step_caps, manifest_sha256, parse_shard_spec,
                      partition_seed_sensitivity,
                      shard_payload, sharder_source_sha256,
                      train_pair_weights, write_manifest)
from federated_forgetting import _data_fingerprints  # noqa: E402

BATCH_SIZE = 32
PARENT_CAP = 500


def fingerprint(payload):
    return _data_fingerprints({"x": payload})["x"]


@pytest.fixture(scope="module")
def parent():
    return nfcorpus_shaped_payload(seed=0)


@pytest.fixture(scope="module")
def singleton():
    return nfcorpus_shaped_payload(seed=7, n_train=703, n_eval=703, n_docs=800)


@pytest.fixture(scope="module")
def shards(parent):
    return shard_payload(parent, "nfcorpus", 3, shard_seed=42)


def federation(parent, singleton, shard_seed=42, n_shards=3,
               conserve=True, cap=PARENT_CAP, min_eval_queries=50):
    data = {"nfcorpus": parent, "arguana": singleton}
    return build_sharded_federation(
        data, ["nfcorpus", "arguana"], {"nfcorpus": n_shards},
        shard_seed=shard_seed, batch_size=BATCH_SIZE,
        parent_max_steps_per_round=cap, conserve_shard_steps=conserve,
        fingerprint=fingerprint, min_eval_queries=min_eval_queries)


# --- partition correctness -------------------------------------------------

def test_pair_count_conserved(parent, shards):
    total = sum(payload_pairs(payload) for _, payload in shards)
    assert total == payload_pairs(parent)


def test_pair_balance(shards):
    counts = [payload_pairs(payload) for _, payload in shards]
    assert max(counts) / min(counts) <= 1.02


def test_queries_atomic(parent, shards):
    for name, payload in shards:
        for qid in payload["train_q"]:
            if qid in parent["train_qrels"]:
                assert payload["train_qrels"][qid] == parent["train_qrels"][qid]


def test_train_disjoint(shards):
    seen = set()
    for _, payload in shards:
        keys = set(payload["train_q"])
        assert seen.isdisjoint(keys)
        seen |= keys


def test_eval_disjoint(shards):
    seen = set()
    for _, payload in shards:
        keys = set(payload["eval_q"])
        assert seen.isdisjoint(keys)
        seen |= keys


def test_train_eval_disjoint(shards):
    for _, left in shards:
        for _, right in shards:
            assert set(left["train_q"]).isdisjoint(set(right["eval_q"]))


def test_union_equals_parent(parent, shards):
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        union = set()
        for _, payload in shards:
            union |= set(payload[section])
        assert union == set(parent[section]), section


def test_parent_fingerprint_reconstructed(parent, shards):
    rebuilt = {"corpus": parent["corpus"], "split_fallback": False}
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        merged = {}
        for _, payload in shards:
            merged.update(payload[section])
        rebuilt[section] = merged
    assert fingerprint(rebuilt) == fingerprint(parent)


def test_no_parent_mutation(parent):
    before = copy.deepcopy(parent)
    shard_payload(parent, "nfcorpus", 3, shard_seed=1234)
    assert parent == before


def test_shard_names_and_order(shards):
    assert [name for name, _ in shards] == [
        "nfcorpus-s0", "nfcorpus-s1", "nfcorpus-s2"]


# --- eval semantics --------------------------------------------------------

def test_shared_corpus_identity(parent, shards):
    for _, payload in shards:
        assert payload["corpus"] is parent["corpus"]


def test_eval_query_split_sizes(parent, shards):
    sizes = sorted(len(scoreable_eval_queries(p)) for _, p in shards)
    total = len(scoreable_eval_queries(parent))
    assert sum(sizes) == total
    assert sizes[-1] - sizes[0] <= 1


def test_micro_average_reconstructs(parent, shards):
    per_query = {qid: (i % 97) / 97.0
                 for i, qid in enumerate(sorted(parent["eval_q"]))}
    scoreable = scoreable_eval_queries(parent)
    monolithic = (sum(per_query[q] for q in scoreable) / len(scoreable))
    weighted, weight = 0.0, 0
    for _, payload in shards:
        qids = scoreable_eval_queries(payload)
        weighted += sum(per_query[q] for q in qids)
        weight += len(qids)
    assert weight == len(scoreable)
    assert abs(weighted / weight - monolithic) < 1e-12


def test_all_gold_docs_present(shards):
    for _, payload in shards:
        for qid, rels in payload["eval_qrels"].items():
            for did, rel in rels.items():
                if rel > 0:
                    assert did in payload["corpus"]


# --- step budget -----------------------------------------------------------

@pytest.mark.parametrize("cap,m,expected", [
    (500, 3, [167, 167, 166]),
    (500, 4, [125, 125, 125, 125]),
    (7, 3, [3, 2, 2]),
    (1, 3, [1, 0, 0]),
    (0, 3, [0, 0, 0]),
    (500, 1, [500]),
])
def test_conserved_caps_sum(cap, m, expected):
    caps = conserved_step_caps(cap, m)
    assert caps == expected
    assert sum(caps) == cap


def test_coverage_fraction_preserved(parent, singleton):
    fed = federation(parent, singleton)
    parent_pairs = payload_pairs(parent)
    parent_coverage = PARENT_CAP * BATCH_SIZE / parent_pairs
    entry = fed.manifest["parents"][0]
    for shard in entry["shards"]:
        assert abs(shard["data_coverage_per_round"] - parent_coverage) < 0.001


def test_steps_per_epoch_ge_cap(parent, singleton):
    fed = federation(parent, singleton)
    entry = fed.manifest["parents"][0]
    for shard in entry["shards"]:
        assert shard["steps_per_epoch_uncapped"] >= shard["step_cap"]


def test_equal_effective_batch_size(parent, singleton):
    fed = federation(parent, singleton)
    for name in fed.slices:
        pairs = payload_pairs(fed.data[name])
        assert pairs >= BATCH_SIZE


def test_step_caps_sum_to_parent_cap(parent, singleton):
    fed = federation(parent, singleton)
    clone_caps = [fed.step_caps[n] for n in fed.slices if n.startswith("nfcorpus")]
    assert sum(clone_caps) == PARENT_CAP
    assert fed.step_caps["arguana"] == PARENT_CAP


def test_unconserved_caps_rejected(parent, singleton):
    with pytest.raises(ShardingError, match="conserve"):
        federation(parent, singleton, conserve=False)


def test_unconserved_caps_allowed_without_a_cap(parent, singleton):
    fed = federation(parent, singleton, conserve=False, cap=0)
    assert set(fed.step_caps.values()) == {0}


# --- randomization and reproducibility -------------------------------------

def test_same_seed_reproduces(parent):
    first = shard_payload(parent, "nfcorpus", 3, shard_seed=42)
    second = shard_payload(parent, "nfcorpus", 3, shard_seed=42)
    assert [sorted(p["train_q"]) for _, p in first] == \
           [sorted(p["train_q"]) for _, p in second]


def test_different_seeds_give_different_splits(parent):
    splits = set()
    for seed in (42, 123, 2024, 7, 99):
        parts = shard_payload(parent, "nfcorpus", 3, shard_seed=seed)
        splits.add(tuple(tuple(sorted(p["train_q"])) for _, p in parts))
    assert len(splits) == 5


def test_different_seeds_keep_perfect_balance(parent):
    for seed in (42, 123, 2024, 7, 99):
        counts = [payload_pairs(p)
                  for _, p in shard_payload(parent, "nfcorpus", 3, seed)]
        assert max(counts) / min(counts) <= 1.02


def test_seed_reoverlap_is_near_chance(parent):
    base = set(shard_payload(parent, "nfcorpus", 3, 42)[0][1]["train_q"])
    other = set(shard_payload(parent, "nfcorpus", 3, 123)[0][1]["train_q"])
    overlap = len(base & other) / len(base)
    assert 0.2 < overlap < 0.5


def test_eval_split_is_seed_sensitive(parent):
    first = shard_payload(parent, "nfcorpus", 3, 42)[0][1]["eval_q"]
    second = shard_payload(parent, "nfcorpus", 3, 123)[0][1]["eval_q"]
    assert set(first) != set(second)


def _with_degenerate_queries(parent):
    """Queries that contribute no training pair, and rows without a partner."""
    payload = {"corpus": parent["corpus"],
               "split_fallback": parent["split_fallback"]}
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        payload[section] = dict(parent[section])
    for i in range(5):
        payload["train_q"][f"ZERO-{i}"] = "only out-of-corpus positives"
        payload["train_qrels"][f"ZERO-{i}"] = {f"absent-{i}": 1}
    for i in range(3):
        payload["train_q"][f"NOQRELS-{i}"] = "no qrels row at all"
    for i in range(2):
        payload["train_qrels"][f"ORPHAN-{i}"] = {sorted(parent["corpus"])[0]: 1}
    for i in range(4):
        payload["eval_q"][f"EMPTY-{i}"] = "empty qrels"
        payload["eval_qrels"][f"EMPTY-{i}"] = {}
    return payload


def test_zero_weight_queries_are_partitioned_not_dropped(parent):
    payload = _with_degenerate_queries(parent)
    shards = shard_payload(payload, "nfcorpus", 3, 42)
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        union = set()
        for _, shard in shards:
            keys = set(shard[section])
            assert union.isdisjoint(keys)
            union |= keys
        assert union == set(payload[section]), section
    rebuilt = {"corpus": payload["corpus"], "split_fallback": False}
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        merged = {}
        for _, shard in shards:
            merged.update(shard[section])
        rebuilt[section] = merged
    assert fingerprint(rebuilt) == fingerprint(payload)


def test_zero_weight_queries_do_not_disturb_the_balance(parent):
    payload = _with_degenerate_queries(parent)
    counts = [payload_pairs(p)
              for _, p in shard_payload(payload, "nfcorpus", 3, 42)]
    assert max(counts) / min(counts) <= 1.02


def test_split_is_independent_of_payload_key_order(parent):
    reversed_payload = {
        "corpus": parent["corpus"],
        "split_fallback": parent["split_fallback"],
    }
    for section in ("train_q", "train_qrels", "eval_q", "eval_qrels"):
        reversed_payload[section] = {
            qid: parent[section][qid]
            for qid in reversed(list(parent[section]))}
    left = shard_payload(parent, "nfcorpus", 3, 42)
    right = shard_payload(reversed_payload, "nfcorpus", 3, 42)
    assert [sorted(p["train_q"]) for _, p in left] == \
           [sorted(p["train_q"]) for _, p in right]
    assert [sorted(p["eval_q"]) for _, p in left] == \
           [sorted(p["eval_q"]) for _, p in right]


def test_weights_match_make_examples_counting(parent):
    weights = train_pair_weights(parent)
    assert sum(weights.values()) == payload_pairs(parent)


# --- manifest and provenance ----------------------------------------------

def test_manifest_reproducible(parent, singleton):
    left = federation(parent, singleton).manifest
    right = federation(parent, singleton).manifest
    assert left["manifest_sha256"] == right["manifest_sha256"]
    assert left == right


def test_manifest_seed_sensitive(parent, singleton):
    left = federation(parent, singleton, shard_seed=42).manifest
    right = federation(parent, singleton, shard_seed=123).manifest
    assert left["manifest_sha256"] != right["manifest_sha256"]


def test_manifest_sha256_covers_the_object(parent, singleton):
    manifest = federation(parent, singleton).manifest
    recomputed = manifest_sha256(manifest)
    assert recomputed == manifest["manifest_sha256"]
    tampered = json.loads(json.dumps(manifest))
    tampered["parents"][0]["shards"][0]["n_train_pairs"] += 1
    assert manifest_sha256(tampered) != manifest["manifest_sha256"]


def test_manifest_required_fields(parent, singleton):
    manifest = federation(parent, singleton).manifest
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["sharder_algorithm"] == SHARDER_ALGORITHM
    assert manifest["sharder_source_sha256"] == sharder_source_sha256()
    assert manifest["shard_seed"] == 42
    assert manifest["batch_size"] == BATCH_SIZE
    assert manifest["parent_max_steps_per_round"] == PARENT_CAP
    assert manifest["conserve_shard_steps"] is True
    assert manifest["expanded_slices"] == [
        "nfcorpus-s0", "nfcorpus-s1", "nfcorpus-s2", "arguana"]
    assert manifest["step_caps"] == {
        "nfcorpus-s0": 167, "nfcorpus-s1": 167, "nfcorpus-s2": 166,
        "arguana": 500}
    entry = manifest["parents"][0]
    assert entry["parent"] == "nfcorpus"
    assert entry["n_shards"] == 3
    assert entry["parent_data_sha256"] == fingerprint(parent)
    assert entry["parent_n_train_pairs"] == payload_pairs(parent)
    assert entry["parent_n_corpus"] == len(parent["corpus"])
    for shard in entry["shards"]:
        for field in ("client", "shard_index", "n_train_queries",
                      "n_train_pairs", "n_eval_queries", "n_corpus",
                      "step_cap", "steps_per_epoch_uncapped",
                      "data_coverage_per_round",
                      "dropped_train_pairs_missing_doc",
                      "dropped_eval_qrels_missing_doc",
                      "train_query_ids_sha256", "eval_query_ids_sha256",
                      "corpus_sha256", "client_data_sha256"):
            assert field in shard, field
    assert len({s["corpus_sha256"] for s in entry["shards"]}) == 1
    assert len({s["client_data_sha256"] for s in entry["shards"]}) == 3


def test_manifest_records_client_fingerprints(parent, singleton):
    fed = federation(parent, singleton)
    recorded = {shard["client"]: shard["client_data_sha256"]
                for entry in fed.manifest["parents"]
                for shard in entry["shards"]}
    for name, payload in fed.data.items():
        assert recorded[name] == fingerprint(payload)


def test_manifest_assertions_recorded(parent, singleton):
    manifest = federation(parent, singleton).manifest
    for name in ("disjoint_train", "disjoint_eval", "train_eval_disjoint",
                 "union_equals_parent", "parent_fingerprint_reconstructed",
                 "shared_corpus_identity", "min_pairs_ge_batch",
                 "equal_effective_bs", "step_budget_conserved",
                 "singleton_unmodified", "unique_slice_names",
                 "gold_docs_present", "dropped_rate_symmetric",
                 "eval_queries_scoreable"):
        assert name in manifest["assertions_passed"], name


def test_write_manifest_is_durable(tmp_path, parent, singleton):
    manifest = federation(parent, singleton).manifest
    path = tmp_path / "m.json"
    write_manifest(manifest, str(path))
    assert json.loads(path.read_text()) == manifest


def test_dropped_pairs_counted(singleton):
    parent = nfcorpus_shaped_payload(seed=3, n_train=600, n_eval=200,
                                     n_docs=900, missing_docs=100)
    data = {"nfcorpus": parent, "arguana": singleton}
    fed = build_sharded_federation(
        data, ["nfcorpus", "arguana"], {"nfcorpus": 3}, shard_seed=42,
        batch_size=BATCH_SIZE, parent_max_steps_per_round=0,
        conserve_shard_steps=True, fingerprint=fingerprint)
    entry = fed.manifest["parents"][0]
    dropped = [s["dropped_train_pairs_missing_doc"] for s in entry["shards"]]
    assert sum(dropped) == 100
    assert entry["dropped_train_pairs_missing_doc"] == 100
    assert all(s["dropped_train_pair_rate"] < 0.01 for s in entry["shards"])


def test_asymmetric_drop_rate_rejected():
    with pytest.raises(ShardingError, match="drop rate"):
        assert_drop_rates("nfcorpus", [0.0, 0.0, 0.008])


def test_high_drop_rate_rejected():
    with pytest.raises(ShardingError, match="drop rate"):
        assert_drop_rates("nfcorpus", [0.03, 0.03, 0.03])


def test_balanced_small_drop_rates_accepted():
    assert_drop_rates("nfcorpus", [0.0069, 0.00691, 0.00689])


# --- singleton and guards --------------------------------------------------

def test_singleton_untouched(parent, singleton):
    fed = federation(parent, singleton)
    assert fed.data["arguana"] is singleton
    assert "arguana" in fed.slices
    assert not any(name.startswith("arguana-s") for name in fed.slices)
    entry = [e for e in fed.manifest["parents"] if e["parent"] == "arguana"][0]
    assert entry["n_shards"] == 1
    assert entry["parent_data_sha256"] == fingerprint(singleton)


def test_unsharded_slices_pass_through(parent, singleton):
    fed = federation(parent, singleton)
    assert fed.slices == ["nfcorpus-s0", "nfcorpus-s1", "nfcorpus-s2",
                          "arguana"]
    assert set(fed.data) == set(fed.slices)


@pytest.mark.parametrize("tokens,message", [
    (["nfcorpus"], "PARENT:N"),
    (["nfcorpus:0"], "at least 1"),
    (["nfcorpus:x"], "PARENT:N"),
    (["nfcorpus:3", "nfcorpus:2"], "repeated"),
    (["absent:3"], "not in --slices"),
])
def test_parse_shard_spec_rejects(tokens, message):
    with pytest.raises(ShardingError, match=message):
        parse_shard_spec(tokens, ["nfcorpus", "arguana"])


def test_parse_shard_spec_accepts():
    assert parse_shard_spec(["nfcorpus:3"], ["nfcorpus", "arguana"]) == \
        {"nfcorpus": 3}


def test_shard_name_collision_rejected(parent, singleton):
    data = {"nfcorpus": parent, "nfcorpus-s0": singleton}
    with pytest.raises(ShardingError, match="collide"):
        build_sharded_federation(
            data, ["nfcorpus", "nfcorpus-s0"], {"nfcorpus": 3}, shard_seed=42,
            batch_size=BATCH_SIZE, parent_max_steps_per_round=0,
            conserve_shard_steps=True, fingerprint=fingerprint)


def test_too_few_pairs_rejected(singleton):
    tiny = small_payload()
    data = {"tiny": tiny, "arguana": singleton}
    with pytest.raises(ShardingError, match="training pairs"):
        build_sharded_federation(
            data, ["tiny", "arguana"], {"tiny": 3}, shard_seed=42,
            batch_size=BATCH_SIZE, parent_max_steps_per_round=0,
            conserve_shard_steps=True, fingerprint=fingerprint)


def test_too_few_eval_queries_rejected(parent, singleton):
    with pytest.raises(ShardingError, match="scoreable eval"):
        federation(parent, singleton, n_shards=7, min_eval_queries=50)


def test_train_eval_overlap_in_parent_rejected(parent, singleton):
    leaky = dict(singleton)
    leaky["eval_q"] = dict(singleton["eval_q"])
    leaky["eval_qrels"] = dict(singleton["eval_qrels"])
    qid = sorted(singleton["train_q"])[0]
    leaky["eval_q"][qid] = "leak"
    leaky["eval_qrels"][qid] = {sorted(singleton["corpus"])[0]: 1}
    data = {"nfcorpus": parent, "arguana": leaky}
    with pytest.raises(ShardingError, match="train/eval"):
        build_sharded_federation(
            data, ["nfcorpus", "arguana"], {"nfcorpus": 3}, shard_seed=42,
            batch_size=BATCH_SIZE, parent_max_steps_per_round=0,
            conserve_shard_steps=True, fingerprint=fingerprint)


def test_missing_gold_doc_rejected(parent, singleton):
    broken = dict(singleton)
    broken["eval_qrels"] = dict(singleton["eval_qrels"])
    qid = sorted(singleton["eval_qrels"])[0]
    broken["eval_qrels"][qid] = dict(broken["eval_qrels"][qid])
    broken["eval_qrels"][qid]["not-in-corpus"] = 1
    data = {"nfcorpus": parent, "arguana": broken}
    with pytest.raises(ShardingError, match="gold document"):
        build_sharded_federation(
            data, ["nfcorpus", "arguana"], {"nfcorpus": 3}, shard_seed=42,
            batch_size=BATCH_SIZE, parent_max_steps_per_round=0,
            conserve_shard_steps=True, fingerprint=fingerprint)


def test_steps_per_epoch_below_cap_rejected(parent, singleton):
    with pytest.raises(ShardingError, match="cannot reach"):
        federation(parent, singleton, n_shards=3, cap=5000)


def test_duplicate_slice_names_rejected(parent, singleton):
    data = {"nfcorpus": parent, "arguana": singleton}
    with pytest.raises(ShardingError, match="distinct"):
        build_sharded_federation(
            data, ["nfcorpus", "nfcorpus", "arguana"], {"nfcorpus": 3},
            shard_seed=42, batch_size=BATCH_SIZE,
            parent_max_steps_per_round=0, conserve_shard_steps=True,
            fingerprint=fingerprint)


def test_missing_payload_rejected(parent):
    with pytest.raises(ShardingError, match="payload"):
        build_sharded_federation(
            {"nfcorpus": parent}, ["nfcorpus", "arguana"], {"nfcorpus": 3},
            shard_seed=42, batch_size=BATCH_SIZE,
            parent_max_steps_per_round=0, conserve_shard_steps=True,
            fingerprint=fingerprint)


# --- Regression from the pre-E3 verification pass (2026-08-31) --------------

def test_unsharded_singleton_below_its_step_cap_is_accepted():
    """The E3 federation's singleton must not be judged by the shard rule.

    ArguAna carries 701 in-corpus training pairs (MEASURED in E1) = 21 steps
    per epoch at batch_size 32, against the 500-step parent cap every E1 run
    already used: the client simply recycles its epoch, as it always has. The
    equal-local-work guard exists to keep SIBLING SHARDS of one parent doing
    the same amount of unique-data work, and a parent with n_shards == 1 has
    no siblings. Applying it there made the pre-registered E3 command line
    exit 2 -- the experiment could not start.
    """
    data = {
        "nfcorpus": nfcorpus_shaped_payload(),
        "arguana": small_payload(n_train=703, n_eval=200, positives=1,
                                 n_docs=8674),
    }
    slices = ["nfcorpus", "arguana"]

    assert sum(train_pair_weights(data["arguana"]).values()) // 32 < 500, (
        "the fixture no longer reproduces ArguAna's sub-cap epoch length")

    new_data, clients, caps, manifest = build_sharded_federation(
        data, slices, shard_spec=parse_shard_spec(["nfcorpus:3", "arguana:1"], slices),
        shard_seed=42,
        batch_size=32, parent_max_steps_per_round=500,
        conserve_shard_steps=True, fingerprint=fingerprint)

    assert clients == ["nfcorpus-s0", "nfcorpus-s1", "nfcorpus-s2",
                       "arguana"]
    assert caps["arguana"] == 500, "the singleton keeps the full parent cap"


def test_a_real_shard_below_its_cap_is_still_refused():
    """The guard the previous test relaxes must still bite where it belongs."""
    data = {"tiny": small_payload(n_train=300, n_eval=200, positives=1,
                                  n_docs=900)}
    with pytest.raises(ShardingError, match="cannot reach its step cap"):
        build_sharded_federation(
            data, ["tiny"],
            shard_spec=parse_shard_spec(["tiny:3"], ["tiny"]), shard_seed=42,
            batch_size=32, parent_max_steps_per_round=300,
            conserve_shard_steps=True, fingerprint=fingerprint)


def test_seed_inert_payloads_are_detectable_before_the_gpu_spend():
    """E3's partition-robustness arm must be able to prove it varies anything.

    The shard seed only breaks LPT ties, so a payload with all-distinct
    per-query pair counts yields one partition no matter the seed. Four such
    runs cost four runs and buy one.
    """
    seeds = [42, 123, 2024, 7, 99]

    # Query i carries exactly i+1 DISTINCT positives, so no two queries tie
    # and the LPT ordering is fully determined before the tie-break is read.
    all_distinct = small_payload(n_train=60, n_eval=40, positives=60,
                                 n_docs=200)
    for index, qid in enumerate(sorted(all_distinct["train_qrels"])):
        docs = sorted(all_distinct["train_qrels"][qid])[:index + 1]
        all_distinct["train_qrels"][qid] = {doc: 1 for doc in docs}
    weights = sorted(train_pair_weights(all_distinct).values())
    assert len(set(weights)) == len(weights), (
        "the fixture failed to make every per-query weight distinct")

    inert, _ = partition_seed_sensitivity(all_distinct, "x", 3, seeds)
    assert inert == 1, "distinct weights must be seed-inert; the probe is wrong"

    varied, signatures = partition_seed_sensitivity(
        nfcorpus_shaped_payload(), "nfcorpus", 3, seeds)
    assert varied > 1, "a tie-rich payload must actually vary with the seed"
    assert len(signatures) == len(seeds)
