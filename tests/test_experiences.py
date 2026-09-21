"""experiences.py: a constructed semantic-shift stream, built and verified from manifests."""
import json

import pytest

import experiences

TEXTS = ([f"apple pie recipe {i}" for i in range(30)] + [f"quantum physics lecture {i}" for i in range(30)]
         + [f"mortgage interest rate {i}" for i in range(30)] + [f"football match score {i}" for i in range(30)])


def test_clustering_is_deterministic_and_fitted_on_training_text_only():
    first = experiences.cluster_queries(TEXTS, k=4, seed=3)
    again = experiences.cluster_queries(TEXTS, k=4, seed=3)
    assert first["assignment"] == again["assignment"]
    assert first["record"]["centroid_sha256"] == again["record"]["centroid_sha256"]
    assert sorted(set(first["assignment"])) == [0, 1, 2, 3]
    held_out = experiences.assign_queries(first, ["apple pie recipe 99", "football match score 99"])
    assert held_out[0] == first["assignment"][0] and held_out[1] == first["assignment"][-1]


def test_too_small_sub_cluster_reruns_with_the_next_seed_then_fails(monkeypatch):
    calls = []
    def tiny(texts, k, seed):
        calls.append(seed)
        return {"assignment": [0] * (len(texts) - 1) + [1], "record": {"seed": seed}}
    monkeypatch.setattr(experiences, "cluster_queries", tiny)
    with pytest.raises(ValueError, match="sub-cluster"):
        experiences.cluster_with_minimum(TEXTS, k=2, seed=10, minimum=5)
    assert calls == [10, 11, 12, 13, 14, 15]


def test_uniform_counts_are_the_largest_every_cell_supports():
    table = {("c0", 0): {"train_eligible": 3000, "eval_eligible": 900},
             ("c0", 1): {"train_eligible": 1500, "eval_eligible": 400},
             ("c1", 0): {"train_eligible": 5000, "eval_eligible": 2000}}
    counts = experiences.choose_counts(table, train_cap=2000, guard_cap=200, test_cap=500)
    assert counts == {"train": 1364, "guard": 136, "test": 400}


def synthetic_msmarco(tmp_path):
    root = tmp_path / "msmarco-passage"; root.mkdir()
    n_q = 120
    lines = []
    for i in range(400):
        lines.append(f"{i}\tpassage number {i} about {'apples' if i % 2 else 'physics'}")
    (root / "collection.tsv").write_text("\n".join(lines) + "\n")
    (root / "queries.train.tsv").write_text("".join(f"{100 + i}\t{TEXTS[i]}\n" for i in range(n_q)))
    (root / "qrels.train.tsv").write_text("".join(f"{100 + i}\t0\t{i}\t1\n" for i in range(n_q)))
    shift = tmp_path / "ms-marco-shift" / "TRAIN"; shift.mkdir(parents=True)
    rows = ["query\ttopic_cluster\ttopic_train\twhword_cluster\twhword_train\tlength_cluster\tlength_train"]
    rows += [f" {TEXTS[i]}\t{i % 2}\t1\t0\t1\t0\t1" for i in range(n_q)]
    (shift / "queries_clustering.tsv").write_text("\n".join(rows) + "\n")
    ev = tmp_path / "ms-marco-shift" / "EVAL"; (ev / "queries").mkdir(parents=True); (ev / "qrel").mkdir()
    for t in (0, 1):
        qs = [(f"{900 + t * 10 + j}", TEXTS[t + 16 * j] + " eval") for j in range(8)]
        (ev / "queries" / f"queries_{t}.tsv").write_text("".join(f"{q}\t{text}\n" for q, text in qs))
        (ev / "qrel" / f"qrel_{t}.json").write_text(json.dumps({q: {str(300 + j): 1} for j, (q, _) in enumerate(qs)}))
    return tmp_path


def fake_bm25(query_texts, k):
    return [[str(200 + sum(map(ord, t)) % 50 + j) for j in range(k)] for t in query_texts]


def test_manifest_splits_are_disjoint_and_the_corpus_follows_the_rule(tmp_path):
    root = synthetic_msmarco(tmp_path)
    manifest = experiences.build_manifest(
        root, clients=(0, 1), experiences_per_client=2, schedule={0: (0, 1), 1: (1, 0)},
        counts={"train": 20, "guard": 4, "test": 3}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25)
    for client in manifest["clients"].values():
        seen = set()
        for cell in client["experiences"].values():
            ids = set(cell["train"]) | set(cell["guard"]) | set(cell["test"])
            assert len(cell["train"]) == 20 and len(cell["guard"]) == 4 and len(cell["test"]) == 3
            assert not (seen & ids)
            seen |= ids
        assert len(client["corpus"]) == 120 == len(set(client["corpus"]))
        relevant = {pid for cell in client["experiences"].values()
                    for pids in cell["qrels"].values() for pid in pids}
        assert relevant <= set(client["corpus"])
    assert manifest["clients"]["0"]["order"] == [0, 1] and manifest["clients"]["1"]["order"] == [1, 0]
    assert experiences.verify_manifest(manifest) is None
    manifest["clients"]["0"]["corpus"][0] = "999"
    with pytest.raises(ValueError, match="digest"):
        experiences.verify_manifest(manifest)


def test_materialised_cell_has_the_shape_the_trainer_consumes(tmp_path):
    root = synthetic_msmarco(tmp_path)
    manifest = experiences.build_manifest(
        root, clients=(0,), experiences_per_client=2, schedule={0: (0, 1)},
        counts={"train": 20, "guard": 4, "test": 3}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25)
    data = experiences.materialise(manifest, root, client="0", experience=1)
    assert set(data) == {"corpus", "train_q", "train_qrels", "guard_q", "guard_qrels", "test_q", "test_qrels"}
    assert len(data["corpus"]) == 120 and set(data["train_q"]) == set(data["train_qrels"])
    assert all(pid in data["corpus"] for rels in data["train_qrels"].values() for pid in rels)
    assert all("text" in doc for doc in data["corpus"].values())


def test_a_topic_can_be_split_into_pseudo_clients(tmp_path):
    root = synthetic_msmarco(tmp_path)
    manifest = experiences.build_manifest(
        root, clients=(0,), experiences_per_client=1, schedule={0: (0,), 1: (0,)},
        counts={"train": 8, "guard": 2, "test": 1}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25, pseudo_clients=2)
    assert set(manifest["clients"]) == {"0", "1"}
    assert all(client["topic"] == 0 for client in manifest["clients"].values())
    train = [set(client["experiences"]["0"]["train"]) for client in manifest["clients"].values()]
    assert not (train[0] & train[1])


def test_a_topic_without_official_evaluation_queries_holds_out_training_queries(tmp_path):
    root = synthetic_msmarco(tmp_path)
    for name in ("queries/queries_1.tsv", "qrel/qrel_1.json"):
        (root / "ms-marco-shift" / "EVAL" / name).unlink()
    manifest = experiences.build_manifest(
        root, clients=(1,), experiences_per_client=1, schedule={1: (0,)},
        counts={"train": 20, "guard": 4, "test": 3}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25)
    client = manifest["clients"]["1"]
    assert client["eval_source"] == "train-holdout" and len(client["eval_pool"]) == 9
    assert set(client["experiences"]["0"]["test"]) <= set(client["eval_pool"])
    assert not (set(client["eval_pool"]) & set(client["experiences"]["0"]["train"]))
    queries, qrels = experiences.eval_queries(manifest, root, "1")
    assert set(queries) == set(client["eval_pool"]) == set(qrels)


def test_bm25_index_is_built_once_and_reloaded(tmp_path):
    pytest.importorskip("bm25s")
    root = synthetic_msmarco(tmp_path)
    first = experiences.bm25_retriever(root)
    assert (root / "msmarco-passage" / "bm25s_index" / "ids.json").exists()
    second = experiences.bm25_retriever(root)
    queries = ["passage about apples", "passage about physics"]
    assert first(queries, 3) == second(queries, 3)
    assert all(len(hits) == 3 for hits in second(queries, 3))


def test_schedules_share_one_build_and_differ_only_in_order(tmp_path):
    root = synthetic_msmarco(tmp_path)
    manifests = experiences.build_manifests(
        root, clients=(0, 1), experiences_per_client=2,
        schedules={"A": {0: (0, 1), 1: (1, 0)}, "B": {0: (1, 0), 1: (0, 1)}},
        counts={"train": 20, "guard": 4, "test": 3}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25)
    a, b = manifests["A"], manifests["B"]
    assert a["schedule"] == "A" and b["schedule"] == "B"
    assert a["clients"]["0"]["order"] == [0, 1] and b["clients"]["0"]["order"] == [1, 0]
    for client in ("0", "1"):
        assert a["clients"][client]["digests"] == b["clients"][client]["digests"]
        assert a["clients"][client]["corpus"] == b["clients"][client]["corpus"]
    experiences.verify_manifest(b)
