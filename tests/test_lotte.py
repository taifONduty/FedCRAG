"""lotte.py: two natural experiences per topic, read by the unchanged driver and validator."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continual_driver as driver  # noqa: E402
import lotte  # noqa: E402
import validate_continual  # noqa: E402
from test_continual_driver import CLEAN_COMMIT, Fakes, fake_encode, initial_state  # noqa: E402

TOPICS = ("cooking", "gaming")
COUNTS = {"train": 10, "guard": 4, "test": 6}
SCHEDULES = {"A": {"0": [0, 1], "1": [1, 0]}}


def synthetic_lotte(tmp_path):
    for topic in TOPICS:
        for split in lotte.SPLITS:
            d = tmp_path / "lotte" / "lotte" / topic / split
            d.mkdir(parents=True)
            (d / "collection.tsv").write_text(
                "".join(f"{i}\t{topic} {split} passage {i}\n" for i in range(200)))
            meta = []
            for kind, base in (("search", 0), ("forum", 100)):
                rows = [{"qid": i, "query": f"{topic} {split} {kind} question {i}",
                         "answer_pids": [base + 2 * i, base + 2 * i + 1]} for i in range(15)]
                (d / f"qas.{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                # the second answer has more votes, except for question 0, a tie
                meta += [{"post_ids": r["answer_pids"], "scores": [2, 2] if r["qid"] == 0 else [1, 3]}
                         for r in rows]
            (d / "metadata.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    return tmp_path


def fake_retriever(topic):
    def retrieve(texts, k):
        rows = []
        for t in texts:
            h = sum(map(ord, t))
            rows.append(["dev-50", "dev-51", "dev-52", f"dev-{53 + h % 40}",
                         f"dev-{140 + h % 40}"][:k])
        return rows
    return retrieve


def build(root, corpus_size):
    return lotte.build_manifests(root, SCHEDULES, COUNTS, corpus_size, 3, fake_retriever,
                                 topics=TOPICS)["A"]


def test_experiences_are_the_two_forum_groups_and_training_keeps_the_top_answer(tmp_path):
    root = synthetic_lotte(tmp_path)
    manifest = build(root, 200)
    assert manifest["hard_k"] == 5
    for c, client in manifest["clients"].items():
        seen = set()
        for e, cell in client["experiences"].items():
            ids = cell["train"] + cell["guard"] + cell["test"]
            assert all(q.startswith(lotte.SPLITS[int(e)]) for q in ids)
            assert not seen & set(ids)
            seen |= set(ids)
            for q in cell["train"]:
                split, kind, i = q.split("-")
                base = 0 if kind == "search" else 100
                top = base + 2 * int(i) + (0 if i == "0" else 1)
                assert cell["qrels"][q] == {f"{split}-{top}": 1}
            for q in cell["guard"] + cell["test"]:
                assert len(cell["qrels"][q]) == 2
        assert len(client["corpus"]) == 200 and client["order"] == SCHEDULES["A"][c]
        answers = {p for cell in client["experiences"].values() for rels in cell["qrels"].values()
                   for p in rels}
        assert answers <= set(client["corpus"])


def test_hard_passages_fall_back_to_the_top_three_only_when_the_top_five_overflow(tmp_path):
    root = synthetic_lotte(tmp_path)
    manifest = build(root, 90)
    assert manifest["hard_k"] == 3
    assert all(client["corpus_parts"]["hard"] == 3 for client in manifest["clients"].values())
    with pytest.raises(lotte.CorpusOverflow):
        build(root, 82)


def test_the_driver_and_validator_run_a_lotte_manifest_unchanged(tmp_path, monkeypatch):
    root = synthetic_lotte(tmp_path)
    path = tmp_path / "lotte_A.json"
    path.write_text(json.dumps(build(root, 200)))
    fakes = Fakes()
    monkeypatch.setattr(driver, "get_git_commit", lambda: CLEAN_COMMIT)
    monkeypatch.setattr(driver, "resolve_local", lambda name: ("fake-model", "", "", False))
    monkeypatch.setattr(driver, "new_model", lambda *a, **k: (object(), 2.0))
    monkeypatch.setattr(driver, "get_adapter_state", lambda model: initial_state())
    monkeypatch.setattr(driver, "set_adapter_state", lambda model, state: None)
    monkeypatch.setattr(driver, "client_train", fakes.train)
    monkeypatch.setattr(driver, "response_encode", fake_encode)
    monkeypatch.setattr(driver, "_runtime_provenance", lambda *a, **k: {"test": True})
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)
    out = tmp_path / "out"
    sys.argv[:] = ["continual_driver.py", "--manifest", str(path), "--data_root", str(root),
                   "--arm", "fedavg-replay", "--seed", "7", "--rounds", "2",
                   "--memory_budget", "6", "--out", str(out)]
    driver.main()
    report = validate_continual.validate_run(out)
    result = json.loads(next(out.glob("continual_*.json")).read_text())
    assert report["rounds_validated"] == 4 and result["eval_pool"] == {}
    assert result["summary"]["G"] >= 0


def test_the_no_shift_control_re_splits_the_same_queries_over_the_same_corpus(tmp_path):
    root = synthetic_lotte(tmp_path)
    manifest = build(root, 200)
    control = lotte.no_shift_manifest(root, manifest, 1)
    for c, client in control["clients"].items():
        source = manifest["clients"][c]
        ids = lambda m: sorted(q for cell in m["experiences"].values()
                               for split in ("train", "guard", "test") for q in cell[split])
        assert ids(client) == ids(source) and client["corpus"] == source["corpus"]
        for cell in client["experiences"].values():
            assert (len(cell["train"]), len(cell["guard"]), len(cell["test"])) == (10, 4, 6)
            assert {q.split("-")[0] for q in cell["train"] + cell["test"]} == {"dev", "test"}
            assert all(len(cell["qrels"][q]) == 1 for q in cell["train"])
            assert all(len(cell["qrels"][q]) == 2 for q in cell["guard"] + cell["test"])
