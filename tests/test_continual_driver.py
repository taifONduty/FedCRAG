"""continual_driver.py and validate_continual.py on a two-client, two-experience stream."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continual_driver as driver  # noqa: E402
import experiences  # noqa: E402
import validate_continual  # noqa: E402
from aggregation_schemes import state_dict_sha256  # noqa: E402
from test_experiences import fake_bm25, synthetic_msmarco  # noqa: E402

CLEAN_COMMIT = "a" * 40


def initial_state():
    return {"lora_A": torch.ones(2, 2, dtype=torch.float32),
            "lora_B": torch.zeros(2, 2, dtype=torch.float32)}


def _vec(text, salt, dim=8):
    digest = hashlib.sha256(f"{salt}|{text}".encode()).digest()
    return np.frombuffer(digest[:dim], dtype=np.uint8).astype(np.float64) / 255.0 - 0.5


def fake_encode(model, state, texts, batch_size):
    shift = float(state["lora_B"].mean())
    rows = np.array([_vec(t, 0) + shift * _vec(t, 1) for t in texts])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class Fakes:
    """Records every training call's query ids and moves B by a client-specific amount."""

    def __init__(self):
        self.calls = []

    def train(self, model, start, data, q_prefix, d_prefix, epochs, batch_size, lr, name,
              max_steps=0):
        self.calls.append((name, sorted(data["train_q"])))
        new = {k: v.clone() for k, v in start.items()}
        new["lora_B"] = new["lora_B"] + 0.1 * (int(name) + 1)
        return new, len(data["train_q"]), 3

    def train_distill(self, model, teacher, start, teacher_state, data, replay_ids,
                      q_prefix, d_prefix, batch_size, lr, name, lam):
        return self.train(model, start, data, q_prefix, d_prefix, 1, batch_size, lr, name)


@pytest.fixture
def stream(tmp_path, monkeypatch):
    root = synthetic_msmarco(tmp_path)
    manifest = experiences.build_manifest(
        root, clients=(0, 1), experiences_per_client=2, schedule={0: (0, 1), 1: (1, 0)},
        counts={"train": 20, "guard": 4, "test": 3}, corpus_size=120, hard_k=2, seed=5,
        retriever=fake_bm25)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    fakes = Fakes()
    monkeypatch.setattr(driver, "get_git_commit", lambda: CLEAN_COMMIT)
    monkeypatch.setattr(driver, "resolve_local", lambda name: ("fake-model", "", "", False))
    monkeypatch.setattr(driver, "new_model", lambda *a, **k: (object(), 2.0))
    monkeypatch.setattr(driver, "get_adapter_state", lambda model: initial_state())
    monkeypatch.setattr(driver, "set_adapter_state", lambda model, state: None)
    monkeypatch.setattr(driver, "client_train", fakes.train)
    monkeypatch.setattr(driver, "client_train_distill", fakes.train_distill)
    monkeypatch.setattr(driver, "response_encode", fake_encode)
    monkeypatch.setattr(driver, "_runtime_provenance", lambda *a, **k: {"test": True})
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)
    return root, manifest, manifest_path, fakes


def run(stream, arm, out, extra=()):
    root, manifest, manifest_path, fakes = stream
    argv = ["continual_driver.py", "--manifest", str(manifest_path), "--data_root", str(root),
            "--arm", arm, "--seed", "7", "--rounds", "2", "--memory_budget", "6",
            "--out", str(out), *extra]
    sys.argv[:] = argv
    driver.main()
    paths = list(Path(out).glob("continual_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text()), paths[0], fakes


def test_fedavg_replay_records_matrix_references_and_memory(stream, tmp_path):
    result, path, fakes = run(stream, "fedavg-replay", tmp_path / "out")
    manifest = stream[1]
    assert [r["position"] for r in result["rounds"]] == [0, 0, 1, 1]
    for client in ("0", "1"):
        first, second = manifest["clients"][client]["order"]
        assert set(result["matrix"]["1"][client]) == {str(first), str(second)}
        cell = result["matrix"]["1"][client][str(first)]["test"]
        assert set(cell["per_query"]["ndcg@10"]) == set(manifest["clients"][client]["experiences"][str(first)]["test"])
        assert result["references"][client][str(first)]["position"] == 0
        earlier_train = set(manifest["clients"][client]["experiences"][str(first)]["train"])
        memory = result["rounds"][2]["memory"][client]
        assert memory["used"] <= 6 and set(memory["replay"]) <= earlier_train
        guard_or_test = {q for cell in manifest["clients"][client]["experiences"].values()
                         for q in cell["guard"] + cell["test"]}
        for name, ids in fakes.calls:
            if name == client:
                assert not (set(ids) & guard_or_test)
    payloads = [torch.load(path.parent / r["state_file"], weights_only=True) for r in result["rounds"]]
    assert state_dict_sha256(payloads[2]["broadcast"]) == state_dict_sha256(payloads[1]["global"])
    assert result["references"]["0"][str(manifest["clients"]["0"]["order"][0])]["sha256"] == state_dict_sha256(payloads[1]["global"])
    summary = result["summary"]
    assert "acquisition" in summary and "regression" in summary
    # regression is measured against each arm's own reference, so the absolute score on the
    # same old queries has to be reported beside it
    cell = next(iter(summary["regression"]["cells"].values()))
    assert {"regression", "bwt", "peak_forgetting", "final_ndcg", "reference_ndcg"} <= set(cell)
    assert summary["final_ndcg_on_earlier_experiences"] == pytest.approx(
        sum(c["final_ndcg"] for c in summary["regression"]["cells"].values())
        / len(summary["regression"]["cells"]))


def test_local_arm_keeps_one_model_per_client(stream, tmp_path):
    result, path, _ = run(stream, "local", tmp_path / "out")
    payload = torch.load(path.parent / result["rounds"][0]["state_file"], weights_only=True)
    assert "global" not in payload
    assert not torch.equal(payload["clients"]["0"]["lora_B"], payload["clients"]["1"]["lora_B"])
    assert result["references"]["0"] and result["references"]["1"]


def test_frozen_arm_trains_nothing(stream, tmp_path):
    result, _, fakes = run(stream, "frozen", tmp_path / "out")
    assert fakes.calls == [] and result["rounds"] == []
    assert result["matrix"]["1"]["0"] and result["frozen"]["0"]


@pytest.mark.parametrize("arm", ["fedavg-replay", "local", "fedavg-replay-distill", "frozen"])
def test_validator_accepts_a_genuine_run(stream, tmp_path, arm):
    run(stream, arm, tmp_path / "out")
    report = validate_continual.validate_run(tmp_path / "out")
    assert report["arm"] == arm and report["rounds_validated"] == (0 if arm == "frozen" else 4)


def test_validator_refuses_a_broken_chain(stream, tmp_path):
    result, path, _ = run(stream, "fedavg-replay", tmp_path / "out")
    state_path = path.parent / result["rounds"][2]["state_file"]
    payload = torch.load(state_path, weights_only=True)
    payload["broadcast"]["lora_B"] += 1.0
    payload["broadcast_state_sha256"] = state_dict_sha256(payload["broadcast"])
    torch.save(payload, state_path)
    with pytest.raises(validate_continual.ContinualValidationError, match="connect"):
        validate_continual.validate_run(tmp_path / "out")


def test_validator_refuses_replay_that_holds_a_test_query(stream, tmp_path):
    result, path, _ = run(stream, "fedavg-replay", tmp_path / "out")
    manifest = stream[1]
    test_id = manifest["clients"]["0"]["experiences"]["0"]["test"][0]
    result["rounds"][2]["memory"]["0"]["replay"][0] = test_id
    path.write_text(json.dumps(result))
    with pytest.raises(validate_continual.ContinualValidationError, match="test"):
        validate_continual.validate_run(tmp_path / "out")


# ------------------------------------------- the distillation loss at lambda = 0


class StubEncoder(torch.nn.Module):
    """Returns the embeddings it is given, so a loss can be compared exactly."""

    def forward(self, feature):
        return {"sentence_embedding": feature["embedding"]}


def test_zero_lambda_reproduces_the_contrastive_loss_exactly():
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    torch.manual_seed(0)
    features = [{"embedding": torch.randn(6, 8)}, {"embedding": torch.randn(6, 8)}]
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    student, teacher = StubEncoder(), StubEncoder()
    reference = MultipleNegativesRankingLoss(student)(features, labels)
    at_zero = driver.ReplayDistillLoss(student, teacher, lam=0.0)(features, labels)
    assert torch.allclose(at_zero, reference, atol=1e-6)
    positive = driver.ReplayDistillLoss(student, teacher, lam=2.0)(features, labels)
    assert positive.item() == pytest.approx(reference.item(), abs=1e-6)


def test_the_distillation_term_penalises_moving_away_from_the_teacher():
    torch.manual_seed(1)
    features = [{"embedding": torch.randn(4, 8)}, {"embedding": torch.randn(4, 8)}]
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    student, teacher = StubEncoder(), StubEncoder()
    same = driver.ReplayDistillLoss(student, teacher, lam=2.0)(features, labels)
    reference = driver.ReplayDistillLoss(student, teacher, lam=0.0)(features, labels)
    assert same.item() == pytest.approx(reference.item(), abs=1e-6)

    class Shifted(StubEncoder):
        def forward(self, feature):
            return {"sentence_embedding": feature["embedding"] + 0.5}
    moved = driver.ReplayDistillLoss(student, Shifted(), lam=2.0)(features, labels)
    assert moved.item() > reference.item()
