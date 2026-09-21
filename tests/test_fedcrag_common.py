"""fedcrag_common: the no-train-split fallback is an ordered protocol, only a
genuinely missing split may trigger it, and an embedding cache is bound to the
texts it was computed from."""
import os

import numpy as np
import pytest

import fedcrag_common

QIDS = [f"q{i:02d}" for i in range(60)]


class FakeLoader:
    """BEIR's loader, reduced to what the fallback logic sees. Test queries are
    supplied in reverse order so file order and sorted order differ."""
    corrupt_train = False

    def __init__(self, path):
        self.path = path

    def load(self, split="test"):
        if split == "train":
            if not os.path.exists(os.path.join(self.path, "qrels", "train.tsv")):
                raise ValueError("File train.tsv not present!")
            if self.corrupt_train:
                raise ValueError("corrupt train qrels")
            return {}, {"t0": "train query"}, {"t0": {"d0": 1}}
        queries = {q: f"text {q}" for q in reversed(QIDS)}
        qrels = {q: {"d0": 1} for q in reversed(QIDS)}
        return {"d0": {"title": "", "text": "doc"}}, queries, qrels


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(fedcrag_common.beir_util, "download_and_unzip",
                        lambda url, root: str(tmp_path))
    monkeypatch.setattr(fedcrag_common, "GenericDataLoader", FakeLoader)
    monkeypatch.setattr(FakeLoader, "corrupt_train", False)
    return tmp_path


def test_missing_train_split_halves_sorted_qids_in_sorted_order(dataset):
    data = fedcrag_common.load_slice_with_train("x", "unused")
    assert data["split_fallback"] is True
    assert list(data["train_qrels"]) == QIDS[:30]
    assert list(data["train_q"]) == QIDS[:30]
    assert list(data["eval_qrels"]) == QIDS[30:]
    assert list(data["eval_q"]) == QIDS[30:]


def test_unreadable_train_split_is_an_error_not_a_fallback(dataset, monkeypatch):
    (dataset / "qrels").mkdir()
    (dataset / "qrels" / "train.tsv").write_text("garbage\n")
    monkeypatch.setattr(FakeLoader, "corrupt_train", True)
    with pytest.raises(ValueError, match="corrupt"):
        fedcrag_common.load_slice_with_train("x", "unused")


class CountingModel:
    def __init__(self):
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_cache_is_not_reused_for_different_texts_of_the_same_shape(tmp_path):
    model = CountingModel()
    cache = str(tmp_path / "corpus.npy")
    fedcrag_common.encode_texts(model, ["a", "bb"], "p:", 8, cache)
    second = fedcrag_common.encode_texts(model, ["a", "ccc"], "p:", 8, cache)
    assert model.calls == 2
    assert second[1, 0] == len("p:ccc")
