"""anchors.py: each retained query keeps its reference top-k, and training holds that order,
or, under the one-sided floor, only the relevant passage's share of it; hard-negative replay
trains that share with no stored target; contrastive consolidation keeps each item's own
reference vectors identifiable through a projector that fit() trains."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

import anchors


class TextStub(torch.nn.Module):
    """Embeds a text by table lookup and passes precomputed batch embeddings through."""
    device = torch.device("cpu")

    def __init__(self, table):
        super().__init__()
        self.table = table

    def tokenize(self, texts):
        return {"embedding": torch.stack([self.table[t] for t in texts])}

    def forward(self, feature):
        return {"sentence_embedding": feature["embedding"]}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_anchors_keep_the_reference_top_k_in_order_and_the_margin_to_the_best_negative():
    c_emb = np.stack([unit([1, 0, 0]), unit([0.9, 0.1, 0]), unit([0, 1, 0]), unit([0.5, 0.5, 0])])
    q_emb = np.stack([unit([1, 0.05, 0])])
    made = anchors.rank_anchors(q_emb, ["q"], {"q": {"p3": 1}}, c_emb,
                                ["p0", "p1", "p2", "p3"], 2, vectors=True)
    sims = q_emb @ c_emb.T
    assert made["q"]["pids"] == ["p0", "p1"]
    assert made["q"]["scores"] == pytest.approx([20 * sims[0, 0], 20 * sims[0, 1]])
    assert made["q"]["margin"] == pytest.approx(20 * (sims[0, 3] - sims[0, 0]))
    assert made["q"]["floor_pids"] == ["p3", "p0"]
    assert made["q"]["floor_scores"] == pytest.approx([20 * sims[0, 3], 20 * sims[0, 0]])
    assert made["q"]["q_vec"] == pytest.approx(q_emb[0])
    assert made["q"]["p_vec"] == pytest.approx(c_emb[3])
    with pytest.raises(ValueError, match="no relevant passage"):
        anchors.rank_anchors(q_emb, ["q"], {"q": {"p9": 1}}, c_emb, ["p0", "p1", "p2", "p3"], 2)


def test_the_anchor_term_vanishes_when_the_order_is_kept_and_grows_when_it_is_lost():
    torch.manual_seed(0)
    table = {t: F.normalize(torch.randn(8), dim=0) for t in ("q", "p1", "p2", "p3")}
    model = TextStub(table)
    features = [{"embedding": torch.randn(6, 8)}, {"embedding": torch.randn(6, 8)}]
    labels = torch.zeros(6)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    contrastive = MultipleNegativesRankingLoss(model)(features, labels)
    kept = [20 * float(table["q"] @ table[p]) for p in ("p1", "p2", "p3")]
    same = anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], kept)], 1, 2.0, 0)
    assert same(features, labels).item() == pytest.approx(contrastive.item(), abs=1e-5)
    lost = anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], kept[::-1])], 1, 2.0, 0)
    assert lost(features, labels).item() > contrastive.item() + 1e-4
    off = anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], kept[::-1])], 1, 0.0, 0)
    assert off(features, labels).item() == pytest.approx(contrastive.item(), abs=1e-6)


def test_the_floor_penalizes_only_a_drop_of_the_relevant_share_and_hard_replay_always_trains_it():
    torch.manual_seed(0)
    table = {t: F.normalize(torch.randn(8), dim=0) for t in ("q", "p1", "p2", "p3")}
    model = TextStub(table)
    features = [{"embedding": torch.randn(6, 8)}, {"embedding": torch.randn(6, 8)}]
    labels = torch.zeros(6)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    contrastive = MultipleNegativesRankingLoss(model)(features, labels).item()
    now = [20 * float(table["q"] @ table[p]) for p in ("p1", "p2", "p3")]
    stored = {"gained": [now[0] - 1.0, now[1], now[2]], "reordered": [now[0], now[2], now[1]],
              "lost": [now[0] + 1.0, now[1], now[2]]}
    loss = {name: anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], scores)], 1, 2.0, 0,
                                         mode="floor")(features, labels).item()
            for name, scores in stored.items()}
    assert loss["gained"] == pytest.approx(contrastive, abs=1e-5)
    assert loss["reordered"] == pytest.approx(contrastive, abs=1e-5)
    assert loss["lost"] > contrastive + 1e-4
    kl = anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], stored["reordered"])], 1, 2.0, 0)
    assert kl(features, labels).item() > contrastive + 1e-4
    share = torch.log_softmax(torch.tensor(now), dim=0)[0].item()
    for scores in stored.values():
        hard = anchors.RankAnchorLoss(model, [("q", ["p1", "p2", "p3"], scores)], 1, 2.0, 0,
                                      mode="hard")
        assert hard(features, labels).item() == pytest.approx(contrastive - 2.0 * share, abs=1e-5)
    with pytest.raises(ValueError, match="anchor mode"):
        anchors.RankAnchorLoss(model, [], 1, 2.0, 0, mode="soft")


def test_consolidation_picks_each_items_own_vectors_through_an_identity_start_projector():
    torch.manual_seed(0)
    table = {t: F.normalize(torch.randn(8), dim=0) for t in ("q1", "q2", "q3", "p1", "p2", "p3")}
    model = TextStub(table)
    own = [(q, [p], (table[q].tolist(), table[p].tolist()))
           for q, p in (("q1", "p1"), ("q2", "p2"), ("q3", "p3"))]
    swapped = [(q, texts, vectors)
               for (q, texts, _), (_, _, vectors) in zip(own, own[1:] + own[:1])]

    def term(items, **kw):
        return anchors.RankAnchorLoss(model, items, 3, 1.0, 0, mode="ckc", **kw)._anchor_term()

    assert term(own).item() < term(swapped).item()
    u, v = torch.nn.Parameter(torch.zeros(8, 2)), torch.nn.Parameter(torch.randn(8, 2))
    projected = term(own, projector=(u, v))
    assert projected.item() == pytest.approx(term(own).item(), abs=1e-6)
    projected.backward()
    assert u.grad.abs().sum() > 0


def test_the_projector_is_a_model_parameter_during_fit_and_is_removed_after(monkeypatch):
    class FitStub(torch.nn.Module):
        device = torch.device("cpu")

        def fit(self, train_objectives, **kwargs):
            self.seen = [name for name, _ in self.named_parameters()]

    model, seen = FitStub(), []
    monkeypatch.setattr(anchors, "set_adapter_state", lambda m, s: None)
    monkeypatch.setattr(anchors, "get_adapter_state", lambda m: {})
    monkeypatch.setattr(anchors, "make_examples", lambda data, q, d: [0, 1, 2, 3])
    monkeypatch.setattr(anchors, "NoDuplicatesDataLoader", lambda examples, batch_size: [examples])
    item = ("q", ["p"], ([1.0, 0.0], [0.0, 1.0]))
    anchors.client_train_anchor(model, {}, {}, [item], "", "", 4, 1e-4, 1.0, 0, mode="ckc",
                                projector_rank=1)
    assert model.seen == ["ckc_u", "ckc_v"] and not list(model.named_parameters())
