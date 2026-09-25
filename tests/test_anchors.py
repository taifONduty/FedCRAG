"""anchors.py: each retained query keeps its reference top-k, and training holds that order."""
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
    made = anchors.rank_anchors(q_emb, ["q"], {"q": {"p3": 1}}, c_emb, ["p0", "p1", "p2", "p3"], 2)
    sims = q_emb @ c_emb.T
    assert made["q"]["pids"] == ["p0", "p1"]
    assert made["q"]["scores"] == pytest.approx([20 * sims[0, 0], 20 * sims[0, 1]])
    assert made["q"]["margin"] == pytest.approx(20 * (sims[0, 3] - sims[0, 0]))


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
