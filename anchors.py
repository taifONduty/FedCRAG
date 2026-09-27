"""Rank-anchored replay (registration section 17). When a client finishes an experience, each
of its training queries keeps the top passages of the experience's acquisition reference and
their scores. When a query is later replayed, training keeps its distribution over those
passages close to the stored one, so the passages that decide its nDCG@10 hold their order.
Under the one-sided floor (section 17.3), only the relevant passage's share among the best
non-relevant ones is held, and only against a drop."""
import math

import numpy as np
import torch
from sentence_transformers.datasets import NoDuplicatesDataLoader
from torch.nn import functional as F

from federated_forgetting import (amp_enabled, get_adapter_state, make_examples,
                                  set_adapter_state)

SCALE = 20.0


def _top(scores, k):
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


def rank_anchors(q_emb, qids, qrels, c_emb, cids, k):
    """For each query: its top-k passages under the reference with their scaled scores; its
    margin, the best relevant passage's score minus the best non-relevant one's; and its floor
    set, the best relevant passage followed by the k - 1 best non-relevant ones."""
    index = {c: i for i, c in enumerate(cids)}
    sims = q_emb @ c_emb.T
    anchors = {}
    for row, q in enumerate(qids):
        scores = sims[row]
        relevant = [index[p] for p, rel in qrels[q].items() if rel > 0 and p in index]
        if not relevant:
            raise ValueError(f"training query {q} has no relevant passage in its corpus")
        best = relevant[int(np.argmax(scores[relevant]))]
        others = scores.copy()
        others[relevant] = -np.inf
        top, floor = _top(scores, k), [best, *_top(others, k - 1)]
        anchors[q] = {"pids": [cids[j] for j in top],
                      "scores": [float(SCALE * scores[j]) for j in top],
                      "margin": float(SCALE * (scores[best] - others.max())),
                      "floor_pids": [cids[j] for j in floor],
                      "floor_scores": [float(SCALE * scores[j]) for j in floor]}
    return anchors


class RankAnchorLoss(torch.nn.Module):
    """The in-batch contrastive loss on every row, plus lambda times the KL divergence from each
    anchored query's stored distribution over its reference top-k to the current one, for the
    next ``per_step`` anchored queries at every step. With ``floor``, the added term is instead
    how far the log share of the first passage, the relevant one, fell below its stored value,
    and zero if it did not fall."""

    def __init__(self, model, anchored, per_step, lam, seed, scale=SCALE, floor=False):
        super().__init__()
        self.model, self.anchored, self.per_step = model, anchored, per_step
        self.lam, self.scale, self.floor = lam, scale, floor
        self._order = np.random.default_rng(seed).permutation(len(anchored)).tolist()
        self._next = 0

    def _embed(self, texts):
        features = self.model.tokenize(texts)
        features = {key: value.to(self.model.device) for key, value in features.items()}
        return F.normalize(self.model(features)["sentence_embedding"], dim=-1)

    def _anchor_term(self):
        take = [self.anchored[self._order[(self._next + i) % len(self._order)]]
                for i in range(self.per_step)]
        self._next += self.per_step
        k = len(take[0][1])
        queries = self._embed([query for query, _, _ in take])
        passages = self._embed([p for _, texts, _ in take for p in texts]).view(len(take), k, -1)
        student = self.scale * torch.einsum("qd,qkd->qk", queries, passages)
        teacher = torch.tensor([scores for _, _, scores in take], device=student.device)
        if self.floor:
            drop = (F.log_softmax(teacher.float(), dim=-1)[:, 0]
                    - F.log_softmax(student.float(), dim=-1)[:, 0])
            return F.relu(drop).mean()
        return F.kl_div(F.log_softmax(student.float(), dim=-1), F.softmax(teacher.float(), dim=-1),
                        reduction="batchmean")

    def forward(self, sentence_features, labels):
        queries, positives = [F.normalize(self.model(f)["sentence_embedding"], dim=-1)
                              for f in sentence_features]
        scores = self.scale * queries @ positives.T
        loss = F.cross_entropy(scores, torch.arange(len(scores), device=scores.device))
        if self.lam > 0 and self.anchored:
            loss = loss + self.lam * self._anchor_term()
        return loss


def client_train_anchor(model, start_state, data, anchored, q_prefix, d_prefix, batch_size,
                        lr, lam, seed, floor=False):
    """One local epoch as in client_train, with each anchored query anchored once."""
    set_adapter_state(model, start_state)
    examples = make_examples(data, q_prefix, d_prefix)
    loader = NoDuplicatesDataLoader(examples, batch_size=batch_size)
    steps = len(loader)
    per_step = math.ceil(len(anchored) / steps) if anchored else 0
    loss = RankAnchorLoss(model, anchored, per_step, lam, seed, floor=floor)
    model.fit(train_objectives=[(loader, loss)],
              epochs=1, steps_per_epoch=steps, optimizer_params={"lr": lr},
              warmup_steps=max(1, int(0.1 * steps)), show_progress_bar=False,
              use_amp=amp_enabled())
    return get_adapter_state(model), len(examples), steps
