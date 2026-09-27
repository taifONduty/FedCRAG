"""Rank-anchored replay (registration section 17). When a client finishes an experience, each
of its training queries keeps the top passages of the experience's acquisition reference and
their scores. When a query is later replayed, training keeps its distribution over those
passages close to the stored one, so the passages that decide its nDCG@10 hold their order.
Under the one-sided floor (section 17.3), only the relevant passage's share among the best
non-relevant ones is held, and only against a drop; under hard-negative replay (17.4), the
relevant passage is trained against those passages without any stored target. Under
contrastive consolidation (17.6), each retained query and its relevant passage keep their
reference embeddings, and the current embeddings, through a small learned map, must pick
their own out of the client's bank, as in C-CLIP's knowledge consolidation."""
import math

import numpy as np
import torch
from sentence_transformers.datasets import NoDuplicatesDataLoader
from torch.nn import functional as F

from federated_forgetting import (amp_enabled, get_adapter_state, make_examples,
                                  set_adapter_state)

SCALE = 20.0
MODES = ("kl", "floor", "hard", "ckc")
PROJECTOR_SCALE = 2.0


def _top(scores, k):
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top], kind="stable")]


def rank_anchors(q_emb, qids, qrels, c_emb, cids, k, vectors=False):
    """For each query: its top-k passages under the reference with their scaled scores; its
    margin, the best relevant passage's score minus the best non-relevant one's; its floor
    set, the best relevant passage followed by the k - 1 best non-relevant ones; and, with
    ``vectors``, the reference embeddings of the query and of that relevant passage."""
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
        if vectors:
            anchors[q]["q_vec"] = [float(x) for x in q_emb[row]]
            anchors[q]["p_vec"] = [float(x) for x in c_emb[best]]
    return anchors


class RankAnchorLoss(torch.nn.Module):
    """The in-batch contrastive loss on every row, plus lambda times the KL divergence from each
    anchored query's stored distribution over its reference top-k to the current one, for the
    next ``per_step`` anchored queries at every step. In mode "floor" the added term is instead
    how far the log share of the first passage, the relevant one, fell below its stored value,
    and zero if it did not fall; in mode "hard" it is the negative log share itself. In mode
    "ckc" each anchored item is (query, [relevant passage], (query vector, passage vector)),
    and the term is the cross-entropy of picking the item's own stored vectors out of all
    anchored items' ones, after the optional low-rank map ``projector`` = (U, V)."""

    def __init__(self, model, anchored, per_step, lam, seed, scale=SCALE, mode="kl",
                 projector=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"anchor mode {mode!r} is not one of {MODES}")
        self.model, self.anchored, self.per_step = model, anchored, per_step
        self.lam, self.scale, self.mode, self.projector = lam, scale, mode, projector
        if mode == "ckc" and anchored:
            self.banks = [torch.tensor([v[i] for _, _, v in anchored], device=model.device)
                          for i in (0, 1)]
        self._order = np.random.default_rng(seed).permutation(len(anchored)).tolist()
        self._next = 0

    def _embed(self, texts):
        features = self.model.tokenize(texts)
        features = {key: value.to(self.model.device) for key, value in features.items()}
        return F.normalize(self.model(features)["sentence_embedding"], dim=-1)

    def _consolidate(self, new, bank, target):
        mapped = new.float()
        if self.projector is not None:
            u, v = self.projector
            mapped = mapped + PROJECTOR_SCALE * (mapped @ v) @ u.T
        logits = self.scale * F.normalize(mapped, dim=-1) @ bank.T
        return F.cross_entropy(logits, target)

    def _anchor_term(self):
        idx = [self._order[(self._next + i) % len(self._order)] for i in range(self.per_step)]
        self._next += self.per_step
        take = [self.anchored[i] for i in idx]
        if self.mode == "ckc":
            target = torch.tensor(idx, device=self.banks[0].device)
            new = [self._embed([query for query, _, _ in take]),
                   self._embed([texts[0] for _, texts, _ in take])]
            return sum(self._consolidate(n, b, target) for n, b in zip(new, self.banks)) / 2
        k = len(take[0][1])
        queries = self._embed([query for query, _, _ in take])
        passages = self._embed([p for _, texts, _ in take for p in texts]).view(len(take), k, -1)
        student = self.scale * torch.einsum("qd,qkd->qk", queries, passages)
        teacher = torch.tensor([scores for _, _, scores in take], device=student.device)
        if self.mode == "kl":
            return F.kl_div(F.log_softmax(student.float(), dim=-1),
                            F.softmax(teacher.float(), dim=-1), reduction="batchmean")
        share = F.log_softmax(student.float(), dim=-1)[:, 0]
        if self.mode == "hard":
            return -share.mean()
        return F.relu(F.log_softmax(teacher.float(), dim=-1)[:, 0] - share).mean()

    def forward(self, sentence_features, labels):
        queries, positives = [F.normalize(self.model(f)["sentence_embedding"], dim=-1)
                              for f in sentence_features]
        scores = self.scale * queries @ positives.T
        loss = F.cross_entropy(scores, torch.arange(len(scores), device=scores.device))
        if self.lam > 0 and self.anchored:
            loss = loss + self.lam * self._anchor_term()
        return loss


def attach_projector(model, dim, rank, seed):
    """A rank-``rank`` residual map on embeddings, registered on the model so that fit()
    trains it with the adapters: U starts at zero, so the map starts as the identity."""
    generator = torch.Generator().manual_seed(int(np.random.default_rng(seed).integers(2**31)))
    model.ckc_u = torch.nn.Parameter(torch.zeros(dim, rank, device=model.device))
    model.ckc_v = torch.nn.Parameter(
        (torch.randn(dim, rank, generator=generator) / math.sqrt(dim)).to(model.device))
    return model.ckc_u, model.ckc_v


def client_train_anchor(model, start_state, data, anchored, q_prefix, d_prefix, batch_size,
                        lr, lam, seed, mode="kl", projector_rank=0):
    """One local epoch as in client_train, with each anchored query anchored once; under
    "ckc" with ``projector_rank``, a fresh projector for this epoch."""
    set_adapter_state(model, start_state)
    examples = make_examples(data, q_prefix, d_prefix)
    loader = NoDuplicatesDataLoader(examples, batch_size=batch_size)
    steps = len(loader)
    per_step = math.ceil(len(anchored) / steps) if anchored else 0
    projector = (attach_projector(model, len(anchored[0][2][0]), projector_rank, seed)
                 if mode == "ckc" and projector_rank and anchored else None)
    loss = RankAnchorLoss(model, anchored, per_step, lam, seed, mode=mode, projector=projector)
    try:
        model.fit(train_objectives=[(loader, loss)],
                  epochs=1, steps_per_epoch=steps, optimizer_params={"lr": lr},
                  warmup_steps=max(1, int(0.1 * steps)), show_progress_bar=False,
                  use_amp=amp_enabled())
    finally:
        if projector is not None:
            del model.ckc_u, model.ckc_v
    return get_adapter_state(model), len(examples), steps
