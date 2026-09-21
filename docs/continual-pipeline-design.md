# Continual pipeline: design

Written 2026-09-21 for the WWW 2027 line of work. Status: proposed, awaiting approval.
Nothing here is registered; the pilot is registered in `registration/E3_PREREGISTRATION.md`
section 14 once the manifests exist and before any training run.

## 1. Question and scope

One shared dense retriever is trained by K organisations whose information needs change
over time. The question is whether learning a client's new information need degrades what
the shared model already retrieved well for that client or another one, measured per client
and per earlier experience on queries never used for training or selection, and whether
ordinary shared training with replay already controls that regression.

This design covers the problem-validation pilot only: the data construction, the driver,
the measurements, four baseline arms and a registered gate. Method arms come after the gate
and get their own design and registration.

Out of scope for the pilot: corpus growth, changed relevance, stale indexes, answer
generation, the measured-response arm, and any claim about privacy.

## 2. Data: an MS-Shift stream

Source files. MS MARCO passage collection (8.84 M passages), the training queries and
qrels, and MS-Shift's `TRAIN/queries_clustering.tsv`, which labels 398,793 training queries
with a topic cluster (0 to 4, plus 5 for "other") and a question-intent group. MS-Shift also
releases, per topic cluster, 5,868 to 6,595 dev-set queries with qrels (`EVAL/`), which
serve as an untouched per-client check. Licence CC BY-NC-SA 4.0; research use.

Clients. Five, one per MS-Shift topic cluster 0 to 4 (30,782 to 35,766 training queries
each). The "other" cluster is not used.

Experiences. Four per client. Within a topic cluster the queries are partitioned into four
sub-topics by k-means (k = 4) on TF-IDF unigram and bigram features reduced by truncated
SVD to 100 dimensions, seed recorded. The construction is independent of any retriever.
If a sub-cluster holds fewer than 2,700 queries the clustering is rerun with the seed
incremented, at most five times, and otherwise the build fails. Alternative considered:
MS-Shift's released intent labels (what, how, who/when/where, other) as experiences. They
need no clustering but describe query form rather than information need, and the smallest
topic-by-intent cell has 1,177 queries; kept as a secondary axis.

Splits per (client, experience), sampled with the manifest seed, disjoint: 2,000 training
queries, 200 guard queries, 500 test queries. Guard queries are the ones a future method may
consult repeatedly; test queries are never used for training, selection or a gate on their
own values before the registered gate is evaluated. The pilot arms use neither guard nor
test queries for anything but measurement.

Order. Every client receives its four experiences in the same order in the pilot,
sub-cluster index 0 to 3 as returned by the seeded k-means. Per-client permuted orders are
a campaign axis, not a pilot axis.

Corpus per client. Fixed for the whole run: every passage judged relevant for any of the
client's selected queries (all four experiences, all three splits, plus that topic's
MS-Shift dev queries), filled up with passages sampled uniformly from the collection to
60,000 passages, seed recorded. This follows the size of CREAM's per-topic MS MARCO
collections. It is a pilot protocol, not a claim that the reduced pool preserves the
original task; the paper says so.

Manifest. `experiences.py build` writes one JSON per stream: seeds, cluster assignment
per query id, split membership, corpus passage ids per client, and sha256 digests of each
list. The registration records the manifest digest. The manifest is built on the Mac; the
GPU machine only trains and evaluates.

Sizes for cost. Training pairs per (client, experience) about 2,100 (MS MARCO has one
judged passage for most queries); 60,000 passages per client to encode per evaluation.

## 3. Training protocol

Backbone `facebook/contriever` (unsupervised; it has never seen MS MARCO labels, so the
stream is new capability), LoRA rank 16, both factors trainable, learning rate 2e-5, batch
32, in-batch negatives, one local epoch per round, mixed precision, the E1 recipe.

An experience is R = 4 communication rounds. In each round every client trains from the
broadcast state on its current experience's training queries plus its replay memory, and
the server averages the adapters with equal weights. Four experiences give 16 rounds.

Replay memory. Per client, at most 256 query families (a query with its judged passages)
drawn from earlier experiences, an equal share per earlier experience, seeded; refilled at
each experience boundary. Arms without replay have an empty memory.

Acquisition reference. For client i and experience s, the shared model at the end of the
last round of experience s. It is stored (adapter state and hash), together with the
per-query scores it obtains on the guard and test queries of (i, s).

## 4. Measurements

After the last round of each experience s the shared model is evaluated on the guard and
test queries of every (i, s') with s' <= s, against client i's fixed corpus. The corpus is
encoded once per client per evaluation; the query sets are cheap.

Recorded per (i, s', s): nDCG@10 and recall@100, and the per-query nDCG@10 of every query
(so paired tests and any later re-analysis need no re-run). Also recorded: the frozen
backbone's per-query scores (round 0) and, at the end, each client's MS-Shift dev queries.

Regression of (i, s') at time s: the positive part, per query, of the acquisition
reference's nDCG@10 minus the current model's nDCG@10, averaged over the split. Gains on
some queries cannot cancel losses on others. Backward transfer and peak forgetting are
reported beside it with their definitions; the positive-part risk is primary.

Acquisition of (i, s) at time s: the current model's nDCG@10 minus the frozen backbone's
on the test queries of (i, s).

## 5. Pilot arms, seeds and gate

Arms: (A) frozen backbone; (B) local-only continual, each client alone with the same
replay budget and the same number of optimiser steps; (C) uniform FedAvg without replay;
(D) uniform FedAvg with replay. Seeds 123, 2024 and 42, so three seeds. Nine trained runs
plus one frozen evaluation.

Gate, to be written into section 14 before the first run and evaluated on test queries
only after every run has finished and validated:

- G2 (adaptation is useful): under D, acquisition averaged over clients and experiences is
  at least 0.020 nDCG@10 at every seed.
- G1 (a problem remains): under D, the regression averaged over clients and earlier
  experiences at the end of the stream is at least 0.010 nDCG@10 at every seed. The
  threshold is about four times the largest seed half-range seen in the thesis's paired
  E1 runs (0.0027).
- G3 (collaboration has value): under D, acquisition is at least that of B, averaged.

Outcomes. G2 fails: the recipe or the stream is at fault; fix and rerun before anything
else. G1 fails with G2 passing: replay controls the regression here; the method work is
not motivated by this stream, and the paper's temporal evidence moves to LoTTE or to a
report that replay suffices. G1 and G2 pass: the method arms are designed and registered.
G3 is reported either way and shapes the framing, not the go decision.

## 6. Cost

Per trained run on an L4: about 16 rounds times 5 clients times 70 steps of LoRA training,
under an hour, plus four evaluations of 300,000 passage encodes and 5 clients' query sets,
about 20 minutes. Nine runs fit in one day of one L4. Manifest building runs on the Mac
(MS MARCO collection 1.0 GB compressed).

## 7. Code

Three new top-level modules and their tests; `federated_forgetting.py` is not changed.

- `experiences.py`: loads MS MARCO and MS-Shift, builds and verifies a manifest, and
  materialises per-(client, experience) data in the shape `client_train` already consumes
  (`corpus`, `train_q`, `train_qrels`, plus `guard_*` and `test_*`).
- `continual_driver.py`: the experience loop over rounds, replay memory, reference
  snapshots, the evaluation matrix, per-query outputs, result JSON with provenance in the
  same form as the static driver (commit, source hashes, data digests, arguments).
  Reuses `client_train`, `fedavg`, `get_adapter_state`/`set_adapter_state` and the
  metric code.
- `regression.py`: per-query nDCG@10 from rankings, positive-part regression against a
  reference, acquisition, and the matrix summaries.

Dependency added: scikit-learn, for TF-IDF, truncated SVD and k-means in the manifest
build only.

Records. Per round: the same state payload as the static driver (clients, broadcast,
global, hashes), so the aggregate can be recomputed from the persisted states. Extending
`validate_e0.py` to the continual record is the first task after the pilot runs are
launched, before any result is quoted.

Tests, one behaviour each: manifest determinism under the same seed and difference under
another; the sub-cluster size rule; disjointness of the three splits; the corpus rule;
positive-part regression on a hand-built example; the driver loop on the synthetic
encoder of `tests/driver_harness.py` (reference stored at each experience end, matrix
shape, replay budget respected, no test query in any training batch).

## 8. Beyond the pilot

LoTTE (3.6 GB, corpora of 0.39 to 2.04 M passages per domain, about 5,200 to 5,600
queries per domain) replicates the pilot with a different query style and several
relevant passages per query; its corpora are capped by the same rule. LongEval-Retrieval
2023 English (three snapshots, June to September 2022, click-model labels) is a
robustness evaluation of trained models, not a training stream; its download requires a
LINDAT/CLARIN account, which the author must create. FreshStack is optional. Each
dataset gets a readiness table built from the actual files before it enters a run.
