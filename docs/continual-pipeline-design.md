# Continual pipeline: design

Written 2026-09-21 for the WWW 2027 line of work; amended the same day after review.
Status: approved for implementation with the amendments below. Nothing here is
registered; the pilot is registered in `registration/E3_PREREGISTRATION.md` section 14
once the manifests exist and before any deciding run.

## 1. Question and scope

One shared dense retriever is trained by K organisations whose information needs change
over time. The question is whether learning a client's new information need degrades what
the shared model already retrieved well for that client or another one, measured per client
and per earlier experience on queries never used for training or selection, and whether
ordinary shared training with replay, or replay with distillation, already controls it.

This design covers the problem-validation pilot: data construction, driver, memory,
measurements, validator, five baseline arms, a calibration stream and a registered gate.
Method arms come after the gate and get their own design and registration.

Out of scope for the pilot: corpus growth, changed relevance, stale indexes, answer
generation, the measured-response arm, and any claim about privacy.

## 2. Data: a constructed semantic-shift stream from MS-Shift

Source files. MS MARCO passage collection (8.84 M passages), training queries and qrels;
MS-Shift `TRAIN/queries_clustering.tsv` (398,793 training queries with a topic cluster 0 to
4, 5 for "other", and an intent group) and `EVAL/` (per topic cluster, 5,868 to 6,595
dev-set queries with qrels). Licence CC BY-NC-SA 4.0; research use.

Clients. Five, one per MS-Shift topic cluster 0 to 4. The "other" cluster is excluded from
the primary experiment and reserved for the calibration stream (section 6).

Experiences. Four per client, a constructed semantic-shift stream, not chronology. Within
a topic cluster the TRAIN queries are partitioned by k-means (k = 4) on TF-IDF unigram and
bigram features reduced by truncated SVD to 50 dimensions and unit-normalised (cosine
geometry). Among the standard settings tried on the feasibility table only (SVD 100 or 50
or 200, with and without normalisation), this one gives the largest smallest sub-cluster;
plain Euclidean k-means left a catch-all cluster of 20,000 to 28,000 queries per topic. The vocabulary, IDF weights,
SVD and centroids are fitted on eligible TRAIN query text only; EVAL queries are assigned
to the frozen centroids afterwards. Recorded in the manifest: the TF-IDF parameters,
vocabulary digest, SVD and k-means seeds, restarts, centroid digests and the assignment of
every query. If a sub-cluster is too small for the chosen counts the clustering is rerun
with the seed incremented, at most five times, otherwise the build fails.

Order schedules. Two predeclared schedules, A and B, each giving every client its own
permutation of its four experiences, fixed in the manifest before any training run and
never chosen after looking at forgetting. Schedule A: client 0 (0,1,2,3), client 1
(2,0,3,1), client 2 (1,3,0,2), client 3 (3,2,1,0), client 4 (0,3,1,2). Schedule B: client 0
(3,1,0,2), client 1 (1,3,2,0), client 2 (2,0,1,3), client 3 (0,2,3,1), client 4 (1,0,2,3).

Splits per (client, experience). Training and guard queries come from TRAIN; test queries
come from the official EVAL queries assigned to that sub-cluster. MS-Shift draws those
evaluation queries from the MS MARCO training pool, so their ids are removed from the
training side before the clustering and can only ever be test queries. The counts are not fixed
here: the builder first reports a feasibility table, client by experience, of eligible TRAIN
queries, eligible EVAL queries and relevant-passage counts, and the largest uniform counts
that every cell supports are then chosen and recorded. Test queries are never manufactured
from training queries. Guard queries are the ones a future method may consult online; the
pilot arms use guard and test queries for measurement only.

Retained unit. One query id together with all its relevance pairs. Pairs of one query
never cross a split boundary.

Corpus per client. Fixed for the whole run and identical across experiences: every passage
judged relevant for any of the client's selected queries, plus deterministic hard
distractors (the top-5 BM25 passages for the client's training, guard and test queries over
the full collection, a frozen lexical process fixed before any training and never a method
under test), plus passages sampled uniformly from the collection, to 60,000 passages in all. The full
passage-id list per client is recorded with its digest. This pool size follows CREAM's
per-topic MS MARCO collections; it is a pilot protocol, and decisive comparisons are to be
repeated on a much larger pool when compute allows.

Manifest. `experiences.py` writes one JSON per stream and schedule: seeds, feature and
clustering records, split membership, corpus ids, chosen counts and the digest of every
list. Section 14 records the manifest digests. The feasibility table and the clustering run
on the laptop; the manifest build itself runs on the GPU machine's CPU, because BM25 over
the full collection needs more than the laptop's 8 GiB of memory.

## 3. Training protocol

Backbone `facebook/contriever` (unsupervised; it has never seen MS MARCO labels), LoRA
rank 16, both factors trainable, in-batch negatives, one local epoch per round, mixed
precision. Learning rate and rounds per experience are calibrated on the calibration
stream (section 6) and frozen before the deciding runs; the E1 values (2e-5, four rounds)
are the starting grid, not a commitment.

An experience is R communication rounds. In each round every client trains from the
broadcast state on its current experience's training queries plus its memory, and the
server averages the adapters with equal weights.

Memory (`memory.py`). Per client one budget of at most 256 retained query ids in total,
covering replay and any guard queries a method consults online, so no arm holds hidden
retained information beyond the budget. Replay is deterministic seeded reservoir sampling
over the training queries of earlier experiences, an equal share per earlier experience,
refilled at each experience boundary. Arms without replay hold an empty memory. The module
owns capacity accounting, insertion and eviction, serialisation and restore, and reports
the budget used in every round record.

Acquisition reference. For client i and experience u, the shared model at the end of the
last round of experience u, stored as an adapter state with its hash, together with its
per-query scores on the guard and test queries of (i, u).

Distillation (arm E only). The teacher is the acquisition reference of the previous
experience, fixed for the whole experience. For replay rows in a batch the loss adds
lambda times the mean squared difference between the student's and the teacher's scaled
cosine scores over the batch passages; current-experience rows use the contrastive loss
alone. lambda is 1.0 unless the calibration stream, under the rule in section 6, picks
another value from {0.5, 1.0, 2.0}.

## 4. Measurements

After the last round of each experience v the shared model is evaluated on the guard and
test queries of every (i, u) with u <= v against client i's fixed corpus. The corpus is
encoded once per client per evaluation. Recorded per (i, u, v): nDCG@10, recall@100 and the
per-query nDCG@10 of every query. Also recorded: the frozen backbone's per-query scores
(round 0) and, at the end, each client's full official EVAL set for its topic.

Regression r(i, u, v), v > u: the positive part, per query, of the acquisition reference's
nDCG@10 minus the current model's, averaged over the split; gains on some queries cannot
cancel losses on others. This is the primary retention measure. Reported beside it, with
their definitions: signed backward transfer, peak forgetting, absolute final nDCG@10, the
full client-by-experience matrix, the worst cell, and the fraction of historical cells with
r >= 0.010.

Acquisition a(i, u): the current model's nDCG@10 minus the frozen backbone's on the test
queries of (i, u) at v = u.

## 5. Pilot arms, seeds and gate

Arms: (A) frozen backbone; (B) local-only continual, each client alone, five independent
models per seed, same memory budget and the same number of optimiser steps; (C) uniform
FedAvg without replay; (D) uniform FedAvg with replay; (E) uniform FedAvg with replay and
reference distillation. Seeds 123, 2024 and 3407, fixed now. Both schedules. Four trained
arms times three seeds times two schedules is 24 runs, plus the frozen evaluation per
schedule.

Gate scalars, on test queries, computed only after every run has validated:

- A = mean over (i, u) of a(i, u).
- G = mean over historical cells (i, u, v), v > u, of r(i, u, v) under arm D.

Thresholds 0.020 for A and 0.010 for G are practical effect sizes, not confidence bounds.
Consistency rule: a criterion passes if the mean of its scalar over the six runs (three
seeds by two schedules) clears the threshold and the scalar clears half the threshold in at
least five of the six runs. G2 (adaptation is useful) is A under arm D; G1 (a problem
remains) is G under arm D; G3 (collaboration has value, reported, not a go criterion) is A
under D at least A under B. Arm E is reported beside D: if E's G falls below half of D's G
while its A stays within 0.005 of D's, distillation already controls the regression and
the framing says so.

Outcomes. G2 fails: the recipe or the stream is at fault; fix and rerun before anything
else. G1 fails with G2 passing: replay controls the regression here, the method work is
not motivated by this stream, and the temporal evidence moves to LoTTE or to a report that
replay suffices. G1 and G2 pass: the method arms are designed and registered.

## 6. Calibration stream and cost

Calibration stream: the same construction applied to MS-Shift's "other" cluster, two
pseudo-clients by k-means (k = 2) and four experiences each, its own manifest. That cluster
has no official evaluation queries, so its pool is a seeded fifteen percent of its training
queries, capped at 3,000 per client because every pool passage must fit the fixed corpus;
its split counts are those of the primary stream, so the recipe is chosen on a stream shaped
like the pilot. On it, rounds
per experience in {2, 4, 8} and learning rate in {2e-5, 5e-5} are compared under arm D and
the pair with the highest mean acquisition is frozen; lambda in {0.5, 1.0, 2.0} is chosen
under arm E as the value with the lowest G whose A is within 0.005 of the best. No deciding
run starts before this is recorded.

Cost: one realistic unit (one client, one experience, one round of training plus one full
retained evaluation pass) is profiled on the L4 first; the pilot's total is extrapolated
from that measurement and written into section 14. The earlier "one L4 day" figure is a
guess and is not registered.

## 7. Code and records

Four new top-level modules and their tests; `federated_forgetting.py` is not changed.

- `experiences.py`: loads MS MARCO and MS-Shift, reports the feasibility table, builds and
  verifies manifests, and materialises per-(client, experience) data in the shape
  `client_train` consumes.
- `memory.py`: the retained-query budget and seeded reservoir replay.
- `regression.py`: per-query nDCG@10 from rankings, positive-part regression, acquisition,
  signed backward transfer, peak forgetting and the matrix summaries.
- `continual_driver.py`: the experience loop over rounds and schedules, the five arms,
  reference snapshots, the evaluation matrix, per-query outputs and a result JSON with the
  same provenance record as the static driver.
- `validate_continual.py`: exists before the first real run. It checks checkpoint-chain
  continuity, the experience sequence against the manifest, source and data digests,
  memory capacity in every round, that no test id appears in any training, replay or guard
  set, acquisition-reference hashes, the recorded client models of local-only runs, and
  seed and configuration identity. A two-client, two-experience synthetic end-to-end test
  runs the driver and the validator together before the calibration stream is launched.

Per round the driver persists the same state payload as the static driver (clients,
broadcast, global, hashes) so the aggregate can be recomputed from the persisted states.

Dependencies added for the manifest build only: scikit-learn (clustering) and bm25s (hard
distractors).

## 8. Beyond the pilot

LoTTE (3.6 GB; corpora of 0.39 to 2.04 M passages per domain; about 5,200 to 5,600 queries
per domain; several relevant passages per query) replicates the pilot with a different
query style; corpora capped by the same rule. LongEval-Retrieval 2023 English (June, July
and September 2022 snapshots, click-model labels in three grades, document time metadata)
is a chronological robustness evaluation of trained models, not a training stream; the
author downloads it from LINDAT/CLARIN into `beir_data/longeval-2023/` with the raw layout
and checksums, and its July and September labels stay sealed from method development. Any
later training on July feedback is a derived protocol and is labelled as such. FreshStack
is optional. Each dataset gets a readiness table built from the actual files before it
enters a run.
