# Supervisor forecasts for E3 — committed before any E3 data

*2026-08-31. Companion to E3_PREREGISTRATION.md §4. The student's signed
predictions are the registered ones; these are mine, on record so both sheets
can be scored in the supplement. Nothing here binds the student.*

| P | my forecast |
|---|---|
| P1 | FedSpan > uniform-over-distributions > plain uniform > q-FedAvg > norm-equalised uniform > n_k (positions 3–5 are coin-flips; the confident calls are the two ends) |
| P2 | E_uniform ≈ 0.010, E_FedSpan ≈ 0.005, E_nk ≈ 0.030 (mean over both singletons) |
| P3 | clone-block mass ≈ 0.58 / 0.52 / 0.48 at rounds 1 / 8 / 15 |
| P4 | **NO** — FedSpan does not beat uniform-over-distributions by >0.022; F1 fires and the redistribution demotion proceeds |
| P5 | **NO** — q-FedAvg does not match FedSpan's singleton protection within 0.022; T-E's differentiator survives |
| P6 | **NO reversal** — FedSpan again loses worst-client and variance to norm-equalised uniform (E3's worst clients ARE the clone shards it takes mass from); F2 and F3 fire |
| P8 | FedSpan mass growth m=2→4 ≈ **0.06** (vs uniform's mechanical 0.167) |

Net forecast: **F1/F2/F3 fire, F4 does not.** The paper ships phenomenon-first
with FedSpan as the mechanism-matched redistribution section — winning the
mechanism outcomes (mass discount, alignment, sub-mechanical growth) while
losing the fairness metrics. That is the coherent outcome, not a failure mode.

## Reasoning in brief

- **Alignment ceiling:** FedSpan's γ is optimal on the Gram by construction,
  so it tops any alignment ordering. The sim's remaining FedSpan-vs-over-dist
  γ gap (~0.07) converts, at E1's observed γ→nDCG exchange rate, to roughly
  0.01–0.015 nDCG — under the 0.022 resolvable margin. Hence P4 NO.
- **The capped-norm wrinkle (main uncertainty in P1's middle):** E3 caps
  shards at 167 steps but singletons keep 500, so singleton deltas are likely
  the LARGER ones. Plain uniform raw-averages deltas — implicitly weighting by
  norm — so it tilts TOWARD the singletons (block direction mass ≈
  3r_s/(3r_s+2r_g) ≈ 0.43–0.46 if r_g ≈ √3·r_s), while norm-equalised uniform
  pins the block at 0.60. This is the opposite of their E1 relationship and is
  why I rank plain uniform above norm-equalised here. Low confidence.
- **P2 basis:** historical capped-regime measurements — n_k scifact E =
  0.025/0.033/0.025, arguana 0.005/0.002/0.005; uniform ≈ 0.000–0.006.
  E3 pushes n_k block mass to ~0.985 (worse) and degrades uniform's clone-
  regime alignment (sim: 0.46→0.30), so I nudge both up from history.
- **P3 basis:** sim ceiling 0.68→0.54 is same-data proxies at K=4; disjoint
  shards are less clone-like and K=5 uniform reference is 0.60, so I shrink
  the discount toward it.
- **P6:** E3's lowest-scoring clients are the NFCorpus shards (~0.33) — the
  very block FedSpan discounts — while the singletons it lifts are the best
  scorers (0.50–0.69). Worst-client and variance therefore move against
  FedSpan by construction of the testbed, not by accident.
- **P8:** T-D gives exactly 0 for exact clones; disjoint near-clones warrant
  a small positive growth. A third of mechanical (≈0.06) is my point guess,
  range 0.03–0.10.
