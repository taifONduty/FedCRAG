# E3 Clone-Federation Pre-Registration

*Supersedes `E3_preregistration_TEMPLATE.md` (2026-08-22). Rewritten 2026-08-31
per the pre-E3 audit (`2026-08-30_pre_E3_audit_verdict.md` §3–§4) and the
external-review triage (`2026-08-31_external_review_triage.md`).*

> **Non-delegable (plan §7).** Read §2 and §3, write §4 in your own words,
> sign §9, and commit **before E3's first GPU run**. Committing a prediction
> you privately doubt is worse than predicting wrong — write what you actually
> expect. Wrong predictions are reported too; that is what makes the right ones
> evidence. §5 is pre-committed and is **not** yours to soften after seeing
> results.

---

## 0. Why this experiment matters more than it used to

E3 was designed as a symmetry/tie check. It is now **the make-or-break
experiment for the FedSpan method**, for a reason discovered on 2026-08-31 in
our own E1 files (§2.2): on the metrics the FL-fairness literature actually
reports — worst-client score and cross-client variance — **plain uniform
weighting on ordinary LoRA beats FedSpan at both seeds.**

FedSpan's one remaining claim rests on two results whose scopes must not be
conflated (corrected 2026-08-31, per the external second review). (i) *Clone
symmetry* (T-E): a deterministic, permutation-equivariant, **loss-only** rule
gives equal weights to clients with IDENTICAL losses. E3's disjoint shards
are near-duplicates, not loss-ties — T-E motivates this experiment, it does
not prove its outcome. (ii) *Direction-duplication invariance*
(MATH_FOUNDATIONS T-D, verified 2026-08-27): duplicating a client DIRECTION
leaves the convex hull, hence the min-norm point, unchanged — while
uniform-over-clients mass on the duplicated block grows mechanically. The
clone federation instantiates T-D approximately; the m-sweep (§1) measures
the deviation near-duplication causes; and whether loss-driven rules actually
fail here is what the q-FedAvg arm **measures rather than assumes**. A clone
federation remains the one regime where uniform is wrong by construction. If
FedSpan cannot win here, it cannot win anywhere, and §5 says what happens then.

---

## 1. The registered experiment

**Federation (K = 5).** NFCorpus partitioned by QUERY into 3 disjoint shards
("three organizations independently indexing the same literature"), plus
**two** singletons: ArguAna and SciFact. Two singletons, not one, so the
primary outcome cannot again reduce to ArguAna alone (audit B1).

- Shards share the parent corpus **by object identity**; train and eval queries
  are partitioned disjointly (LPT, seeded random tie-break). Gold-document
  eviction is structurally impossible.
- `--conserve_shard_steps`: per-shard caps [167, 167, 166] summing to the
  monolithic 500, so the clone block does not get 3× the work (audit B4).
- R = 15 rounds, capped regime.

**Arms (6).**

| arm | what it isolates |
|---|---|
| **FedSpan** (`--fedspan_direction_policy exact`) | the method |
| **norm-equalised uniform** (`fixed`, w = 1/K, same 1/r_k rule and step policy) | **the true foil.** Kills the norm-equalisation confound (audit B2) |
| **uniform-over-distributions** (`fixed`, w = 1/9,1/9,1/9,1/3,1/3) | the oracle that already knows which clients are clones (audit B3) |
| **plain uniform** | the naive baseline everyone uses — and the one that beat FedSpan in E1 |
| **n_k** | the pathology arm; establishes the phenomenon inside this federation |
| **q-FedAvg** | the loss-driven representative; the direct test of T-E |

Dropped, with reasons stated here so it is not a silent cut: **AFL**
(redundant with q-FedAvg as the loss-only representative), **MGDA** (≡ FedSpan
at ε=1 on normalised updates — a near-duplicate arm), **CRAFT** (its equality
constraint is structurally infeasible in a clone federation: ρ ∉ range(C)
whenever shards differ in size, measured relative residual 0.77, output not a
simplex vector — **marked dead, not deferred**).

`q` for q-FedAvg is taken from E2. **If E2 has not run when E3 launches, q is
pre-registered at q = 1.0** (the canonical q-FFL setting) and that substitution
is recorded in §8 rather than chosen after seeing E3.

**Instrumentation (behavior-neutral, declared before signature).** Every
FedSpan-pipeline arm logs shadow sketch telemetry
(`--fedspan_shadow_sketch 1024 4096`): the sketched Gram a shared per-round
Gaussian projection would have produced, the direction it would have chosen,
and that direction's worst-case alignment in the TRUE geometry — next to the
true solve, every round. Applied weights are bit-identical with telemetry on
and off (under test; commit `fedspan-e3-build`). Purpose: turns the CPU
feasibility probe for a secure-aggregation-compatible Gram
(`2026-08-31_secure_gram_sketch_feasibility.md`) into measured-in-deployment
evidence at ~2 min CPU per run. The sketch is diagnostics only and is not an
arm; nothing in §5 reads it.

**Budget: 33 runs ≈ 69 L4-h ≈ $63.**
6 arms × 3 seeds (42/123/2024) = 18; K=2 clone control (FedSpan +
norm-equalised uniform) × 3 seeds = 6; partition-robustness = 4;
nondeterminism floor (one cell repeated at a fixed seed) = 1;
**m-sweep light (adopted 2026-08-31)** = 4: `nfcorpus:2` and `nfcorpus:4`
cells, FedSpan + norm-equalised uniform, seed 42 only, launched only after
the P0 gate passes on the main grid. With two singletons, uniform-over-clients
puts clone-block mass m/(m+2) on the block **mechanically**: 0.500 / 0.600 /
0.667 at m = 2 / 3 / 4. The m-sweep measures whether FedSpan's block mass
grows sub-mechanically — the graceful-deviation-from-invariance curve a
single m = 3 point cannot show. SECONDARY mechanism outcome: nothing in §5
reads it, and it cannot rescue a §5 miss.

---

## 2. On the record BEFORE any prediction is written

### 2.1 CPU simulation (measured 2026-08-22; K=4 proxy, cross-seed NFCorpus)

Achieved worst-case alignment γ (§3 convention) and clone-block mass:

| round | clone cos | uniform γ / mass | over-distributions γ / mass | n_k γ / mass | FedSpan γ / mass |
|---:|---|---|---|---|---|
| 1 | 0.23–0.24 | 0.460 / 0.750 | 0.398 / 0.500 | 0.052 / 0.994 | **0.585** / 0.677 |
| 8 | 0.60–0.61 | 0.279 / 0.750 | 0.488 / 0.500 | **−0.092** / 0.994 | **0.617** / 0.568 |
| 15 | 0.75–0.76 | 0.299 / 0.750 | 0.585 / 0.500 | **−0.045** / 0.994 | **0.657** / 0.542 |

Three things to notice before predicting:

1. **n_k's alignment goes NEGATIVE** by round 8. The aggregate update points
   *away* from the singleton. That is the erosion mechanism, visible directly.
2. **uniform-over-distributions closes 80%** of the uniform→FedSpan gap at
   round 15 (0.299 → 0.585 → 0.657) using only the knowledge of which clients
   are clones. FedSpan must beat *this*, not plain uniform.
3. **The ordering flips.** At round 1 over-distributions (0.398) is *worse*
   than plain uniform (0.460); by round 8 it is far better. Redundancy has to
   accumulate before discounting it pays.

**This simulation is a CEILING, not an expectation.** It models same-data
cross-seed proxies, not disjoint thirds; real shards will be less clone-like.

### 2.2 E1 facts that constrain what FedSpan can claim (2026-08-31)

Absolute final nDCG@10, frozen-A FedSpan vs ordinary-LoRA uniform:

| | mean | **worst client** | **cross-client var** |
|---|---:|---:|---:|
| FedSpan s123 / s2024 | 0.4607 / 0.4590 | 0.2498 / 0.2444 | 0.03090 / 0.03144 |
| AB-uniform s123 / s2024 | 0.4578 / 0.4569 | **0.2662 / 0.2689** | **0.02604 / 0.02558** |

FedSpan wins the mean by 0.003 (noise) and **loses worst-client and variance at
both seeds.** It takes from FiQA (worst client, 0.25) and gives to SciFact and
ArguAna (the two best, 0.69 and 0.55). Also: FedSpan beat frozen-A uniform 4/4
clients at seed 123 but only **2/4 at seed 2024**.

Predict knowing this. A prediction that ignores §2.2 is not a real prediction.

### 2.3 Power

E1 singleton gaps give sd ≈ 0.0087. At 3 paired seeds the 95% half-width is
**≈ 0.022 nDCG**. E3 resolves singleton effects of about 0.02 and **nothing
smaller**. Since over-distributions already closes 80% of the alignment gap,
the decisive contrast may sit below this floor — which is itself a reportable
outcome, not a failure to be papered over.

---

## 3. The γ convention — settled here, because it inverts the answer

γ names three different scalars in our code and notes. **All predictions and
all reported comparisons use convention C.**

| | definition | cross-arm ratio (r8, sim) |
|---|---|---:|
| A | raw `min_k (Cw)_k` | ×2.05 |
| B | `sqrt(wᵀCw)` — what the solver returns as `min_norm_value` | **×0.93** |
| **C** | **`min_k (Cw)_k / sqrt(wᵀCw)`** — achieved worst-case alignment | **×2.21** |

**The trap:** convention B is the solver's own output, and comparing arms in it
says FedSpan is *worse* than uniform (0.617 vs 0.667). That is because
`sqrt(wᵀCw)` is the mixture NORM, which equals the achieved alignment only at
the optimum — where FedSpan sits and no other arm does. Convention C is the
only quantity comparable across arms; it agrees with B at the FedSpan optimum
(verified: 0.656714 vs 0.656787) and with the duality identity.

**Erosion E, operationalised.** For client c in arm a at seed s:
`E(c,a,s) = max_{t≤T} nDCG@10_c(t) − nDCG@10_c(T)` — how much of its own best
the client gives back by the end. Reported per client, per seed, paired across
arms. Positive E is erosion.

---

## 4. MY PREDICTIONS

Committed BEFORE the first E3 GPU run. Date: 2026-08-31.

**Provenance (recorded honestly):** these predictions were adopted by the
student after reviewing the supervisor's forecast sheet
(`2026-08-31_supervisor_E3_forecasts.md`), with the evidence and reasoning for
each laid out in the supervised session of 2026-08-31. Adopted-after-review is
the registered status; both sheets are scored in the supplement. P7 is the
student's own sentence, transcribed verbatim from their reply.

**P1 — Arm ordering by mean singleton (ArguAna + SciFact) final nDCG@10.**
MY ORDER: FedSpan > uniform-over-distributions > plain uniform > q-FedAvg >
norm-equalised uniform > n_k. Positions 3–5 are declared coin-flips; the
confident calls are the two ends. q-FedAvg lands mid-pack because its h_k
normalisation removes the size skew, leaving only a loss tilt — and the three
shards are loss-ties while cross-domain loss differences are modest against a
157:1 size story. Plain uniform is ranked above norm-equalised uniform because
of the capped-norm wrinkle: shards are capped at 167 steps while singletons
keep 500, so singleton deltas are likely larger, and raw averaging implicitly
weights by norm — tilting toward the singletons (block direction mass
≈ 0.43–0.46 vs norm-equalised 0.60). Low confidence on that pair.

**P2 — Erosion E on the singletons** (mean over ArguAna and SciFact):
E_uniform ≈ 0.010  E_FedSpan ≈ 0.005  E_nk ≈ 0.030
Basis: capped historical measurements (n_k scifact E 0.025/0.033/0.025,
arguana ≈ 0.004; uniform 0.000–0.006), nudged up because E3 raises n_k's
block mass to ≈0.985 and degrades uniform's clone-regime alignment.

**P3 — Measured FedSpan clone-block mass** at rounds 1 / 8 / 15
(uniform = 0.60 at K=5; sim ceiling was 0.68→0.54 at K=4):
0.58 / 0.52 / 0.48 — the sim is same-data proxies, so its discount is shrunk
toward the K=5 uniform reference for disjoint shards.

**P4 — Does FedSpan beat uniform-over-distributions** on singleton nDCG@10 by
more than the 0.022 resolvable margin (§2.3)?
**NO.** Over-distributions already closes ~80% of the alignment gap; the
remaining γ gap (~0.07) at E1's observed γ→nDCG exchange rate converts to
≈0.01–0.015, under the margin. **This prediction expects F1 to fire and the
§5.1 demotion to proceed.** Signing it means running E3 for the mechanism
chain and the invariance evidence, with a YES as upside surprise.

**P5 — Does q-FedAvg protect the singletons?**
**NO** — it does not match FedSpan's singleton protection within 0.022 (the
F4 operationalisation), for the same reason it lands mid-pack in P1. So F4 is
predicted NOT to fire: T-E's differentiator survives. Deliberate tension with
P4: FedSpan is predicted to lose to the oracle that is told who the clones
are, but beat the rule that can only see losses — that conjunction is the
narrowed claim itself.

**P6 — Worst-client and cross-client variance.** Reversal of §2.2? **NO.**
E3's worst-scoring clients are the NFCorpus shards (~0.33) — the block FedSpan
discounts — while the singletons it lifts are the best scorers (0.50–0.69).
Worst-client and variance move against FedSpan by construction of the testbed.
**F2 and F3 are predicted to fire.**

**P8 — m-sweep (secondary).** Uniform's clone-block mass grows mechanically
by 0.167 from m=2 to m=4. FedSpan's grows by: ≈ 0.06 (range 0.03–0.10;
sub-mechanical). Not 0.00 — that is the exact-clone answer to a near-clone
question.

**Net position signed into the record:** F1, F2, F3 are predicted to fire and
F4 not to. The expected outcome is the phenomenon-first paper with FedSpan as
the mechanism-matched redistribution section — winning the mechanism outcomes
(mass discount, alignment, sub-mechanical growth) while losing the fairness
metrics. The fallback clause §5.1 is not a feared contingency; it is the
predicted path.

**P7 — The disconfirming result I accept in advance.**
Student's reply, transcribed verbatim (2026-08-31): "Just test them properly,
don't do any mistakes - keep sure of it, and start it already!"
**Registrar's note, for the record:** this names no disconfirming outcome, so
NO student-specific drop condition beyond §5.1 is registered. The binding
demotion rules remain §5.1 exactly as written. The supervisor's own
mechanism-failure criterion (P0 passes with high clone cosines yet FedSpan's
block mass stays ≈0.60 — the solver failing to discount redundancy it can
demonstrably see) stands in `2026-08-31_supervisor_E3_forecasts.md` as a
SUPERVISOR-held commitment, not attributed to the student.

---

## 5. PRE-COMMITTED DECISION RULES — binding, not revisable after results

**Primary outcome:** mean singleton (ArguAna, SciFact) final nDCG@10, paired
across the 3 seeds.
**Co-primary:** worst-client final nDCG@10 and cross-client variance — because
§2.2 showed these are exactly where FedSpan loses, and they are the
conventional q-FFL/Ditto metrics a reviewer will compute whether we report them
or not.
**Secondary (mechanism only, never a substitute for the primary):** γ
(convention C), clone-block mass, erosion E.
**Decisive comparator:** **uniform-over-distributions**, not plain uniform.

### 5.1 The fallback clause

**Any ONE of the following fires the demotion:**

- **F1.** FedSpan does not beat uniform-over-distributions on the primary
  outcome by more than **0.022** (the resolvable margin, §2.3), averaged over
  paired seeds.
- **F2.** FedSpan loses worst-client nDCG@10 to norm-equalised uniform at **≥2
  of 3 seeds**.
- **F3.** FedSpan's cross-client variance exceeds norm-equalised uniform's at
  **≥2 of 3 seeds**.
- **F4.** q-FedAvg matches FedSpan's singleton protection to within 0.022 —
  which refutes T-E, the only claim no baseline can make.

**Consequence, pre-committed:**

> FedSpan is demoted from the title and headline to a single method section.
> The paper ships **phenomenon-first**: the lose-lose result and depth-selective
> erosion are the contribution; FedSpan is reported as a principled response
> that is baseline-competitive, with its losses stated in the same table as its
> wins. The method claim narrows to **redistribution**, not improvement.

**And, explicitly, what we will NOT do:** select another testbed; add an arm
post-hoc to rescue the result; re-run E3 with different shards after seeing the
outcome; restore E8; promote a secondary mechanism outcome (γ, clone mass) to
primary because the primary missed. A miss is reported as a miss.

### 5.2 What does NOT trigger demotion

FedSpan winning the singletons while losing the mean is the **redistribution**
story, and it is publishable if stated honestly. Losing the mean is not a
failure; losing to uniform-over-distributions is.

---

## 6. Gates that must pass BEFORE the full spend

- **G0 — partition non-degeneracy (free, CPU).** Run
  `e3_shard.partition_seed_sensitivity(nfcorpus_payload, "nfcorpus", 3, seeds)`
  on the **real BEIR payload**. The shard seed only breaks LPT ties, so a
  payload whose per-query pair counts are all distinct is seed-inert and the 4
  partition-robustness runs would buy exactly one. **Require ≥ 3 distinct
  partitions across the 3 seeds**; otherwise drop those 4 runs and report the
  measured count. *(Not yet checked — no BEIR data on the dev machine.)*
- **P0 — round-1 clone-cosine go/no-go.** After the first round of the first
  seed only: require mean clone-block cosine **≥ 0.15** and clone-singleton
  cosine strictly below it. If it fails, the shards are not a clone federation,
  E3 tests nothing, and the run **stops** with the measured cosines reported.
- **G1 — nondeterminism floor.** The repeated cell bounds run-to-run GPU
  variation. Any arm difference below that floor is reported as
  indistinguishable, regardless of sign.

---

## 7. Scoring rule for the predictions in §4

Fixed now so a near-miss cannot be argued into a hit afterwards.

- **P1 (ordering):** Kendall τ between predicted and realised ranking,
  **plus** each of the 15 pairwise comparisons scored hit/miss individually.
  Report both; the τ alone can hide a wrong call on the decisive pair.
- **P2, P3 (numeric):** hit if within the 95% paired interval; near-miss if
  within 2×; else miss. Sign errors are always misses, however small.
- **P4, P5, P6 (binary):** hit/miss, no partial credit.
- **P7:** not scored — it is a commitment, and §5 enforces it.
- **P8 (numeric, secondary):** same hit/near-miss rule as P2–P3; mechanism
  narrative only — it cannot offset any §5 trigger.

Every prediction is reported, hit or miss, in the paper's supplement.

---

## 8. Deviations log — fill during and after E3

| date | deviation from this registration | reason |
|---|---|---|
| | | |

---

## 9. PART B — Companion program registration (approved 2026-08-31)

The experiments below are registered before any of them runs, so no result
can be selected after the fact. Each carries its committed prediction or
decision rule. All use the E1 configuration (Contriever-110M, LoRA r=16,
full-work R=8) unless stated. Engineering gate for every item: the cell runs
end-to-end on a smoke round and `validate_e0.py` accepts it BEFORE the
multi-seed spend — the pre-E3 audit's lesson, applied forward.

### 9.1 E-local — the outside option (blocks the headline)

Each silo fine-tuned ALONE from the same initialization, LoRA config, and LR,
trainable-A+B coordinate. Budget-matched: 8 full epochs of its own data (the
local step count it would execute across the federated full-work run:
nfcorpus 27,640 / fiqa 3,536 / scifact 224 / arguana 168 steps). Evaluated
after every epoch; the registered comparison point is epoch 8
(budget-matched), epoch 1 secondary, full trajectory reported — no best-epoch
selection. Seeds 42/123/2024. 12 runs ≈ $10.

**On record before running:** the historical seed-42 1-epoch control (dirty
commit `ee1d881f0ae6`) already shows every client under n_k below its
local-only outside option and every client under uniform above it.

**Decision rule.** The paper claims "worse than declining federation" for a
client iff fed-n_k < budget-matched local-only at ≥2/3 seeds, paired.
"Federation rational for every silo under uniform" requires uniform >
budget-matched local-only for all four clients at ≥2/3 seeds. Anything less:
claims stay at the pre-federation-backbone level. Both outside options
(1-epoch and 8-epoch) are reported; where they disagree, the one LESS
favourable to our claim is the headline comparator.

### 9.2 P3 — frozen-A + n_k (mechanism discriminator)

Frozen-A coordinate, n_k weighting, full-work, seeds 42/123/2024. 3 runs ≈
$18. Freezing A removes the factor-aggregation residual by construction while
keeping the suspect weighting.

**Decision rule.** Erosion persists (both minorities below the frozen
backbone at ≥2/3 seeds) → the "conflict, not residual" mechanism claim is
licensed and the coordinate gap closes. Erosion disappears → the residual
dynamics we currently dismiss are causally implicated: the mechanism section
is rewritten and "conflict, not residual" is withdrawn, not softened.

### 9.3 P1 — q-dose–response (the model's sharpest test)

w_k ∝ n_k^q, q ∈ {0.25, 0.5, 0.75}, trainable-A+B, full-work, seeds
123/2024 (endpoints q=0 and q=1 exist at both seeds). 6 runs ≈ $34. Needs a
small driver extension (weighting exponent); TDD + validator acceptance
before spend.

**Registered predictions** (from the system model, committed before data):
minority erosion E_k monotone increasing in q; majority final score
non-monotone in q, peaking strictly below q=1; majority-vs-rest Gram cosine
decreasing in q; minority leave-one-out alignment decreasing in q — the
dose–geometry–outcome chain.
**Falsifier, accepted in advance:** a flat or erratic minority-erosion curve
in q falsifies assumption A1 — weight-skew is then the wrong explanatory
variable, and the paper says so.

### 9.4 FedMGDA+ native-step arm (the "F vs E" question)

Source-faithful FedMGDA+ (arXiv:2006.11489, Algorithm 1, ε=1, its decaying
global step) in the frozen-A coordinate, full-work, seeds 42/123/2024.
3 runs ≈ $15. Exact hyperparameters transcribed from the source before
launch, recorded in §8.

**Decision rule.** If FedMGDA+ matches FedSpan on the primary outcomes within
the resolvable margin, FedSpan's remaining delta is the step law alone, and
the paper presents it exactly as that — an instantiation choice on known
geometry, not a new aggregation method. Accepted in advance.

### 9.5 M1 — query-level margin replay (no training)

From saved E1 round states (confirmed saved for the trainable arms):
per-query nDCG@10 and score margins at cutoffs {1, 3, 5, 10, 20, 100}, every
round, n_k and uniform arms, seeds 123/2024. Replay-only GPU ≈ $5.

**Registered prediction:** top-10 relevant-document exits concentrate in the
lowest pre-round margin quartile and are more frequent in rounds where the
client's leave-one-out alignment is negative. If flips are margin-independent,
"depth-graded damage" demotes from mechanism to descriptive observation.

### 9.6 Second backbone — decisive cells only

BGE-base-en-v1.5. Headroom gate first (per-silo local fine-tune must beat its
frozen backbone; any silo failing the gate → report and stop, plan §4 E5
rule). Then {n_k, uniform} × full-work × 3 seeds = 6 runs + gate ≈ $34.
**Claim rule:** qualitative replication (minority erosion under n_k, none
under uniform) upgrades the phenomenon to two-backbone; non-replication is
reported as a scope limit, not hidden.

### 9.7 Explicitly out

The angle-intervention experiment and the secure-Gram MPC section are STRETCH
items: registered as designs, run only if §9.1–§9.6 and Part A finish early.
E8 stays cut. E4 — see §10.

## 10. Recorded program decisions (2026-08-31, student-approved)

1. **"Continual" is dropped from the WWW 2027 paper.** E4 is removed from the
   submission program. The word leaves the title, abstract, and claims; C2
   (no replay) remains a design constraint of the testbed, described as such,
   not claimed as a demonstrated continual capability. The thesis keeps
   continual adaptation as future work. Rationale: E4 buys one title word at
   the cost of a second literature, a baseline family, and compute that
   §9.1–§9.6 use better (external second-review, triaged 2026-08-31).
2. **The §9 program is approved** at the budget in §11.
3. Title decision stays deferred until E3 lands; no candidate title contains
   "continual".

## 11. Program budget

| item | runs | est. $ |
|---|---:|---:|
| Part A: E3 incl. m-sweep light | 33 | 63 |
| 9.1 E-local | 12 | 10 |
| 9.2 frozen-A n_k | 3 | 18 |
| 9.3 q-sweep | 6 | 34 |
| 9.4 FedMGDA+ | 3 | 15 |
| 9.5 margin replay | replay | 5 |
| 9.6 second backbone | 6 + gate | 34 |
| **total** | **~64** | **~$179** |

Cap including 20% operational reserve: **$215** — inside the master plan's
reserved envelope.

## 12. Signature — covers Parts A and B

I have read §2 (including the E1 results in §2.2 that count against the
method), §3, §5, and Part B (§9–§11). My predictions in §4 are what I actually expect. I accept
the §5.1 fallback clause as binding before seeing any E3 result.

Signed (name, date): **Turjo, 2026-08-31** — attestation given in the
supervised session reply of 2026-08-31 ("Name: Turjo"), transcribed at the
student's instruction. Predictions adopted-after-review per the §4 provenance
note.

Commit hash of this file at registration: recorded in `REGISTRATION_LOCK`
by the follow-up lock commit (a commit cannot contain its own hash).
