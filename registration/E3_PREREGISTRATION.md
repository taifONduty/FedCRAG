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
| 2026-08-31 | Interpretation clarified, no run/budget change: "K=2 clone control" is implemented literally as the two-client no-clone anchor federation {nfcorpus unsharded, arguana} × {FedSpan, norm-equalised uniform} × 3 seeds, per the original audit wording. Encoded in `e3_manifest.py`, pinned by tests. | The signed §1 phrase was ambiguous between this and an m=2-shards reading; the literal reading avoids double-running the m-sweep's m=2 cells and keeps the registered count at exactly 33. |
| 2026-08-31 | §9.4 hyperparameters transcribed from arXiv:2006.11489 as required: Algorithm 1 — updates normalized to unit length (line 5), λ* = min-norm over {λ∈Δ, ‖λ−λ0‖∞≤ε} (line 6) with ε=1 making the ball inactive (Eq. 25), update w−η_t·d_t un-normalized (line 9). Schedule: their settled choice is exponential decay η_t = β^⌊t/100⌋, β = decay^{100/T} (§6.1.5); below 100 rounds that staircase never fires, so the continuous form η_t = η0·decay^{(t−1)/(T−1)} is implemented, preserving the design target η_T/η_1 = decay. η0 = 1.0, decay = 0.1 — the centres of their Table-5 grids (η∈{0.5,1,1.5,2}, decay∈{0…½}, their example 1/10). Implemented as `--fedspan_step_policy fedmgda` (same direction, solver, and gates as FedSpan; only the step law differs), commit-tested. | Registered precondition of §9.4. |
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

Commit hash of this file at registration: 1b7d397a52a9102289723e1d34e2aa2242d7f070

## 13. Addendum — Option A benchmark blocks (registered 2026-09-06, before any A1/A2/A3 data)

Signed sections above are unchanged. This addendum records the decision rules for the
blocks accepted by Turjo on 2026-09-06 ("option A of course") and the standing of the
Part B items as of that date. Deviations from Part B are stated, not hidden.

**Standing of Part B on 2026-09-06.** 9.1 E-local complete (3 seeds; licensed 3/4 silos,
refused for NFCorpus). 9.2 P3 complete (0/3; withdrawal executed). 9.4 FedMGDA+ complete
(3 seeds; outcome (b) with the mean-drift caveat — worst-client outside the margin, mean
inside at its edge). 9.5 margin replay complete (clause 1 confirmed 4/4, clause 2
falsified 4/4). **9.3 q-sweep and Part A (E3) will not run before the WWW deadline** —
decided 2026-09-06, to be stated in the paper's limitations; both stay registered for the
thesis. 9.6 second backbone is expanded below.

**A0 — matched-total-work control.** Registered separately in
`experiments/equal_work/REGISTRATION.md` (branch `codex/equal-total-work`, commit 2d6ec4b).
Reading rule fixed in `research_workspace/supervisor/2026-09-06_equal_work_pilot_reading_guide.md`
before its result: minorities still below backbone at both seeds ⇒ work skew is not the
cause, interaction stands; both recover ⇒ A4 gains a work term and "two switches" becomes
three; disagreement ⇒ preserved, next test is the clean seed-42 pair.

**A1 — second backbone, BGE-base-en-v1.5 (expands 9.6).** Gate: independent per-silo
fine-tune must beat frozen on every silo (as for Contriever); failure ⇒ BGE dropped, the
E5-base gate is tried once, and if that fails the paper stays single-backbone and says so.
Cells: AB $n_k$, AB uniform, frozen-A $n_k$, frozen-A uniform; seeds 123, 2024, 42; R=8,
one local epoch, lr 2e-5, rank 16 (Contriever recipe; no retuning). Order: s123 AB $n_k$ →
AB uniform → frozen-A $n_k$ (decisive trio) → remainder. **Predictions:** (i) under AB
$n_k$ at least one of SciFact/ArguAna finishes below the BGE backbone at ≥2 of 3 seeds;
(ii) under AB uniform no silo finishes below the backbone; (iii) frozen-A $n_k$ removes
any below-backbone outcome at ≥2 of 3 seeds. **Falsifiers:** (i) fails ⇒ the phenomenon is
reported as Contriever-specific and the title's generality is withdrawn to "in a Contriever
federation"; (iii) fails while (i) holds ⇒ the interaction claim is narrowed to Contriever
and the mechanism section says so. Primary outcome frozen-anchored nDCG@10 per client;
co-primary worst-client and cross-client variance; resolvable margin 0.022 carried over.

**A2 — published baselines (new).** q-FFL ($q{=}1$, its loss-power rule), AFL (its mixture
ascent, $\eta{=}0.1$), FedNova (normalised averaging by local steps), each in the
trainable coordinate at seeds 123 and 2024, Contriever, same recipe, hyperparameters from
the source papers as implemented on 2026-08-22 and not retuned. Validator recomputation
references and tamper tests for all three must exist and pass before any run counts.
**Decision rule:** a baseline that keeps both minorities at or above the backbone at both
seeds while costing the majority no more than 0.022 relative to AB $n_k$ is reported as
removing the harm; the paper then presents FedSpan's step law as one instantiation among
working fixes and the mechanism section cites that baseline as a third switch. A baseline
that reduces but does not remove the harm is reported as such. No baseline is retuned on
our data before its result is recorded.

**A3 — exactness control (new, to be built).** Both LoRA factors trainable, $n_k$ weights,
server applies a FedEx-LoRA-style residual correction so that the applied update equals
the exact average of the products; seeds 123 and 2024. **Prediction:** the below-backbone
harm is removed (both minorities ≥ backbone) at both seeds. **Falsifier:** harm persists at
both seeds ⇒ inexact aggregation is *not* the operative ingredient; the interaction is
re-attributed to the capacity/parameterisation change that freezing $A$ also makes, and the
paper says "freezing a factor removes the harm; we could not attribute this to exactness
alone". Requires its own registration file and validator reference before launch.

**Budget for this addendum:** ≈ 180 GPU-h ≈ $170 measured; authorised $190; table freeze
2026-10-05; no compute after the freeze. Cumulative project compute on 2026-09-06 ≈ $285.

Recorded by the supervisor session at Turjo's instruction, 2026-09-06.

### 13.1 Amendment to A2 — q-FFL's Lipschitz constant (registered 2026-09-06, before any q-FFL run)

The transcribed q-FFL uses the source heuristic L = 1/lr. With full-epoch local updates
this makes every delta weight of order 1e-5 (analysis from recorded E1 round-1 update
norms: supervisor/2026-09-06_qffl_degeneracy_note.md), i.e. a no-op arm that would report
the frozen backbone under the q-FFL label. This is a protocol degeneracy, not a tuning
question, and is fixed before launch as follows.

- The driver gains `--qffl_L` (commit after 78c6dab). Absent, behaviour and filenames are
  unchanged (L = 1/lr). Supplied, L is recorded in the run contract, the result args and
  the filename tag `-L<value>`; the validator recomputes q-FFL's weights with the recorded L.
- **Rule for L (fixed now, evaluated once, before the full runs):** run one q-FFL
  preflight round at seed 123 with L = 1/lr (which records the broadcast-point losses
  F_k and the update norms ||Δ_k|| for round 1). Set
  L* = Σ_k F_k / Σ_k ||Δ_k||², rounded to two significant figures, so that the two
  terms of q-FFL's normaliser h_k are of equal total size at round 1. Use q = 1 as
  transcribed. L* is written into this section with its inputs before the two full runs
  start and is not changed afterwards.
- The paper reports both facts: that the published constant is degenerate under this
  training recipe (one sentence, with the 1e-5 figure), and the rescaled arm's results,
  labelled "q-FFL, L rescaled by registered rule".
- Falsifier and decision rule for A2 are unchanged (§13).

Approved by Turjo 2026-09-06 ("go with option 1, add the qffl_L flag").

#### 13.1.1 L* fixed from the preflight (2026-09-07 10:30 UTC, before any full q-FFL run)

Preflight run: `federated_contriever_seed123_weighted-qffl_r1.json` (commit 42e8246, one full
round, legacy L = 1/lr = 50000, canonical validator passed). Recorded inputs:

| client | broadcast-point loss F_k | update norm ‖Δ_k‖ | ‖Δ_k‖² |
|---|---|---|---|
| NFCorpus | 5.3673 | 3.0747 | 9.454 |
| FiQA | 3.3492 | 1.4647 | 2.145 |
| SciFact | 2.0033 | 0.1966 | 0.0387 |
| ArguAna | 1.5226 | 0.1483 | 0.0220 |

L* = Σ F_k / Σ ‖Δ_k‖² = 12.2424 / 11.6595 = 1.05 → **L* = 1.0** (two significant figures).

Confirmation of the degeneracy on real data: the legacy round applied weights
9.2e-6 / 5.7e-6 / 3.4e-6 / 2.6e-6 (sum 2.1e-5) and moved nDCG@10 by at most 3e-5 on any
client. With L* = 1.0 the same inputs give weights 0.225 / 0.140 / 0.084 / 0.064 (sum 0.51):
a damped step ordered by loss, as q-FFL intends. The A2 q-FFL runs use `--qffl_L 1.0`,
q = 1, seeds 123 and 2024, and are labelled "q-FFL, L rescaled by registered rule".

### 13.2 Block A5: aggregation by measured response (registered 2026-09-08, before any full run)

Design: research_workspace/supervisor/2026-09-08_solution_design_response_aggregation.md.
Arm: `--weighted --weight_by response-maxmin --lora_mode trainable-ab`, Contriever, rank 16,
one local epoch, eight rounds, seeds 123 and 2024. Defaults: dev fraction 0.10 of each
client's training queries (minimum 30, never test queries), lattice step 0.125, scales
0.5 / 1.0 / 1.5, two verified candidates, floor = the frozen backbone's dev nDCG@10 with
delta 0, two halvings. The offline round-1 pilot (results_vm/PILOT_CANDIDATES_20260908/)
motivated the arm and fixed the verification step; nothing below was seen in a full run.

Predictions, in order of confidence.

P1. No client's test nDCG@10 ends below the frozen backbone at either seed.
P2. The worst-client final test nDCG@10 is at least uniform's (E1) at both seeds, and the
    across-client variance is at most uniform's.
P3. FiQA's final test nDCG@10 exceeds its value under every other arm run so far (E1
    uniform 0.266 / 0.269) at both seeds.
P4. The applied weights put more weight on FiQA's update than on NFCorpus's in at least
    five of eight rounds at both seeds (the round-1 pilot's structure persists).

Falsifiers (design note, F3 and F4).

F3. Any client below the frozen backbone on test at either seed: the floor did not
    transfer from dev to test; the method is reported as failed and FedSpan stays the
    paper's repair.
F4. Halving triggered in more than half of the rounds: the response model is not usable at
    the step sizes that matter; the paper describes the arm as a verified search.

Decision rule. P1 and P2 at both seeds: the arm replaces FedSpan as the paper's repair
section, with the pilot as motivation. P1 only: the arm is reported beside FedSpan as a
second repair with its measured guarantee. Neither: a negative result in the appendix. The
handicap is stated either way: the arm trains on 90 percent of each client's training
queries, the baselines on 100 percent.

Cost: about 21 GPU-hours per seed (six extra full evaluations per round). Runs after A2.


### 13.2.1 Amendment: candidate set and selection statistic (registered 2026-09-12 04:27 UTC, before any full arm run; the two bracketed choices of the 10 September draft settled by the saved-round study under the rules written before its results were seen)

Reason. The literature review of 10 September (research_loop/L2) quantifies the optimism of
selecting the maximum over many candidates on dev splits of 70 to 550 queries (about 0.06
nDCG@10 per round on the two small silos at 50 candidates), and the geometry measured on the
seed-123 E1 states (research_loop/2026-09-11_solution_loop.md, section 3) shows that the harm
acts through update magnitudes: uniform weights still give the majority's direction a cosine
of 0.92 to 0.99 with the aggregate. The registered lattice of about 500 candidates is therefore
replaced by a compact set that acts on magnitudes, and selection uses a statistic that a small
silo cannot win through noise.

Arm: `--weighted --weight_by response-maxmin --lora_mode trainable-ab --response_candidates
compact --response_select mean --response_eq_scales 0.5,1.0,2.0 --response_game_scales 1.0,2.0
--response_eq_top 2,3 --response_model_pick`, all other flags as registered in 13.2.

Candidates per round, fixed before anything is measured: uniform; n_k; FedNova; the four
vertices; eq(s) with v_k = s rbar / (K r_k) for s in {0.5, 1, 2}, r_k the product-space norm
of client k's update and rbar their mean; eq{m}(s), magnitudes equalised among the m largest
updates only, m in {2, 3}; game(s) with v_k = s w*_k rbar / r_k for s in {1, 2}, w* the
unit-direction max-min game weights. Chosen by the response model after the K + 1 encodes:
the per-client greedy soup over the vertices, the best uniform subset, and the response
model's best lattice point (model pick). At most 15 contenders.

Selection rule: mean, the worst client's mean paired gain on its held-out queries (the
registered v1 statistic). The two best contenders by that rule among those whose predicted
means clear the floor are verified exactly; the better measured one is applied if its
measured means clear the floor; two halvings; zero step. Floor unchanged: the frozen
backbone's dev nDCG@10 with delta 0. The pessimistic and Pareto rules stay implemented and
their statistics are recorded for every candidate, so the choice can be audited afterwards.

Recorded per round and checked by the validator: r_k, the cosine Gram, w* and the game
value, the family vectors recomputed from the persisted states, the picks against the
recorded statistics, and the magnitude shares of the applied aggregate.

Predictions added to P1 to P4 (unchanged): P5, the applied candidate is from the eq or game
family in at least four of eight rounds at both seeds. P6, the majority's magnitude share of
the applied aggregate is below 0.5 in at least six of eight rounds at both seeds. Falsifier
F5: if P5 fails at both seeds the magnitude theory of section 3 is not what the measured
responses reward, and the paper reports the arm as a measured search without that
explanation.

How the choices were settled. Decision rules written 2026-09-11 00:40 UTC after study round
nk4 and before any other round (research_loop/13_2_v2_addendum_draft.md): the families stay
if a family candidate beats uniform's worst-silo test gain at three or more of the five
remaining rounds (uniform 4, nk 2, uniform 2, nk 6, uniform 6); the selection rule is the one
whose split-half selection on test queries yields the largest total held-out gain summed over
those rounds, among rules whose held-out worst-silo gain is never below uniform's by more than
0.005 in any round; the model pick is included only if the lattice winner's dev-selected
worst-silo advantage transfers to test at three or more of the five. Outcome, from
research_loop/code/decide_addendum.py on the study JSONs (record
research_loop/study/decide_final_2026-09-12T0426Z.txt): families qualify at nk 6, uniform 2
and uniform 6 (3 of 5, kept); the model pick transfers at 4 of 5 (included); summed held-out
gain mean 0.591, pareto 0.585, pessimistic 0.527, none ever below uniform's worst silo by
more than 0.005 (mean). The study's dev split was contaminated (the saved states had trained
on those queries), which is why the rules were written on test responses; the arm's own dev
split is held out of training and has no such artifact.

### 13.2.2 Execution note for block A5 (written 2026-09-11 16:26 UTC, the clock of commit af95536, while the seed-123 same-split baseline runs and before any arm run; the heading first carried a forward-dated stamp of 17:05 UTC, corrected in the next commit)

Order and hardware. The seed-123 same-split uniform baseline is running on the Tokyo L4
(chain8, commit 8eca0e6, started 13:31 UTC). Stage 2 on the same machine, launched only
after 13.2.1 is committed: the arm at seed 123, the arm at seed 2024, then the same-split
uniform baseline at seed 2024, each validated by validate_e0 before the next starts, with
power-off at the end. Because that chain cannot be relied on to finish all four runs before
the pre-defense report deadline (13 September, 17:59 UTC), the seed-2024 same-split uniform
baseline is also executed on the Azure T4 (same commit, data, recipe and mixed precision),
launched after the response study on that machine ends. Precedence, fixed now: for the
paper, the Tokyo copy is the seed-2024 comparator once it validates, and the T4 copy is
reported beside it as a hardware replication; for the pre-defense report, whichever
validated copy exists at writing time is used and its hardware is stated. No table mixes
hardware silently; every use of the T4 copy says so. Each run's provenance records the
platform, the backbone snapshot hash and the data hash, so any difference is visible.

Recipe note. The A5 runs use the driver's default of gradient checkpointing on, as every
A2 run did (commit f25c1cb); the E1 runs had it off. Checkpointing recomputes activations
in the backward pass and changes memory and speed only; the loss and its gradient are the
same functions of the parameters. The A5 baselines and arms share the setting, so the
comparison inside the block is unaffected. Training-critical arguments are otherwise those
of E1: batch 32, evaluation batch 256, learning rate 2e-5, rank 16, trainable A and B, one
local epoch, eight rounds, mixed precision on, held-out fraction 0.10 with minimum 30.

Flags for stage 2, every value from 13.2 or 13.2.1, written out so the record is explicit:
`--weighted --weight_by response-maxmin --lora_mode trainable-ab --response_dev_fraction 0.1
--response_dev_min 30 --response_lattice_step 0.125 --response_scales 0.5,1.0,1.5
--response_verify 2 --response_floor frozen --response_floor_delta 0.0 --response_halvings 2
--response_candidates compact --response_select <rule>` plus the family flags
(`--response_eq_scales 0.5,1.0,2.0 --response_game_scales 1.0,2.0 --response_eq_top 2,3` if
the families stay, `--response_eq_scales none --response_game_scales none
--response_eq_top none` if the decision rule drops them; 'none' switches a family off,
added to the driver and validator under test on 2026-09-11 at commit 308d23d) and
`--response_model_pick` only if the rule includes the model pick. The baseline at seed
2024 adds `--dev_holdout` to the common flags and nothing else. The chain script refuses
to start unless the repository is at the commit carrying 13.2.1 and every placeholder is
filled from the recorded output of the decision script.


### 13.2.3 Record of deviations and harness corrections (written 2026-09-21 15:25 UTC, the clock of the commit that adds it; after both A5 runs; no recorded result changes)

Deviations of the executed A5 runs from 13.2.1, found by the source audit of 17 September
2026 (AUDIT_REPORT.md, findings F04 and F05) and confirmed on the run records on 21 September:

1. Contender count. 13.2.1 says "at most 15 contenders". Both runs logged 19 or 20 contenders
   in every round: the fixed points, the family candidates and the derived picks. The
   registered selection rule, floor and exact verification were applied to that larger list.
   The cap is not amended after the fact; the runs are reported with the counts they logged.
2. The game family. 13.2.1 calls w* "the unit-direction max-min game weights". The code
   (response_arm.update_geometry) solves the max-min LP over the simplex, max_w min_k (Cw)_k,
   and records its payoff as game_value. That LP is not the unit-direction (minimum-norm) game:
   on the three-client exhibit of tests/test_direction_policy.py the LP direction reaches a
   worst-case cosine of 0.4116 where 0.5484 is attainable. The game(s) candidates of both runs
   were built from the LP weights; their results stay attached to that solver. From commit
   b6e2590 the record also carries game_min_cosine, the cosine the LP direction achieves.
3. Response-model fidelity. The design's 0.010 bound on the model's error among the exactly
   verified candidates is exceeded once, 0.0138 in round one at seed 123 (seed 2024: 0.0079).
   Exact verification decided the applied candidate in every round, as registered.

Harness corrections on branch www27, commits 1ecca87 to f3bdf47 (21 September 2026). None
alters a recorded result; each has a regression test.

- The no-train-split fallback (ArguAna) keeps its sorted halves in order; before, the
  example order depended on the Python hash seed. Membership is unchanged. New ArguAna runs
  therefore do not reproduce the historical example order bit for bit.
- The fallback is used only when qrels/train.tsv is absent; any other loading error stops
  the run.
- validate_e0 requires each round's broadcast to be the previous round's global and the first
  broadcast to be the recorded initial adapter state. E1 uniform seed 123, E1 n_k seed 123
  and A2 AFL seed 2024 were re-validated under this check on 21 September 2026 and pass.
- The data fingerprint covers the held-out dev queries; provenance hashes response_arm.py
  and response_aggregation.py; the driver refuses to overwrite an existing result file.
- --loss_batch_size names the batch of the q-FFL/AFL loss estimate; its default is the
  historical --eval_batch_size, which the CLI had described as speed-only.
- The result field BWT is renamed round1_to_final_drift; it was never a backward-transfer
  statistic over experiences.
- run_e3.sh exits nonzero on failure.

The prospective temporal protocol will be registered in a section of its own before any
run of it.


### 14 Block T1: the temporal pilot on a constructed MS-Shift stream (registered 2026-09-21 16:36 UTC, the clock of the commit that adds it; before any run of the continual driver)

Design: docs/continual-pipeline-design.md at this commit. Code: experiences.py, memory.py,
regression.py, continual_driver.py, validate_continual.py (commits fccc848 to 8652bee).
Every run is validated by validate_continual.py before it counts; test-query values are
not read before every run of the block has validated.

Construction, all fixed here. Clients 0 to 4 are MS-Shift topic clusters 0 to 4. Within a
client, TRAIN queries with judgements are partitioned into four experiences by k-means
(k = 4, ten restarts, seed 1, rerun with the next seed at most five times if a sub-cluster
falls below the training-plus-guard count) on TF-IDF unigram and bigram features (min_df 2,
sublinear tf) reduced by truncated SVD to 50 dimensions and unit-normalised; the fit uses
training-side text only, and the official EVAL queries of the topic are assigned to the
frozen centroids. Splits per (client, experience), sampled with the manifest seed and
disjoint: 1,674 training, 167 guard and 350 test queries, the largest uniform counts every
cell of the feasibility table supports (smallest cells: 1,841 training-eligible and 350
evaluation-eligible queries). Test queries come only from the official EVAL queries. A
retained unit is one query id with all its judgements.

Schedules. A: client 0 (0,1,2,3), 1 (2,0,3,1), 2 (1,3,0,2), 3 (3,2,1,0), 4 (0,3,1,2).
B: client 0 (3,1,0,2), 1 (1,3,2,0), 2 (2,0,1,3), 3 (0,2,3,1), 4 (1,0,2,3).

Corpus per client, fixed across experiences: every passage judged relevant for any selected
query or any official EVAL query of the topic, plus the top-10 BM25 passages (bm25s, English
stopwords, over the full 8.84 M collection) of every selected training, guard and test
query, plus passages sampled uniformly to 60,000 in all.

Calibration stream: topic 5 ("other"), which has no official EVAL queries, split by k-means
(k = 2) into clients 50 and 51; its evaluation pool is a seeded 15 percent of its training
queries held out before clustering; counts 2,000 / 200 / 500; order client 50 (0,1,2,3),
client 51 (2,0,3,1). Recipe grid on it, arm D (uniform FedAvg with replay), seed 123, budget
256: rounds per experience in {2, 4, 8} by learning rate in {2e-5, 5e-5}; the pair with the
highest mean acquisition A is frozen. Then lambda in {0.5, 1.0, 2.0} under arm E at that
pair: the value with the lowest G among those whose A is within 0.005 of the best A. The
frozen recipe is written into 14.1 before the pilot starts. Fixed regardless: Contriever
(facebook/contriever, unsupervised), LoRA rank 16, alpha 32, dropout 0.1, both factors
trainable, batch 32, one local epoch per round, in-batch negatives, mixed precision,
gradient checkpointing, evaluation batch 256.

Pilot. Arms: (A) frozen; (B) local-only continual, one model per client, same budget and
steps; (C) uniform FedAvg without replay; (D) uniform FedAvg with replay; (E) D with
distillation from the previous experience's acquisition reference on replay rows. Seeds
123, 2024 and 3407. Both schedules. 24 trained runs and two frozen evaluations, all on one
L4 with the frozen recipe.

Measurements. After each experience, per-query nDCG@10 and recall@100 on the guard and test
queries of every experience learned so far, against the client's fixed corpus; the
acquisition reference of (client, experience) is the shared model (the client's own model
under B) at the end of that experience. Primary retention: the positive-part regression
against the reference, per query, averaged over the split. Also reported: signed backward
transfer, peak forgetting, absolute nDCG@10, the full client-by-experience matrix, the
worst cell, the fraction of historical cells at or above 0.010, and each client's whole
evaluation pool at the end.

Gate, on test queries, evaluated once every run has validated. A = mean over (client,
experience) of acquisition against the frozen backbone; G = mean over historical cells
(u < v) of the positive-part regression. Thresholds 0.020 for A and 0.010 for G are
practical effect sizes. A criterion passes if the mean of its scalar over the six runs
(three seeds by two schedules) clears the threshold and the scalar clears half the
threshold in at least five of the six. G2 (adaptation is useful): A under D. G1 (a problem
remains): G under D. G3 (collaboration has value, reported, not a go criterion): A under D
at least A under B. Arm E is reported beside D: if E's G is below half of D's G while E's A
is within 0.005 of D's, distillation already controls the regression and the paper says so.

Outcomes. G2 fails: the recipe or the stream is at fault; fix, re-register and rerun. G1
fails with G2 passing: replay controls the regression on this stream; no method is
motivated by it, and the temporal evidence moves to LoTTE or to that finding. G1 and G2
pass: method arms are designed and registered separately.

To be appended as 14.1 before the pilot: the manifest digests (schedules A and B and the
calibration stream) with the commit that built them, the measured cost of one unit on the
L4 and the extrapolated pilot cost, and the frozen recipe with the calibration outcomes.

Amendment to 14, before any run of the continual driver (2026-09-22 18:32 UTC, the clock of
the commit that adds it). The first manifest build, at commit 793ce22 on the L4, stopped
because client 0 had 79,169 relevant and top-10 BM25 passages, more than the 60,000-passage
pool. The construction therefore uses the top-5 BM25 passages per selected query instead of
the top-10; everything else in 14 is unchanged. No training run had started.

Second amendment to 14, before any run of the continual driver (2026-09-22 20:57 UTC, the clock
of the commit that adds it). The manifest build at commit 89ad9f1 completed its retrieval and
was then refused by its own verifier with "client 0: splits overlap". The cause is a property
of the release: every MS-Shift EVAL query id is also in MS MARCO's queries.train.tsv (6,595 of
6,595 for topic 0, and all of topics 1 to 4), because MS-Shift draws its evaluation queries
from the training pool rather than from the official dev set. The builder had therefore placed
some ids in an experience's training or guard split and the same ids in another experience's
test split. No training run had started, and no such manifest was written.

Correction: a topic's official EVAL query ids are removed from the training-side pool before
the clustering, so they can only ever be test queries. The registered rule for the counts is
unchanged; re-deriving it from the corrected feasibility table (smallest cells 1,489
training-eligible and 350 evaluation-eligible queries) gives 1,354 training, 135 guard and
350 test queries per client-experience, replacing 1,674 / 167 / 350 in section 14. The
calibration stream is unaffected at 2,000 / 200 / 500; its evaluation pool is a held-out
fifteen percent of its own training queries, which the builder already removed from training.
Everything else in 14 and its first amendment stands.

Third amendment to 14, before any run of the continual driver (2026-09-22 23:44 UTC, the clock
of the commit that adds it). The two primary manifests (schedules A and B) were built and
verified at commit e8f98d2. The calibration manifest was then refused by the corpus rule:
client 50 needed 92,653 relevant and hard passages against a 60,000-passage pool. The cause
is the size of that stream's evaluation pool. MS-Shift's "other" cluster has 234,037 training
queries and no official evaluation queries, so a fifteen percent held-out pool is tens of
thousands of queries, and every pool passage must be in the client's fixed corpus.

Correction, in the calibration stream only: the held-out pool is capped at 3,000 queries per
client, near the size of an official pool, and the calibration split counts are those of the
primary stream (1,354 / 135 / 350) rather than 2,000 / 200 / 500, so the recipe is chosen on
a stream shaped like the pilot. Neither change touches the official-evaluation path or the
two primary manifests, which stand as built. No training run had started.

Correction to the third amendment, before any run (2026-09-22 23:55 UTC). The cap of 3,000 was
too small in the other direction: it left the calibration stream's smallest cell with 171
evaluation queries, fewer than the 350 a test split needs, and the build refused it. The cap
is a property of the topic, not of a client, and the four candidate values were evaluated on
the real counts before rerunning: 3,000 gives a smallest cell of 118, 6,000 gives 247, 8,000
gives 342 and 12,000 gives 501. The cap is 12,000, the smallest of these that leaves every
cell above its test count; the corpus then has an upper bound of about 50,600 passages
against the 60,000 limit. The primary stream does not use this path and its manifests stand.

### 14.1 Manifests, measured cost and the frozen recipe (written 2026-09-22 08:51 UTC, the clock of
the commit that adds it; stage 1 complete, before any pilot run)

Manifests, built on the L4 at commit e8f98d2 (primary) and c83162e (calibration), each
verified by experiences.verify_manifest before it was written. Local copy:
research_workspace/results_vm/T1_20260921/manifests.

    primary_A.json        9bd8a61b7f08c30542de6cca58999b021df8b133f72065f7fa38d54ce451c5b1
    primary_B.json        989b666ad274c1325c20c4a584cbff637d6df869f172b78db8e9647b86862a9d
    calibration_cal.json  03e1d9ff15600531fed0ed445d116e6acc02ae77f4b7a488b2d54f60f9d52827

Counts 1,354 / 135 / 350 in all three. Each client's corpus is 60,000 passages: for the five
primary clients about 13,000 relevant, 31,000 BM25-hard and 15,000 random; for the two
calibration clients about 18,800, 32,700 and 8,400. Schedules A and B share every split and
corpus and differ only in each client's order, as designed. Evaluation pools are the whole
official MS-Shift set per topic (5,868 to 6,595) for the primary clients and 9,254 and 2,746
held-out training queries for the calibration clients.

Measured cost. The profile run (calibration manifest, two clients, four experiences, one
round each) took 2,105 s and validated four rounds. Evaluation dominates: about 480,000
passage encodes at roughly 240 per second. Calibration wall times, two clients: 2,248 and
2,243 s at two rounds, 2,540 and 2,539 s at four, 3,139 and 3,125 s at eight, and 3,573,
3,569 and 3,571 s for the three arm-E runs. Extrapolated to five clients at eight rounds,
about 7,700 s per pilot run, so about 54 hours for the 24 pilot runs and the two frozen
evaluations on one L4. This replaces the guess of one L4 day in section 6 of the design.

Calibration outcome, arm D, seed 123, acquisition A and regression G:

    rounds  lr      A        G
    2       2e-5    0.0552   0.01920
    2       5e-5    0.0719   0.02847
    4       2e-5    0.0688   0.02632
    4       5e-5    0.0841   0.02640
    8       2e-5    0.0825   0.02697
    8       5e-5    0.0939   0.03788

The registered rule, highest acquisition, selects eight rounds per experience at learning
rate 5e-5. Acquisition was still rising at both edges of the grid, so this is a boundary
choice rather than an interior optimum; the grid was fixed before the runs and is not
widened after seeing them.

Arm E at that recipe, lambda and its A and G: 0.5 gives 0.09109 and 0.01851; 1.0 gives
0.09144 and 0.01855; 2.0 gives 0.09073 and 0.01830. The registered rule, lowest G among
those within 0.005 of the best A, selects lambda 2.0.

Frozen recipe for the pilot: eight rounds per experience, learning rate 5e-5, lambda 2.0,
with the fixed settings of section 14.

Observation to carry into the report, not a gate outcome. On this calibration stream the
condition stated in section 14 for arm E is met: E's regression 0.01830 is below half of D's
0.03788, while E's acquisition 0.09073 is within 0.005 of D's 0.09393. Reference
distillation therefore halves the regression at no measurable cost in acquisition here. This
is one seed on the two-client calibration stream and is not the gate, which is evaluated on
the primary stream over three seeds and two schedules; but if it recurs there, the framing
required by section 14 applies and the paper must position any new mechanism against
distillation rather than against replay alone.

Correction to 14.1 and specification of the distillation arm, before the pilot
(2026-09-22 17:16 UTC, the clock of the commit that adds it).

The sentence in 14.1 reading "halves the regression at no measurable cost in acquisition" is
withdrawn as an overstatement. The accurate statement of the same numbers: on the
single-seed, two-client calibration stream, reference distillation reduces mean positive-part
regression from 0.03788 to 0.01830, a reduction of 51.7 percent, while acquisition falls from
0.09393 to 0.09073, a loss of 0.0032, about 3.4 percent of the replay arm's acquisition gain.
No statistical test was performed and none is claimed; the word "significant" is not used of
this result. The residual regression of 0.01830 is still 1.83 times the 0.010 practical
threshold, so distillation reduces the regression substantially and does not eliminate it.
Whether a further mechanism has a useful role is an open question the pilot exists to
inform, not a question these two numbers settle.

Specification of arm E, so the arm is reproducible from the record. Teacher: the acquisition
reference of the immediately preceding experience, that is the shared model at the end of
experience u - 1, fixed for the whole of experience u and not refreshed within it; at the
first experience there is no teacher and the arm trains exactly as arm D. Scope: the
distillation term applies only to replay rows, that is the retained queries of earlier
experiences, never to rows of the current experience. Loss: for each training batch, scores
are 20.0 times the cosine between the unit-normalised query and passage embeddings of that
batch, the same scale and similarity the contrastive loss uses; the term is lambda times the
mean squared difference between the student's and the teacher's scores over the replay rows
of the batch, taken over the batch's own passages as candidates. There is no temperature and
no separate candidate pool. At lambda zero the loss is exactly the contrastive loss, which is
asserted against the library implementation in tests/test_continual_driver.py, so arms D and E
differ only by the added term. The teacher's extra forward pass and its stored state are
recorded as arm E's additional cost and are not charged against the retained-query budget,
because the teacher is a model, not retained queries.

Retained-query budget, restated because it governs any later method. One budget of 256 query
ids per client covers replay and any guard queries a method consults online. The pilot's arms
consult no guard queries, so their whole budget is replay; validate_continual refuses a run
whose replay holds a guard or test id, whose reserved set holds a test id, or whose training
ids are anything other than the current experience's training queries together with the
replay. Guard queries cannot become a second, unaccounted memory.

Reporting. Because regression is measured against each arm's own acquisition reference, an
arm can show less regression by acquiring less. Every result table therefore reports, beside
acquisition and positive-part regression, the absolute nDCG@10 on the earlier experiences'
test queries, the reference's score on the same queries, the per-client and per-experience
cells, the worst cell and the fraction of cells at or above 0.010.

Launch conditions. The pilot runs only from an approved commit: run_continual.sh refuses
unless the repository is at EXPECT_COMMIT with a clean working tree and the three manifest
digests match their recorded SHA256SUMS. Each run records its driver exit status, and a run
interrupted part way is never silently resumed. The frozen recipe, the run list, the
manifests and the source commit are fixed before the first run and are not changed in
response to early scores; a pause is warranted by an implementation or data fault, not by a
disappointing seed.

Deviation during the pilot (recorded 2026-09-22 21:43 UTC). Section 14 states that test-query
values are not read before every run of the block has validated. After the first run,
pilot-frozen-A-s123, validated at 19:36 UTC, its per-client nDCG@10 on the evaluation pools
(0.6508, 0.5921, 0.5523, 0.4488, 0.4842 for clients 0 to 4) was read and reported. These are
the untrained backbone's scores; the recipe had been frozen in 14.1 and no run, setting or
decision was changed in response. The rule is otherwise kept: until all 26 runs validate,
only wall time, exit status and validation state are read.

Measured cost so far, replacing the projections: pilot-frozen-A-s123 8,248 s, and
pilot-local-A-s123 7,510 s with 32 rounds validated. The campaign is projected from these
two measurements at about 57 hours; the distillation arm's own cost is not yet measured.

### 14.2 Pilot outcome (written 2026-09-25 06:08 UTC, the clock of the commit that adds it; after all 26 runs of block T1 validated)

Completion. The chain ran on the L4 from commit fda2e25 between 2026-09-22 17:18 and
2026-09-25 03:40 UTC and wrote DONE. All 26 runs exited 0 and passed validate_continual.
Wall time per run: frozen 8,247 and 8,248 s; local 7,510 to 7,559 s; fedavg 7,529 to
7,559 s; fedavg-replay 7,912 to 8,000 s; fedavg-replay-distill 9,109 to 9,159 s; 209,563 s
(58.2 hours) in all.

Gate. t1_gate.py at fda2e25 was run on the L4 and again, on a second machine, from the copied
records; the two outputs are identical.

| criterion | threshold | mean over the six runs | runs at half / at full threshold | result |
|---|---:|---:|---:|---|
| G2, A under D | 0.020 | 0.1075 | 6 / 6 | passes |
| G1, G under D | 0.010 | 0.0233 | 6 / 6 | passes |
| G3, A under D at least A under B (reported) | | 0.1075 against 0.1079 | | not met |

Registered outcome: G1 and G2 pass, so method arms are designed and registered separately.
Arm E does not meet its reported criterion: its G, 0.0140, is not below half of D's
(0.0116), and its A, 0.0932, is not within 0.005 of D's 0.1075. The calibration observation
in 14.1 (one seed, two clients: G lower by 51.7 percent at an A lower by 0.0032) did not
carry over to the primary streams, where E's G is 40 percent lower than D's and its A
0.0143 lower. This concerns the calibration-selected lambda of 2.0 only.

Required reporting, test queries, mean over runs (arm A has one seed per schedule). Cells
are (client, experience, evaluation position) with the evaluation after the experience was
learned; a worst cell is the largest over all runs of the arm.

| arm | A | G | BWT | reference nDCG@10 | nDCG@10 on earlier experiences at the end | cells at or above 0.010, per run | worst cell |
|---|---:|---:|---:|---:|---:|---:|---|
| A frozen | 0 | 0 | 0 | 0.4057 | 0.4044 | 0 of 30 | none |
| B local | 0.1079 | 0.0565 | -0.0112 | 0.5116 | 0.4978 | 30 of 30 | 0.1104, client 0, experience 3 |
| C fedavg | 0.1049 | 0.0244 | 0.0119 | 0.5020 | 0.5171 | 30 of 30 | 0.0388, client 4, experience 0 |
| D fedavg-replay | 0.1075 | 0.0233 | 0.0157 | 0.5032 | 0.5231 | 30 of 30 | 0.0470, client 4, experience 0 |
| E fedavg-replay-distill | 0.0932 | 0.0140 | 0.0043 | 0.4945 | 0.4986 | 18 to 24 of 30 | 0.0325, client 4, experience 0 |

The per-run values, the full client-by-experience matrix, peak forgetting, guard-query
values and each client's evaluation pool are in the report that t1_report.py writes from the
records. t1_report.py recomputes A and G from the per-query records without regression.py,
continual_driver.summarise or t1_gate.py. For all 26 runs they agree with the stored values
within 1e-9, and its checks find no problem: the frozen arm's scores never change, each
acquisition reference equals the scores recorded at its own position, the query set of every
cell is fixed over time, and in every test cell the signed change equals the mean improvement
minus the mean positive-part regression.

Observations outside the registered decision, recorded as development evidence for the
method design and not as tested claims. Paired by schedule and seed:
- D against B: A lower by 0.0005, G lower by 0.0332 (lower in 6 of 6 runs), and nDCG@10 on
  earlier experiences at the end higher by 0.0253.
- D against C: A higher by 0.0026 and G lower by 0.0012 (lower in 4 of 6 runs).
- E against D: G lower by 0.0092 (6 of 6), and nDCG@10 on earlier experiences at the end
  lower by 0.0245; E's acquisition references are also lower (0.4945 against 0.5032).
- Under C and D the mean signed change on earlier experiences is positive (BWT 0.0119 and
  0.0157), while 15.2 and 14.6 percent of (query, evaluation) pairs lose at least 0.010.
- Regression grows with the age of an experience in 19 of the 20 (trained arm, client) rows;
  the exception is client 3 under C (0.0301 at age 2, 0.0288 at age 3).
- Guard and test queries give similar A and G. Scores on the retained replay queries were
  not recorded, so retained and unseen queries cannot be compared from these records.

Because the pilot now informs the method design, a later method claim treats these streams
as development data; its confirmation uses query families or collections not used for the
design (LoTTE, LongEval) under its own registration.

Records. gs://fedcrag-t1-archive/T1_20260921, a private bucket with public access prevention
enforced (Standard class, asia-northeast1), holds the block's directory as written on the L4:
1,198 files, 73.4 GB, all 26 runs' checkpoint chains, the calibration and profile runs, the
manifests, the chain log and a SHA256SUMS that lists every other file. The size and CRC32C of
every object match the files on the L4's disk. pilot-fedavg-replay-A-s123 was restored from
the bucket into a separate directory: its 38 files match their SHA-256, validate_continual
passes on the restored copy against the restored manifest, and re-scoring its final checkpoint
on client 0 reproduces all 1,940 recorded guard and test scores of the four experiences
exactly when the scoring process has first run one training call, as the driver's process had.
Scored in a fresh process, the same state gives identical scores in two separate runs, but 2
of the 1,940 differ from the record (one guard and one test query of experience 3, by 0.044
and 0.031). Scoring therefore depends slightly on whether the process has trained; the
training call leaves TF32 off and the float32 matmul precision at "highest", and the cause is
not identified. The same check on the untrained backbone: before any training it reproduces
the recorded frozen scores exactly; after one training call, 5 of the 1,940 scores differ and
no split's mean moves by more than 0.0004. The frozen scores were computed before training and
the trained states' scores after it, so acquisition against the frozen backbone carries this
effect; on client 0 it is 0.00003 in the mean over the four test splits, against a threshold
of 0.020. G, backward transfer and the trained states' absolute scores are each computed
within one condition.

Scoring precision (recorded 2026-09-25 06:45 UTC, the clock of the commit that adds it).
The difference left unexplained in 14.2 has a cause. With mixed precision on CUDA,
sentence-transformers' training call leaves the model's forward wrapped in fp16 autocast, and
the wrapper stays after training. In block T1 every trained state was therefore scored under
fp16 autocast, and the untrained backbone, scored at the start of each run before any
training, in fp32. In a fresh process, scoring the restored final state of
pilot-fedavg-replay-A-s123 under explicit fp16 autocast reproduces all 1,940 recorded scores
on client 0, and the wrapper is present on the model after one training call and absent before
it. From the commit that adds this note, continual_driver scores every state, the untrained
backbone included, under explicit fp16 autocast. With that code and no training in the
process, the final state again reproduces all 1,940 recorded scores, and the untrained
backbone gives the five changed scores that 14.2 measured after a training call. T1's
trained-state scores, G, backward transfer and absolute scores are unaffected; its A carries
the difference measured in 14.2.

### 15 Development study on the T1 streams (registered 2026-09-25 06:49 UTC, the clock of the commit that adds it; before any of its runs)

Status. Development. These runs characterise the trade-off between acquisition and
regression on the streams read in 14.2, and fix, by the rules below, three settings of the
LoTTE confirmation block, which is registered separately before any LoTTE run. No result here
is a confirmatory claim.

Fixed. Manifest primary_A (schedule A, digest 9bd8a61b), seed 123, and the recipe of 14.1
except where a run below changes it: learning rate 5e-5, LoRA rank 16, batch 32, one local
epoch per round, a retained-query budget of 256, scoring under explicit fp16 autocast. The
runs start from the commit that implements arm F with its tests passing; 15.1 records that
commit and the digests before the first run.

Runs, each validated by validate_continual:
1. D-r4: arm D, 4 rounds per experience.
2. D-r2: arm D, 2 rounds per experience.
3. E-0.25: arm E, lambda 0.25, 8 rounds per experience.
4. E-0.1: arm E, lambda 0.1, 8 rounds per experience.
5. F: arm F below, 8 rounds per experience.
T1's pilot-fedavg-replay-A-s123 (arm D, 8 rounds) and pilot-fedavg-replay-distill-A-s123
(arm E, lambda 2.0) are the anchors.

Arm F, fedavg-replay-accept: arm D with a reference-based acceptance check on the server's
update, inside the same budget. From the second experience on, each client's 256 retained
queries are 64 guard queries and 192 replay queries. The guard queries are drawn from the
guard splits of the client's earlier experiences by the equal-share, seeded rule that draws
replay, and are redrawn at the start of each experience; replay fills the other 192 slots as
in arm D. Check pool, per client and experience: the relevant passages of its guard queries,
the top-5 BM25 passages of each guard query, and 1,000 passages drawn from the client's corpus
with a fixed seed. The top-5 passages come from the saved BM25 index of the manifest build and
the same query texts; the build cached only the union of each client's hits, so they are
computed once into a side file whose digest every F run records. A guard query's score is its
nDCG@10 ranked within the pool. At the start of each experience every guard query is scored
within the pool under the acquisition reference of the experience it belongs to. In every
round, with g the model broadcast at the start of the round and c the uniform average of the
clients' trained states, the server tries the step sizes 1, 1/2 and 1/4 in that order on
g + s(c - g) and keeps the first whose mean positive-part regression over all clients' guard
queries (reference score minus candidate score, floored at zero) is at most 0.010; if none
qualifies, g is kept for the round. Each round records the steps tried, their regressions and
the step kept. validate_continual recomputes the kept model from the persisted states and
checks the recorded choice against the rule. Guard scores use the same fp16 autocast as every
other score.

Rules fixed now for the LoTTE block:
- Arm D keeps 8 rounds per experience. D-r4 and D-r2 only characterise the trade-off.
- Arm E uses the lambda among 0.1, 0.25 and 2.0 with the highest mean nDCG@10 on the guard
  queries of the first three experiences after the last experience; a tie goes to the smaller
  lambda.
- Arm F enters the LoTTE block only if it is implemented, tested and its run here has
  validated by 2026-09-30 23:59 UTC; otherwise it is left to the method study.

Reported for every run: A, G, backward transfer, the absolute nDCG@10 on earlier experiences
beside the reference nDCG@10, the per-client and per-experience cells, and for arm F the step
kept in every round. The (A, G) points of arm D at 2, 4 and 8 rounds and of arm E at each
lambda are shown together.

Cost at T1's measured run times: at most 7,955 s for each D run, about 9,137 s for each E run,
and for F the time of a D run plus its pool checks, at most about 0.6 million passage encodes.

### 15.1 Launch record for the development study (written 2026-09-25 07:07 UTC, the clock of the commit that adds it; before its first run)

Code. The runs start from the commit that adds this section. It contains arm F as committed in
f544a9a (550 tests passing on the L4) and the LoTTE builder of 05c7c10, which these runs do
not use; for an MS-Shift manifest the driver behaves as at f544a9a.

Inputs. Manifest primary_A.json, sha256
9bd8a61b7f08c30542de6cca58999b021df8b133f72065f7fa38d54ce451c5b1, the file the pilot used.
Guard hits guard_hits_primary_A.json, sha256
9510b7414b745675ccb977068c87dff61d42464b6a741cf64631be67a9c2d9d6: the top-5 BM25 passages of
all 2,700 guard queries (540 per client), retrieved with the saved index of the manifest build
in 8 min 50 s; every one of the 13,500 passages lies in its client's corpus.

Launch. On the L4, in this order, output in ~/D15_20260925: D-r4, D-r2, E-0.25, E-0.1, then F,
each validated before the next starts. The launcher refuses unless the repository is at this
commit with a clean tree and the manifest digests match; the machine powers off when the chain
ends. Until all five runs validate, only wall time, exit status and validation state are read.

### 16 Block L1: confirmation on LoTTE (registered 2026-09-25 07:10 UTC, the clock of the commit that adds it; before the LoTTE manifests are built and before any LoTTE run)

Purpose. Test, on a collection not used for any design decision, the observations that 14.2
recorded as development evidence on MS-Shift. Nothing in this block is tuned on LoTTE: the
recipe, the arms and their settings come from 14.1, from 15 and from the rules of 15.

Data. LoTTE (the ColBERTv2 release, lotte.tar.gz, 3,576,167,599 bytes, sha256
37c0f39af23a6e3464f63395a4d04a22b91fe59c1aa64ea1773a8aff113c7ab5). Clients 0 to 4 are its
topics lifestyle, recreation, science, technology and writing. A client's experience 0 is the
topic's dev split and experience 1 its test split; the two come from different StackExchange
forums and share no passage. Each experience pools the split's search and forum queries (5,156
to 5,571 per topic in all).

Construction (lotte.py as committed in 05c7c10, seed 1). Per client and experience, sampled
and disjoint: 1,949 training, 194 guard and 350 test queries, T1's rule applied to the
smallest experience (lifestyle dev, 2,493 queries). A training query is judged by its
highest-voted answer only, a tie going to the smaller passage number, because forum queries
have up to 292 answers where MS MARCO has about one; guard and test queries are judged by all
their answers. Corpus per client, fixed across experiences: every answer passage of every
selected query, the top-5 BM25 passages (bm25s, English stopwords, over the topic's two
collections) of every selected query, and passages drawn uniformly from those collections to
60,000 in all. If any client's answer and hard passages exceed 60,000, the hard passages are
the top-3 for every client; the bound computed from the query files is 61,282 for technology
and at most 57,767 for the others. Schedules: in A, clients 0, 2 and 4 learn experience 0 then
1, and clients 1 and 3 the reverse; in B every client is reversed.

Arms, recipe and seeds. B local, C fedavg, D fedavg-replay, E fedavg-replay-distill with the
lambda chosen by the rule of 15, and F fedavg-replay-accept if and only if the condition of 15
is met. Eight rounds per experience, learning rate 5e-5 and every other setting of 14.1, a
retained-query budget of 256, scoring under explicit fp16 autocast. Seeds 123, 2024 and 3407
under both schedules: 24 runs without F, 30 with it. The untrained backbone is scored at the
start of every run, so there is no separate frozen run.

Hypotheses, each over the six runs of an arm (three seeds by two schedules), on test queries.
A consistency criterion passes if the mean over the six runs clears the threshold and at least
five of the six runs clear half of it, as in 14.

- H1, useful acquisition: A under D is at least 0.020 (consistency criterion).
- H2, regression remains under replay: G under D is at least 0.010 (consistency criterion).
- H3, a positive average hides per-query losses: under D the mean backward transfer over the
  six runs is above zero, and at least 10 percent of the (query, evaluation) pairs of the
  earlier experience, pooled over the six runs, lose at least 0.010.
- H4, shared training retains more than local training: G under D is lower than under B in at
  least five of the six runs paired by schedule and seed, and the mean nDCG@10 on the earlier
  experience at the end of the stream is higher under D than under B.

Reported without a pass or fail rule: E and F against D and C against D (paired differences in
A, G and end nDCG@10 on the earlier experience), the per-client and per-experience cells, the
worst cell, the reference nDCG@10, and the six per-run values of every quantity.

Outcomes. Each hypothesis is reported as confirmed or not, and the paper states which of the
observations in 14.2 replicate on LoTTE and which do not, with their numbers. No LoTTE result
changes a setting, and no LoTTE run is repeated because of its scores. With two experiences
each client has one historical cell, so the growth of regression with an experience's age
stays a development observation from T1.

Launch conditions as in 14 and 15.1. To be appended as 16.1 before the first LoTTE run: the
manifest digests and the hard-passage depth the build used, the lambda and the F decision
under 15, a profile run's measured cost and the launch commit. LoTTE test scores are read only
after every run of the block has validated.

Deviation in block T1 (recorded 2026-09-25 07:16 UTC, the clock of the commit that adds it).
Section 14 defines arm B as local-only continual training with the same memory budget and the
same number of optimiser steps. continual_driver gave retained queries only to the replay
arms, so T1's arm B kept none: on client 0 in the last round it trained on 1,437 examples in
44 steps, as plain FedAvg (C) did, against 1,703 examples and 53 steps under replay (D). The
gate (G1 and G2) uses arm D only and is unaffected. G3, reported only, compared D with a local
arm without retained queries. The observation in 14.2 of D against B mixes federation with
replay; the contrast without that confound is C against B, neither with retained queries: G
lower by 0.0320 in 6 of 6 paired runs, nDCG@10 on earlier experiences at the end higher by
0.0193, and A lower by 0.0031.

Amendment to 16, before any LoTTE run (recorded 2026-09-25 07:16 UTC, the clock of the commit
that adds it). Arm B is local-only continual training with the retained-query budget of 256
drawn as in D (driver arm local-replay, added in the previous commit), as 14 defines it. Arm
B0, local-only training without retained queries (driver arm local, as T1's B ran), is added
so that T1's contrast without the replay confound is repeated. The block has 30 runs without F
and 36 with it. H4 stands as written, with this B. A fifth hypothesis is added:

- H5, federation against local training, neither with retained queries: G under C is lower
  than under B0 in at least five of the six runs paired by schedule and seed, and the mean
  nDCG@10 on the earlier experience at the end of the stream is higher under C than under B0.
