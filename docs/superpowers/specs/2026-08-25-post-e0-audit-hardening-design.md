# Post-E0 Audit Hardening Design

**Status:** Approved for implementation on 2026-08-25

**Evidence baseline:** `fedspan-e0-cured` at `7325bf56381c24c6a4af013688bdd417c95d7d7d`

**Implementation branch:** `fedspan-post-e0-audit`

## Purpose

Strengthen the E0 evidence gate without changing the completed models or
silently rewriting their records. The repaired validator must independently
certify the method-defining FedSpan direction and every round-to-round state
transition. Future campaigns must record valid per-round timings. The completed
E0 campaign keeps its valid total row runtimes while its unrecoverable
per-round timing breakdown is explicitly marked unavailable.

## Evidence boundary and claim strength

- Commit `7325bf5` remains the recorded execution identity for all eleven E0
  rows. The closeout will resolve that recorded 12-character identifier to the
  unique full Git object
  `7325bf56381c24c6a4af013688bdd417c95d7d7d`.
- The repair is developed on a new branch and is not used to relabel the
  implementation that generated the artifacts.
- Existing result JSON, state, manifest, log, and resource files are read-only
  inputs to post-hoc validation.
- No independently signed pre-audit digest is currently known. Therefore the
  strongest warranted historical claim is **post-hoc internally consistent
  with the recorded clean execution commit**, not independently proven
  immutable since execution. The newly preserved inventory will become the
  canonical tamper-evident baseline and its digest will be anchored outside
  the artifact directory.
- A GPU rerun is not required when an artifact passes the stronger scientific
  checks.
- If a scientific check fails, only the affected row or rows are eligible for
  rerun after the failure is diagnosed. Invalid legacy per-round timing alone
  never triggers a model rerun.

## Approaches considered

### 1. Strengthen the existing validator — selected

Add independent direction and continuity checks to `validate_e0.py`, introduce
a strict future timing schema, and surface legacy timing as unavailable. This
keeps one authoritative validation entry point and reuses the existing manifest
and aggregate checks.

### 2. Add a separate post-hoc auditor — rejected

This would preserve the old validator but create two competing meanings of
`VALIDATED`, duplicate state reconstruction, and invite drift between tools.

### 3. Make timing strict and rerun all of E0 — rejected

This has a simple rule but spends GPU time without changing model states or
retrieval metrics. The missing timing data cannot invalidate otherwise correct
scientific state chains.

## Architecture

### A. Independent FedSpan direction audit

`validate_e0.py` will reconstruct the method decision from persisted state
files, result diagnostics, and the declared method contract. It must not import
or call the production FedSpan solver.

For each successful `normmaxmin` round, the validator will:

1. Reconstruct every client's PEFT-scale-aware effective raw-B update from the
   saved broadcast and client tensors. Derive each row scale directly from the
   persisted shared A tensor and derive the PEFT factor from the frozen E0
   execution contract; do not trust the diagnostic's `module_scales` as the
   reconstruction input. Cross-check the derived scales against both the
   method contract and round diagnostic.
   Require the declared LoRA rank to match every persisted A/B tensor pair and
   bind row-scale semantics to tensors: unit means `c≈1`; numeric means
   `c≈declared`; peft-init requires consistent recorded mode/measurement while
   disclosing that the discarded pre-orthogonalization tensor prevents a
   stronger origin proof.
2. Bind the recorded step policy, direction policy, absolute/relative activity
   tolerances, mixture-norm tolerance, coefficient-cap setting, and LoRA rank
   to their frozen-manifest command values. Canonically parse and bind every
   other execution-relevant argument, reject duplicate or unknown flags,
   verify row metadata and archived data fingerprints, and recompute the
   recorded run-configuration hash.
3. Recompute finite norms, activity thresholds, active indices, normalized
   client directions, and the active cosine Gram.
4. Compare the recomputed active set and Gram with the persisted diagnostic.
5. Recompute the declared direction objective independently:
   - `minnorm`: enumerate all nonempty simplex faces for the at-most-four-client
     E0 problem, solve a rank-aware augmented KKT system on each face, retain
     feasible candidates with small KKT residual, and select the smallest
     `q*=w^T C w` value, including singular cancellation faces;
   - `maxmin-lp`: solve the explicit validation LP directly from the recomputed
     Gram, without calling the production aggregation module.
6. Check the recorded simplex weights against the independently optimal value
   and verify simplex feasibility and objective tolerances.
7. Keep the quantities distinct: minnorm optimizes `q*=w^T C w` and records
   `min_norm_value=sqrt(max(q*,0))`; maxmin optimizes
   `t*=min(Cw)` and records it as `solver_objective_gamma`; the applied
   normalized direction achieves `min(Cw)/sqrt(w^T C w)`; and shortfall is
   `sqrt(max(q*,0))-achieved`. Recompute each independently, together with the
   resolved step norm and raw-delta coefficients.
8. Require inactive coefficients to be exactly zero and active coefficients to
   satisfy

   `v_i = resolved_step_norm * w_i / (client_norm_i * mixture_norm)`.

9. Use tensor-derived scales for scientific geometry and certify the recorded
   scales numerically. Because a float32 persisted A can round the pre-cast
   scale that produced legacy byte hashes, replay legacy solved/applied hashes
   with only the now-certified recorded scales, then compare the recorded-scale
   and derived-scale effective vectors numerically. Keep certificates separate:
   replay production diagnostics on the recorded-scale Gram; run the
   independent oracle on the tensor-derived Gram. Propagate their measured Gram
   perturbation into scientific objective/direction bounds, and fail closed as
   boundary-indeterminate when that uncertainty overlaps the cancellation
   threshold. Replay the production coefficient formula only with certified
   recorded-scale norms/mixture; assess the actual persisted step separately in
   tensor-derived scientific geometry rather than demanding derived-scale
   coefficients that production never computed.
10. Independently reproduce success versus every deterministic fallback branch
    (activity, step legality, cancellation, coefficient cap, and
    reconstruction conditions). If the audit oracle succeeds, refuse a
    recorded runtime solver error/failure/invalid fallback; a fabricated zero
    fallback is not accepted merely because its hashes agree.

Float32 stored-A orthogonality uses the established
`1e-6 * max(1, c^2)` allowance. Float64 Gram, simplex, objective, and
coefficient-formula checks use explicit tight absolute-plus-relative
tolerances. The existing `5e-6 * max(1, step_norm)` allowance applies only to
materialized effective vectors/state application, not to coefficient values.

The check must reject repaired-hash mutations that negate all FedSpan delta
coefficients, substitute feasible-but-suboptimal minnorm or maxmin weights, or
fabricate a zero fallback. It must accept an alternative weight vector on a
genuinely non-unique optimal face.

### B. Exact round continuity

During validation, retain the preceding round's global state and hash. For every
boundary `t -> t+1`:

- require `round_(t+1).broadcast_state_sha256` to equal
  `round_t.global_state_sha256`;
- compare tensor key sets, shapes, dtypes, and values exactly as defense in
  depth;
- report the boundary and first mismatching key on failure.

Also bind the round-1 broadcast hash to
`method_contract.initial_adapter_state_sha256`.

The check must reject a coherent translation mutation that changes the same
later-round B tensor in broadcast, every client, and global state, repairs all
local hashes, and leaves deltas/aggregation internally consistent.

### C. Timing schema and runner behavior

Resource parsing/writing will move from the embedded shell snippet into a small
Python module with a testable interface. New records use
`fedcrag-e0-resources/2`.

For future runs:

- launch the training interpreter unbuffered;
- keep the timestamping interpreter unbuffered and flush each emitted line;
- emit machine-readable start and completion markers for every round;
- carry the launcher row ID into markers through an explicit telemetry-only
  environment value;
- require exactly one ordered start/end pair and one positive duration for
  every completed round, without duplicates, crossed pairs, or denominator
  drift;
- timestamp every boundary with wall-clock and monotonic nanoseconds, compute
  all durations only from the monotonic clock, record every inter-round gap,
  and require the exact integer partition
  `pre_ns + sum(round_ns) + sum(between_round_ns) + post_ns = elapsed_ns`;
  wall time is used only for human-readable UTC provenance;
- capture and require success from training, timestamp filtering, and `tee`;
- bind the record to the expected run ID and ordered UTC start/finish values;
- during validation, reparse the raw timestamped log and GPU-sample file,
  compare their duration/sample/peak values with the JSON, and verify raw-file
  hashes;
- fail validation when a schema-v2 timing record violates these rules.

For every completed E0 schema-v1 record:

- preserve the original file unchanged;
- validate total `elapsed_seconds`, GPU sampling, and the remaining resource
  fields as before;
- return `round_timing_valid: false`,
  `round_timing_status: legacy-buffered-unavailable`, and no publishable
  per-round timing values regardless of whether the buffered values happen to
  look plausible;
- do not fail scientific validation solely because of this known telemetry
  defect.

No timing values will be inferred, redistributed, or fabricated.

### D. Documentation

Update `README.md` so that:

- the canonical FedSpan command includes
  `--frozen_a_row_scale peft-init`;
- both row-scale modes are described as explicit choices with no implicit safe
  default;
- E0 is consistently described as an eleven-row campaign;
- E0 is described as correctness/attribution evidence, not a paper-scale
  efficacy result;
- legacy E0 per-round timings are explicitly unavailable while total row
  runtimes remain usable.

Update the executable CLI help at the same time so it does not describe
`unit` as a default when frozen-A paper-grade execution requires an explicit
row-scale choice.

## Test-first implementation

Every production change follows a witnessed red-green cycle.

1. Add repaired-hash negated-direction, feasible-suboptimal-direction for both
   policies, alternative non-unique optimum, and fabricated-fallback tests;
   confirm the invalid mutations pass the old validator unexpectedly.
2. Implement only the independent direction checks required to distinguish
   all of them correctly.
3. Add a coherent repaired-hash broken-continuity mutation and a repaired
   initial-state mutation; confirm they pass the old validator unexpectedly.
4. Implement exact hash and tensor continuity checks.
5. Add an actual unbuffered pipeline subprocess test, explicit round start/end
   marker tests, raw-evidence mutation tests, legacy-schema tests, and
   schema-v2 strict failure tests before extracting the timing module and
   changing the runner.
6. Add README/manifest assertions before correcting documentation.
7. Run focused tests after every repair and the full suite after all repairs.
8. Re-run both mutations against the final validator as adversarial regression
   evidence.

Tests will exercise real persisted state payloads through the existing mocked
driver harness. Production solver calls are forbidden in the new independent
direction oracle.

## Post-hoc E0 validation and preservation

After local verification:

1. Start `thesis-fedcrag` only long enough to access the persistent disk.
2. Confirm `/home/turjo/FedCRAG_E0_RESULTS/COMPLETE.json` names all eleven rows
   and commit `7325bf56381c`.
3. Run the post-E0 validator from the new clean branch against the frozen
   manifest and every existing row directory.
4. Produce one machine-readable summary and one Markdown closeout containing
   pass/fail status, direction-oracle residuals, continuity counts, aggregate
   tolerances, valid total runtimes, and the explicit legacy timing limitation.
5. Export original artifacts, manifests, logs, validation reports, and SHA-256
   checksums to:

   `/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/`

6. Snapshot a sorted relative-path inventory of regular files, refuse links or
   special files, hash remotely before and after copying, and verify local
   names, sizes, and hashes against it. Preserve the validator commit as a Git
   bundle. Keep separate source-equality and complete-package checksum
   manifests, and anchor the complete-package digest outside the artifact
   tree.
7. Install an unconditional shutdown trap immediately after any VM start. Stop
   and verify the VM even when validation, export, or checksum comparison
   fails. Do not promote a staged preservation directory to canonical status
   until `TERMINATED` is observed.

No failed scientific check will be bypassed. A failure pauses publication
promotion and triggers diagnosis before any rerun decision.

## Acceptance criteria

- The original 200-test baseline remains green.
- New direction and continuity mutation tests fail before their fixes and pass
  after their fixes.
- The final validator refuses every invalid repaired-hash direction,
  fabricated-fallback, initial-state, and continuity mutation while accepting
  alternate genuinely optimal weights.
- Singular cancellation and non-unique optimum cases are classified correctly;
  feasible-but-suboptimal decisions and fabricated fallbacks are refused.
- A clean schema-v2 timing fixture and the real launcher/filter pipeline pass;
  zero, duplicated, missing, crossed, raw-evidence-mismatched, pipeline-failed,
  or irreconcilable schema-v2 timings fail.
- Legacy E0 timing records remain scientifically validatable but explicitly
  return unavailable per-round timing status.
- README commands match the actual CLI and eleven-row launcher.
- All eleven E0 directories pass the strengthened scientific validator, or the
  exact failing rows and invariants are reported without rerunning anything.
- Exported artifacts match remote SHA-256 checksums.
- The VM is confirmed terminated after preservation.

## Non-goals

- Changing training, aggregation, model weights, datasets, metrics, or E0
  hyperparameters.
- Recovering E0 per-round timings that were never captured correctly.
- Improving E0 retrieval scores.
- Launching E1-E5 before the E0 post-hoc gate is resolved.
- Refactoring unrelated W3 campaign changes or modifying the dirty
  `w3-campaign` checkout.
