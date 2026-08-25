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

## Evidence boundary

- Commit `7325bf5` remains the immutable execution identity for all eleven E0
  rows.
- The repair is developed on a new branch and is not used to relabel the
  implementation that generated the artifacts.
- Existing result JSON, state, manifest, log, and resource files are read-only
  inputs to post-hoc validation.
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
2. Bind the recorded step policy, direction policy, absolute/relative activity
   tolerances, mixture-norm tolerance, coefficient-cap setting, and LoRA rank
   to their frozen-manifest command values.
3. Recompute finite norms, activity thresholds, active indices, normalized
   client directions, and the active cosine Gram.
4. Compare the recomputed active set and Gram with the persisted diagnostic.
5. Recompute the declared direction objective independently:
   - `minnorm`: enumerate all nonempty simplex faces for the at-most-four-client
     E0 problem, solve the equality-constrained quadratic on each face, retain
     feasible candidates, and select the smallest `w^T C w` value;
   - `maxmin-lp`: solve the explicit validation LP directly from the recomputed
     Gram, without calling the production aggregation module.
6. Check the recorded simplex weights against the independently optimal value
   and verify simplex feasibility and objective tolerances.
7. Recompute mixture norm, minimum achieved cosine, min-norm reference value,
   direction shortfall, resolved step norm, and raw-delta coefficients.
8. Require inactive coefficients to be exactly zero and active coefficients to
   satisfy

   `v_i = resolved_step_norm * w_i / (client_norm_i * mixture_norm)`.

9. Recompute solved/applied effective-step hashes and retain the existing
   aggregate/state checks.

Simplex feasibility, Gram/scalar agreement, and independent objective agreement
use an absolute tolerance of `1e-8 * max(1, |reference|)`. Applied-state and
coefficient reconstruction retain the existing `5e-6 * max(1, step_norm)`
float32 materialization allowance.

The check must reject the demonstrated repaired-hash mutation that negates all
FedSpan delta coefficients and materializes the opposite global step.

### B. Exact round continuity

During validation, retain the preceding round's global state and hash. For every
boundary `t -> t+1`:

- require `round_(t+1).broadcast_state_sha256` to equal
  `round_t.global_state_sha256`;
- compare tensor key sets, shapes, dtypes, and values exactly as defense in
  depth;
- report the boundary and first mismatching key on failure.

The check must reject the demonstrated mutation that changes a later broadcast
and repairs its local hash while leaving the round internally consistent.

### C. Timing schema and runner behavior

Resource parsing/writing will move from the embedded shell snippet into a small
Python module with a testable interface. New records use
`fedcrag-e0-resources/2`.

For future runs:

- launch the training interpreter unbuffered;
- keep the timestamping interpreter unbuffered and flush each emitted line;
- require exactly one positive duration for every completed round;
- require ordered round markers without duplicates;
- require `0.5 * elapsed_seconds <= sum(round_elapsed_seconds) <=
  elapsed_seconds + 5`; the lower allowance covers model/data initialization
  and the frozen evaluation before round 1, while the upper allowance covers
  timestamp rounding;
- fail validation when a schema-v2 timing record violates these rules.

For completed E0 schema-v1 records:

- preserve the original file unchanged;
- validate total `elapsed_seconds`, GPU sampling, and the remaining resource
  fields as before;
- return `round_timing_valid: false`,
  `round_timing_status: legacy-buffered-unavailable`, and no publishable
  per-round timing values when a zero/implausible breakdown is detected;
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

## Test-first implementation

Every production change follows a witnessed red-green cycle.

1. Add a repaired-hash negated-direction mutation test and confirm it passes the
   old validator unexpectedly.
2. Implement only the independent direction checks required to make it fail.
3. Add a repaired-hash broken-continuity mutation test and confirm it passes the
   old validator unexpectedly.
4. Implement exact hash and tensor continuity checks.
5. Add delayed-marker resource tests, legacy-schema tests, and schema-v2 strict
   failure tests before extracting the timing module and changing the runner.
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

6. Verify local checksums against the remote originals.
7. Stop the VM after the export and verification complete.

No failed scientific check will be bypassed. A failure pauses publication
promotion and triggers diagnosis before any rerun decision.

## Acceptance criteria

- The original 200-test baseline remains green.
- New direction and continuity mutation tests fail before their fixes and pass
  after their fixes.
- The final validator refuses both adversarial mutations.
- A clean schema-v2 timing fixture passes; zero, duplicated, missing, or
  irreconcilable schema-v2 timings fail.
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
