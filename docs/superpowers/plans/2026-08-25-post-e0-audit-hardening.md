# Post-E0 Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended for adversarial review) or `executing-plans` to execute this plan task-by-task. Keep the tasks sequential; do not combine commits or skip witnessed RED/GREEN gates.

**Goal:** Independently certify the completed eleven-row E0 campaign's FedSpan decisions and round state chain, make future per-round resource timing trustworthy, correct the operator documentation, and preserve the strengthened validation evidence without changing or relabeling the original E0 artifacts.

**Architecture:** Keep `validate_e0.py` as the single evidence gate, but move the small exact simplex-game solvers into an audit-only module that imports no production aggregation code. Reconstruct all method-defining geometry from persisted tensors plus the frozen execution contract, then compare it with the recorded decision. Add exact cross-round state continuity. Extract resource-log parsing into a schema-aware Python module used by both the launcher and validator. Treat schema-v1 E0 round timings as unavailable telemetry while retaining their valid total runtimes.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, Bash, JSON, Git, GCP CLI.

## Global constraints

- Work only in `/Users/turjo/Desktop/FedCRAG/worktrees/fedspan-post-e0-audit` on branch `fedspan-post-e0-audit`; do not modify the dirty `w3-campaign` checkout.
- Preserve commit `7325bf56381c24c6a4af013688bdd417c95d7d7d` as the recorded execution identity for every E0 artifact; do not imply an unavailable pre-audit tamper-proof guarantee.
- Never edit the completed E0 result, state, manifest, resource, or log files in place.
- The audit oracle must not import or call `aggregation_schemes.fedspan_delta_weights`, `_min_norm_simplex_weights`, or `maxmin_weights`.
- Each production repair begins with a test that is observed failing for the intended reason.
- A scientific validation failure pauses the closeout. Timing-schema-v1 limitations alone do not trigger a model rerun.
- Do not invent legacy per-round durations. Publish them as unavailable and retain only the measured total row runtime.
- Save review findings and closeout reports beneath `/Users/turjo/Desktop/FedCRAG/review_outputs/` or `/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/`.

---

### Task 0: Freeze the executable plan and baseline

**Files:**
- Commit: `docs/superpowers/specs/2026-08-25-post-e0-audit-hardening-design.md`
- Commit: `docs/superpowers/plans/2026-08-25-post-e0-audit-hardening.md`

- [ ] **Step 1: Record the implementation range and commit the reviewed documents**

```bash
git rev-parse 7325bf5^{commit}
git add docs/superpowers/specs/2026-08-25-post-e0-audit-hardening-design.md \
  docs/superpowers/plans/2026-08-25-post-e0-audit-hardening.md
git commit -m "docs: plan post-E0 audit hardening"
git status --porcelain
```

Expected: the first command resolves to
`7325bf56381c24c6a4af013688bdd417c95d7d7d`; the final command prints nothing.
Use `7325bf5..HEAD` as the immutable review range throughout execution.

---

### Task 1: Create the independent direction-game oracle

**Files:**
- Create: `e0_direction_oracle.py`
- Create: `tests/test_e0_direction_oracle.py`
- Reference only: `tests/reference_solvers.py`

**Interfaces:**
- `min_norm_simplex_oracle(gram) -> {weights, objective, simplex_residual, constraint_violation}`
- `maximin_simplex_oracle(gram) -> {weights, objective, simplex_residual, constraint_violation}`
- Neither function reads diagnostics or production solver output.

- [ ] **Step 1: Write independent-oracle unit tests**

Cover identity, fully aligned, two-client asymmetric, the nontrivial three-client harness Gram, and invalid/non-symmetric/nonfinite matrices. Add hard-coded analytic singular cases, especially the antipodal Gram `[[1,-1],[-1,1]]` whose optimum is `w=(0.5,0.5), q*=0`; do not use `tests.reference_solvers` as the expected value for singular cases because its inverse-based implementation skips singular faces. For nonsingular valid cases, compare objective values with the existing reference functions and check feasibility rather than demanding a unique weight vector:

```python
result = min_norm_simplex_oracle(gram)
expected_value, _ = min_norm_reference(gram)
assert result["objective"] == pytest.approx(expected_value, abs=1e-10)
assert np.sum(result["weights"]) == pytest.approx(1.0, abs=1e-10)
assert np.min(result["weights"]) >= -1e-10
```

For maximin, assert `result["objective"] == min(gram @ weights)` and compare with `maximin_reference`.

- [ ] **Step 2: Run the focused tests and witness RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_direction_oracle.py -q -p no:cacheprovider
```

Expected: collection fails because `e0_direction_oracle.py` does not exist.

- [ ] **Step 3: Implement exact audit-only enumeration**

Validate a square, finite, symmetric matrix. Enumerate all nonempty supports for min-norm. For each face solve the augmented KKT system

```text
[ C_SS  1 ] [w] = [0]
[ 1^T   0 ] [λ]   [1]
```

with a rank-aware solve/least-squares path and explicit residual check; enumerate all subfaces so boundary optima remain candidates. Discard infeasible candidates and select the minimum `w @ C @ w`. Enumerate the vertices of the explicit maximin polytope `{(w,t): 1^T w=1, w>=0, Cw>=t1}` and select the largest feasible `t`.

Use float64 and deterministic lexicographic tie-breaking. Return residuals computed from the selected result. Keep the module imports limited to `itertools`, `math`, and `numpy`.

- [ ] **Step 4: Run oracle tests GREEN and prove production-solver independence**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_direction_oracle.py -q -p no:cacheprovider
rg -n "aggregation_schemes|fedspan_delta_weights|_min_norm_simplex_weights|maxmin_weights" \
  e0_direction_oracle.py
```

Expected: tests pass; `rg` returns no matches.

- [ ] **Step 5: Commit the oracle unit**

```bash
git add e0_direction_oracle.py tests/test_e0_direction_oracle.py
git commit -m "test: add independent E0 direction oracle"
```

---

### Task 2: Reject repaired-hash FedSpan direction mutations

**Files:**
- Modify: `tests/driver_harness.py`
- Modify: `tests/test_validate_e0.py`
- Modify: `validate_e0.py`
- Use: `e0_direction_oracle.py`

**Interfaces:**
- Add private validator helpers for deriving frozen-A geometry scales, recomputing activity/Gram/optimality, and verifying coefficient construction.
- Extend the successful `normmaxmin` round summary with compact direction residuals for closeout reporting.

- [ ] **Step 1: Make the mocked launch contract explicit**

Extend `tests.driver_harness.build_argv` so every harness run records `--lora_rank 16`; extend `normmaxmin` rows with the frozen E0 values:

```python
"--fedspan_active_abs_tol", "1e-12",
"--fedspan_active_rel_tol", "1e-8",
"--fedspan_mixture_norm_tol", "1e-6",
```

Keep the coefficient-cap flag absent, matching the E0 manifest. Make the
persisted harness tensors rank-consistent: use a 16-row A and a B whose second
dimension is 16, embedding the existing small client directions in the leading
coordinates. Set every scale record's `row_scale_mode` to the actual launched
mode rather than the generic label `constant`.

- [ ] **Step 2: Add the repaired-hash opposite-direction mutation test**

Run a genuine one-round frozen-A `normmaxmin` fixture. Negate every recorded/proposed coefficient and the applied B-step, then independently repair:

- the global tensor state and its state hash;
- `application.applied_state_sha256`;
- the solved and applied effective-step hashes;
- the recorded delta-coefficient arrays.

Do not change the client states, Gram, simplex weights, objective, or contract. Assert:

```python
with pytest.raises(E0ValidationError, match="direction|coefficient"):
    validate_run_directory(tmp_path)
```

The mutation helper in the test must calculate hashes directly and must not call the new validator helper.

Add stronger repaired-hash mutations for both direction policies: choose a
feasible but suboptimal simplex vector, recompute its positive coefficient
formula, mixture/gamma/achieved/shortfall/solver diagnostics, materialized
global state, and every affected hash. Require refusal specifically for
objective suboptimality. Add a non-unique optimal-face fixture whose alternate
optimal weights must be accepted, proving the validator compares feasibility
and objective rather than one preferred weight vector.

Before production changes, also add parameterized repaired-diagnostic/status
mutations for every fallback branch plus a two-round mutation that rewrites one
genuine success as a zero fallback while another round remains nonzero. The
invalid cases expect refusal; the alternate non-unique optimum is a passing
characterization test before and after hardening.

- [ ] **Step 3: Witness RED against the old validator**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider \
  -k "opposite_direction or negated_direction or suboptimal or fabricated_fallback or fallback_status"
```

Expected: each invalid mutation test FAILS because no `E0ValidationError` is
raised; the alternate non-unique optimum remains a passing baseline.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider -k nonunique
```

Expected: PASS before and after hardening.

- [ ] **Step 4: Bind every method-defining manifest field**

Canonically parse the complete manifest argv with the driver's actual CLI
semantics. Reject duplicate and unknown flags. Compare all execution-relevant
arguments, including model, slices, metrics, seed, rounds, local epochs, LoRA
rank/mode, train/eval batch sizes, learning rate, max steps, data root,
`--save_states`, `--no_grad_ckpt`, weighting fields, frozen-A row scale, and:

```text
lora_rank
fedspan_step_policy
fedspan_step_norm
fedspan_direction_policy
fedspan_active_abs_tol
fedspan_active_rel_tol
fedspan_mixture_norm_tol
fedspan_max_abs_delta_weight
```

Represent an omitted cap as `None`. Add parameterized drift tests that mutate each value independently and assert a field-specific refusal.

Also require row `coordinate`, `arm`, `regime`, and `max_steps` metadata to
agree with argv/result values and recompute `run_configuration_sha256` using
the same canonical payload. Add `--data-root` to campaign validation: resolve
the manifest's exact `--data_root`, require that archived root to exist, load
the four slices without network access, recompute the canonical section/key
JSON fingerprints used by `_data_fingerprints`, and compare them with both
result provenance and the configuration-hash input. If the archived data root
is unavailable, fail the independent dataset-content gate and report only a
recorded-fingerprint cross-binding—not content verification. Tests use a tiny
archived data root and cover duplicate flags, an unknown flag, row metadata
drift, configuration-hash drift, data-content drift, and missing data.

Bind the declared LoRA rank to every persisted module pair:
`A.shape[0] == B.shape[1] == lora_rank`, with compatible outer dimensions in
broadcast, every client, and global state. Add a repaired-hash shape/rank
mismatch test.

- [ ] **Step 5: Derive scales from saved A tensors, not the diagnostic**

For each frozen-A module, compute:

```python
gram_a = A.double() @ A.double().T
c_squared = float(torch.diagonal(gram_a).mean())
row_scale_c = math.sqrt(c_squared)
geometry_scale = 2.0 * row_scale_c
```

Require off-diagonal values near zero and every diagonal near `c_squared` using
the established float32 storage allowance
`1e-6 * max(1, c_squared)`. The factor `2.0` is bound to the frozen E0 contract
`lora_alpha = 2 * lora_rank`; require manifest/result `lora_rank == 16` and
cross-check `frozen_a_row_scale_records` plus diagnostic `module_scales`
against the derived values with tight float64 absolute-plus-relative scalar
tolerances.

Use the tensor-derived scales for all scientific geometry. Legacy exact hashes
were produced with the pre-float32-cast recorded scale, so replay the byte-exact
hashes using only the now numerically certified recorded scale. Separately
compare the recorded-scale and derived-scale effective vectors/norms within the
explicit vector tolerance in Step 6.

Construct both `C_recorded` and `C_derived` and record
`delta_gram = max(abs(C_recorded-C_derived))`. Replay byte hashes and all
production FW diagnostics on `C_recorded`, the exact geometry production
solved. Run the independent scientific oracle only on `C_derived`.

Bind row-scale semantics, not only labels. Every scale record's
`row_scale_mode` must equal the manifest mode. `unit` requires derived `c≈1`;
a numeric declared mode requires `c≈declared`; `peft-init` requires mode
consistency and `measured_init_row_rms≈row_scale_c≈derived c`. Document that
the discarded pre-orthogonalization tensor prevents stronger post-hoc proof of
the peft-init origin. Add a unit-labeled/non-unit-A repaired-record mutation.

- [ ] **Step 6: Reconstruct and verify the direction decision**

From the persisted broadcast/client tensors and derived scales:

1. recompute each client block and norm;
2. recompute `threshold = max(abs_tol, rel_tol * largest_finite_norm)`;
3. recompute active mask/indices and normalized active directions;
4. recompute the cosine Gram;
5. call the audit-only min-norm and maximin oracles;
6. verify recorded simplex feasibility and the declared policy's independently optimal objective;
7. keep and test the four quantities separately: minnorm compares
   `w^T C w` to `q*`; recorded `min_norm_value` compares to
   `sqrt(max(q*,0))`; maxmin compares `min(Cw)` and
   `solver_objective_gamma` to `t*`; achieved cosine is
   `min(Cw)/sqrt(w^T C w)` and shortfall is
   `sqrt(max(q*,0))-achieved`;
8. recompute the median-active resolved step;
9. require inactive coefficients to be exactly zero and replay active
   production coefficients with certified recorded-scale client norms and
   `mixture_norm_recorded`:

```python
expected = resolved_step_norm * weight / (client_norm * mixture_norm)
```

Do not apply the tight coefficient tolerance to a counterfactual
tensor-derived coefficient formula that production never used. Instead,
reconstruct the actual persisted applied vector under tensor-derived scales and
evaluate its norm/direction with the perturbation-aware scientific bounds
below.

Use these explicit tolerances:

```text
scalar/scale/Gram/objective/coefficient: abs(error) <= 1e-10 + 1e-8*abs(reference)
simplex sum/nonnegativity:               1e-10
KKT residual:                            1e-10*max(1, n, ||K||inf*||x||inf, ||rhs||inf)
recorded-vs-derived vector:              ||a-b||2 <= 1e-10 + 1e-8*max(||a||2,||b||2)
stored-A orthogonality only:             1e-6*max(1,c_squared)
```

Reserve `5e-6 * max(1, resolved_step_norm)` solely for float32 materialized
effective-vector/application checks. Compare the recorded solution by
feasibility and objective, not weight identity, because optimum weights can be
non-unique. Include a Gram fixture where `q*`, `sqrt(q*)`, `t*`, and the
normalized achieved cosine are all numerically distinct.

For a declared `minnorm` success, replay the Frank-Wolfe certificate
`gap = w^T C_recorded w - min(C_recorded w)` from the recorded weights,
require the recorded gap to match at the tight recorded-geometry tolerance,
require `converged == (gap <= recorded_tol)`, and bind the recorded solver
tolerance to the frozen production value `1e-14`.

On `C_derived`, allow simplex quadratic/payoff optimality error no larger than
`2*delta_gram + 1e-10`. For normalized direction, set
`direction_uncertainty = (4*delta_gram + 1e-10) /
max(mixture_norm_derived, mixture_norm_tol)`. Require minnorm achieved cosine
and shortfall to agree with the derived optimum within
`direction_uncertainty + 1e-8`. If
`mixture_norm_derived <= mixture_norm_tol + sqrt(delta_gram + 1e-10)`, refuse
the round as boundary-indeterminate instead of claiming an exact
success/fallback classification. Apply the corresponding `2*delta_gram`
objective allowance to maxmin.

Add a multi-module peft-init fixture with slightly different certified
recorded and tensor-derived scales and a small-but-non-boundary mixture that
must pass both the tight recorded coefficient replay and the
perturbation-aware derived-vector audit. Also add a repaired,
self-consistent near-balanced antipodal successful-step mutation whose
`q-q*` passes the generic squared-objective tolerance but whose achieved cosine
is `-1`; recorded and derived geometry coincide there, so its zero perturbation
allowance must still force refusal by the normalized-optimality/certificate
checks.

- [ ] **Step 7: Independently certify fallback branches**

For every `normmaxmin` round, not only successful rounds, enumerate every
production status explicitly: `no_active`, `invalid_step_norm`, `singleton`,
`optimal`, `solver_error`, `solver_failure`, `solver_invalid`,
`near_cancellation`, `coefficient_limit`, and `reconstruction_failure`.
Recompute deterministic activity, step legality, singleton/success,
cancellation, cap, and reconstruction branches. If the independent audit
oracle finds a feasible optimum, refuse a historical `solver_error`,
`solver_failure`, or `solver_invalid` fallback rather than treating it as the
expected method decision. Require the recorded success/fallback/status and
diagnostic fields to match. Make every Step 2 invalid mutation turn GREEN only
after these checks are implemented.

- [ ] **Step 8: Run focused direction tests GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py tests/test_e0_direction_oracle.py \
  -q -p no:cacheprovider -k "direction or manifest or scale or coefficient or fallback"
```

Expected: all focused tests pass, including the repaired-hash mutation refusal.

- [ ] **Step 9: Commit the direction audit unit**

```bash
git add validate_e0.py tests/driver_harness.py tests/test_validate_e0.py
git commit -m "fix: independently validate E0 FedSpan directions"
```

---

### Task 3: Enforce exact round-to-round broadcast continuity

**Files:**
- Modify: `tests/test_validate_e0.py`
- Modify: `validate_e0.py`

- [ ] **Step 1: Add a repaired local-hash continuity mutation**

After Task 2 is committed but before any continuity production change, add a
test that builds a valid two-round fixture. Apply the same constant translation to one
round-2 B tensor in the broadcast, every client, and global state so all local
deltas and aggregation remain unchanged. Independently repair the broadcast,
client, global, application, solved-step, and applied-step hashes affected by
the translation. The test expects an `E0ValidationError` naming the
`round_1 -> round_2` boundary; in Step 2 it fails specifically because the
current validator raises no exception.

Add a separate repaired-hash round-1 replacement whose local relationships are
coherent but whose broadcast hash no longer matches
`method_contract.initial_adapter_state_sha256`.

- [ ] **Step 2: Witness RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider \
  -k "continuity or initial_boundary"
```

Expected: both tests FAIL because the old validator accepts each locally
consistent mutation.

- [ ] **Step 3: Add exact hash and tensor-chain checks**

First require round 1's actual broadcast hash to equal
`method_contract.initial_adapter_state_sha256`. Retain the preceding round's
global state/hash in `validate_run_directory`. Before validating round `t+1`,
require:

```python
current["broadcast_state_sha256"] == previous["global_state_sha256"]
```

Then compare key sets, tensor shapes, dtypes, and `torch.equal` values. Raise with the boundary and first mismatching key. Return `initial_boundary_checked=True` and `continuity_boundaries_checked = max(num_rounds - 1, 0)` in the validation summary.

- [ ] **Step 4: Run continuity and full validator tests GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider
```

- [ ] **Step 5: Commit the continuity unit**

```bash
git add validate_e0.py tests/test_validate_e0.py
git commit -m "fix: enforce exact E0 round continuity"
```

---

### Task 4: Introduce schema-v2 timing and honest legacy handling

**Files:**
- Create: `e0_resources.py`
- Create: `tests/test_e0_resources.py`
- Modify: `tests/test_fedspan_driver.py`
- Modify: `tests/test_validate_e0.py`
- Modify: `tests/test_e0_manifest.py`
- Modify: `federated_forgetting.py`
- Modify: `validate_e0.py`
- Modify: `run_e0.sh`

**Interfaces:**
- `parse_timestamped_rounds(log_path, expected_run_id, num_rounds, started_mono_ns, finished_mono_ns) -> {pre_ns, round_ns, between_round_ns, post_ns}`
- `build_resource_record(...) -> dict`
- `validate_resource_record(record, expected_run_id, num_rounds, log_path, samples_path) -> normalized timing/resource summary`
- Exact writer CLI:

```text
e0_resources.py write --run-id ID --run-dir DIR --log PATH --samples PATH \
  --started-wall-ns INT --finished-wall-ns INT \
  --started-mono-ns INT --finished-mono-ns INT --num-rounds INT
```

- Timestamp filter CLI: `e0_resources.py timestamp`, reading stdin and writing
  `wall_time_ns<TAB>monotonic_ns<TAB>line` to stdout with every line flushed.
- Clock CLI: `e0_resources.py clock`, returning wall and monotonic nanoseconds
  from one interpreter invocation for launcher start/finish boundaries.
- The launcher transports its row identity through telemetry-only environment
  variable `FEDCRAG_E0_RUN_ID`; the driver refuses malformed IDs when set and
  embeds it in every marker. It does not infer identity from an output path.

- [ ] **Step 1: Write timing parser and schema-policy tests**

Before any production edit, add tests requiring the driver to emit
machine-readable `E0_ROUND_START run_id N/M` and
`E0_ROUND_END run_id N/M` markers around each complete round body when
`FEDCRAG_E0_RUN_ID` is set, and to reject missing/malformed telemetry identity
in E0 launcher mode. Add launcher `timing-selftest` behavioral expectations
and synthetic timestamped-log tests. Cover:

- ordered start/end pairs produce strictly positive durations plus explicit
  pre-round and post-round overhead;
- missing, duplicate, crossed, wrong-denominator, wrong-run-ID, out-of-order,
  and zero-duration pairs fail schema v2;
- integer monotonic identity
  `pre_ns + sum(round_ns) + sum(between_round_ns) + post_ns ==
  finished_mono_ns-started_mono_ns` holds exactly and all markers lie within
  the boundaries; include deliberate delays between round pairs;
- forward/backward simulated wall-clock jumps do not change durations, which
  are computed exclusively from `time.monotonic_ns()`; wall nanoseconds are
  retained only for human-readable UTC provenance;
- every schema v1 record—even one with plausible positive reconciled
  values—returns `round_timing_valid=False`,
  `round_timing_status="legacy-buffered-unavailable"`, and
  `round_elapsed_seconds=None`;
- schema v1 still rejects malformed totals, round-count truncation, and invalid GPU fields;
- schema v2 returns `round_timing_valid=True` and the measured durations;
- swapping another row's resource JSON, mutating only JSON timing/GPU values,
  or mutating raw log/sample evidence is refused;
- build/validation/replace failure leaves an existing destination unchanged
  and cleans its temporary sibling.

- [ ] **Step 2: Witness each independent RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_fedspan_driver.py -q -p no:cacheprovider -k e0_marker
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_manifest.py -q -p no:cacheprovider -k timing
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_resources.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider -k resource
```

Expected: each command independently fails for its intended missing
marker/pipeline/module/schema behavior; a missing-module collection error is
not allowed to mask the driver or launcher RED evidence.

- [ ] **Step 3: Implement `e0_resources.py`**

Move timestamp parsing and resource-record construction out of the embedded
shell heredoc. Use schema `fedcrag-e0-resources/2`, reject ambiguous marker
streams, preserve existing determinism/GPU metadata, record run ID, wall and
monotonic nanosecond boundaries, pre/round/post monotonic durations, raw
log/sample SHA-256 hashes and every between-round monotonic gap,
sample count, and peak. Validate the complete record before replacement; use a
same-directory temporary, flush and `fsync`, call `os.replace`, and clean up in
`finally`. The writer exits nonzero with a concise stderr error and never
prints a success claim on failure.

Implement the driver markers only after the tests are RED. Read
`FEDCRAG_E0_RUN_ID`, validate its restricted run-ID syntax, and flush both
start and end markers. For schema v1, structural omissions still fail, but schema identity is
decisive: never publish any per-round v1 values, even when they look plausible.

- [ ] **Step 4: Make the launcher unbuffered and use the module**

In `run_e0.sh`:

```bash
export PYTHONUNBUFFERED=1
```

Run the timestamping interpreter with `-u`, flush each emitted line, and
replace the resource heredoc with the exact `e0_resources.py write` CLI above.
Obtain both boundary clock pairs through `e0_resources.py clock`; never use
integer `date +%s` for duration arithmetic.
Capture `PIPESTATUS` immediately after the training/filter/tee pipeline and
require all three components to exit zero.

Factor that exact pipeline into a shell function and expose a zero-training
`timing-selftest` subcommand. A hermetic test supplies a tiny producer that
prints start/end markers around short sleeps and exercises the actual
launcher/filter/tee path. Assert timestamps arrive progressively and yield
positive round durations. Force the producer, filter, and tee stages to fail
one at a time and require nonzero launcher status.

- [ ] **Step 5: Add launcher contract assertions**

In `tests/test_e0_manifest.py`, assert the launcher contains
`PYTHONUNBUFFERED=1`, uses the extracted module, and no longer embeds
`fedcrag-e0-resources/1` writing logic. Execute `timing-selftest`; do not rely
only on source-string checks.

- [ ] **Step 6: Wire validator output to timing status**

Replace `_validate_resource_record`'s single-schema logic with the module API,
passing the expected run ID and the actual raw log/sample paths. Reparse raw
schema-v2 evidence and require recorded durations, sample count, peak, and raw
hashes to match. Parse/compare ordered timestamps and require
the exact monotonic partition identity; wall-clock ordering is descriptive and
must not determine durations. Include in the returned
validation summary:

```json
{
  "elapsed_seconds": 123.0,
  "round_timing_valid": false,
  "round_timing_status": "legacy-buffered-unavailable",
  "round_elapsed_seconds": null
}
```

Never turn unavailable values into zeros or estimates.

- [ ] **Step 7: Run timing, launcher, and validator tests GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_resources.py tests/test_fedspan_driver.py \
  tests/test_e0_manifest.py tests/test_validate_e0.py \
  -q -p no:cacheprovider -k "resource or timing or e0_marker"
bash -n run_e0.sh
bash run_e0.sh timing-selftest
```

- [ ] **Step 8: Commit the timing unit**

```bash
git add e0_resources.py federated_forgetting.py validate_e0.py run_e0.sh \
  tests/test_e0_resources.py tests/test_fedspan_driver.py \
  tests/test_e0_manifest.py tests/test_validate_e0.py
git commit -m "fix: make E0 resource timing auditable"
```

---

### Task 5: Correct README and campaign semantics

**Files:**
- Modify: `README.md`
- Modify: `federated_forgetting.py`
- Modify: `tests/test_e0_manifest.py`

- [ ] **Step 1: Add documentation contract tests**

Parse the canonical corrected-FedSpan fenced command and compare its complete
method-defining flag/value set against a frozen normmaxmin manifest row:
`--lora_rank 16`, explicit `--frozen_a_row_scale peft-init`, step/direction
policies, three FedSpan tolerances, and intentional absence of a coefficient
cap. Assert the E0 section says eleven rows; no README or parser-help text calls
`unit` an implicit/default behavior; and the README states that legacy E0
per-round timing is unavailable while total row runtime is usable.
Replace the stale blanket statement that no E0 run exists with an accurate
status: the external eleven-row correctness campaign completed, strengthened
post-hoc validation is pending, and no paper-scale efficacy claim follows.

- [ ] **Step 2: Witness RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_manifest.py -q -p no:cacheprovider -k "readme or help"
```

- [ ] **Step 3: Update the operator documentation**

Make the canonical command fully explicit, describe `unit` and `peft-init` as
separate declared choices with no safe implicit default, replace ten-row
wording with eleven-row wording, and state that E0 is
correctness/attribution evidence rather than paper-scale efficacy evidence.
Add the legacy timing limitation without weakening total-runtime provenance.
Correct `federated_forgetting.py --help` at the same time.

- [ ] **Step 4: Run documentation tests GREEN and inspect the diff**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_manifest.py -q -p no:cacheprovider
git diff --check
git diff -- README.md federated_forgetting.py tests/test_e0_manifest.py
```

- [ ] **Step 5: Commit the documentation unit**

```bash
git add README.md federated_forgetting.py tests/test_e0_manifest.py
git commit -m "docs: correct the E0 execution contract"
```

---

### Task 6: Full local verification and same-rigor adversarial re-review

**Files:**
- Modify only if findings require a cure: files from Tasks 1-5
- Create: `/Users/turjo/Desktop/FedCRAG/review_outputs/2026-08-25_POST_E0_AUDIT_HARDENING_LOCAL_REVIEW.md`

- [ ] **Step 1: Run the complete local gate**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
bash -n run_e0.sh
bash run_e0.sh manifest > /tmp/fedcrag-e0-manifest.txt
git diff --check 7325bf5..HEAD
git diff --stat 7325bf5..HEAD
git status --porcelain
```

Expected: all tests pass; exactly eleven manifest rows; the implementation
range is whitespace-clean; and the worktree is empty.

- [ ] **Step 2: Re-run the complete adversarial mutation set explicitly**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_validate_e0.py -q -p no:cacheprovider \
  -k "opposite_direction or negated_direction or suboptimal or nonunique or fabricated_fallback or continuity or initial_boundary"
```

Expected: the tests pass because the validator refuses every invalid
repaired-hash mutation and accepts the alternate genuinely optimal solution.

- [ ] **Step 3: Conduct independent multi-lens review**

Use separate reviewers, sequentially integrating findings between rounds:

1. mathematical lens: singular oracle faces, squared/unsquared objectives,
   coefficients, suboptimal solutions, fabricated fallbacks, and non-unique
   optima;
2. artifact-security lens: hashes, initial/cross-round continuity, complete
   manifest/config/data binding, trust-anchor claim strength, and fail-closed
   behavior;
3. reproducibility lens: schema v1/v2 timing, raw-evidence binding, actual
   launcher buffering/pipeline failures, README/operator semantics, and remote
   shutdown behavior.

Each reviewer must inspect the actual `7325bf5..HEAD` diff and tests, report
only concrete findings with file/line evidence, and attempt at least one
counterexample. Cure every valid finding with a new failing regression test
before code. Repeat the same three lenses after cures until no actionable
finding remains.

- [ ] **Step 4: Record the local closeout**

Write the review report with commit IDs, exact commands/results, mutation evidence, any findings/cures, remaining limitations, and a clear statement that no E0 artifacts were modified and no model rerun has been authorized or performed.

- [ ] **Step 5: Commit any review-driven cures and the canonical report pointer**

Commit code/tests per finding rather than squashing unrelated cures. Keep the report in the outer root review folder; if a repository copy is needed, add only a short link/index under `docs/`.

---

### Task 7: Revalidate, preserve, and close the completed E0 campaign

**Files/artifacts:**
- Create: `post_e0_closeout.sh`
- Create: `tests/test_post_e0_closeout.py`
- Read-only remote source: `/home/turjo/FedCRAG_E0_RESULTS/`
- Create local preservation root: `/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/`
- Create: `validation_summary.json`
- Create: `2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md`
- Create: `SOURCE_SHA256SUMS`
- Create: `PACKAGE_SHA256SUMS`

- [ ] **Step 1: Write closeout-driver tests without implementation**

Add hermetic tests around a fake `gcloud` executable and temporary remote/local
trees. Cover project/zone discovery, start-if-terminated, the successful
snapshot, validation/copy/checksum failures, existing-destination refusal,
unique failed-attempt preservation, and unconditional shutdown. Test zero and
two matching instances, literal `(unset)` project, stop-command failure,
malformed/query-failed status, and bounded retries that never reach
`TERMINATED`. Preserve the original failure code unless shutdown verification
itself produces the distinct critical failure. Publication is refused in every
failure case. Execute the script in tests with `/bin/bash`, record
`/bin/bash --version`, and require compatibility with the workspace's Bash
3.2 feature set.

- [ ] **Step 2: Witness RED before creating the driver**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_post_e0_closeout.py -q -p no:cacheprovider
```

Expected: FAIL because `post_e0_closeout.sh` does not exist. Implement the
driver only after preserving this RED output.

- [ ] **Step 3: Implement the complete fail-safe driver**

Implement these phases inside `post_e0_closeout.sh`, in this exact order:

1. Require a clean worktree. Resolve the project and all zones without
   `--limit`; reject an empty/`(unset)` project and require exactly one nonempty
   matching zone. Pass `--project` and `--zone` thereafter.
2. Install an EXIT/INT/TERM cleanup trap before starting the VM. The trap uses
   bounded stop/status retries and succeeds only after observing
   `TERMINATED`; it preserves the original failure unless termination cannot be
   verified, which is a distinct critical exit.
3. Refuse an existing canonical destination. Create a unique sibling attempt
   directory; on failure retain it under a unique failure ID. Start the VM only
   if needed and record before/after states. Never launch training.
4. Create a Git bundle for the exact clean audit commit, copy it to remote
   `/tmp`, and clone a fresh `/tmp/fedspan-post-e0-audit-<sha>`. Record the
   bundle hash, audit commit, exact command, and Python/Torch/NumPy versions.
   Never check out over the execution tree or write under the artifact root.
5. Require `COMPLETE.json`, manifest, and results to record the existing
   12-character identity `7325bf56381c`; resolve it uniquely and require full
   object `7325bf56381c24c6a4af013688bdd417c95d7d7d`. Require eleven validated
   rows.
6. Refuse every symlink/special entry under the remote artifact root. Create a
   NUL-safe sorted relative-file inventory with byte sizes and SHA-256 outside
   that root. Copy the inventoried manifest into remote staging, verify its
   path/size/hash, and use exactly that staged manifest for validation.
7. Resolve the manifest's archived `--data_root`, recompute the four canonical
   slice fingerprints without network access, and run the strengthened
   validator over all eleven rows. Write machine/Markdown summaries outside
   the artifact root. A scientific failure is exported and exits nonzero; it
   never authorizes a rerun.
8. Put exact remote evidence only in `$ATTEMPT/artifacts/` and audit commit,
   bundle, commands, environment, validator output, closeout tooling, and
   reports only in `$ATTEMPT/audit/`. Transfer exactly the regular-file
   inventory, then require remote pre/post and local artifact path/size/hash
   inventories to match. Reconfirm the exported manifest digest.
9. Write `SOURCE_SHA256SUMS` for remote/local source equality, then explicitly
   stop and verify `TERMINATED`. Keep the EXIT trap as an idempotent fallback
   guarded against double cleanup.
10. Write the final shutdown record into `audit/`, then generate and verify
    `PACKAGE_SHA256SUMS` over `SOURCE_SHA256SUMS` and every finalized `audit/`
    file, excluding only `PACKAGE_SHA256SUMS` itself.
11. Atomically rename the staged attempt to canonical `2026-08-25/` only after
    the finalized package checksum, shutdown verification, and every other
    gate pass.

The script discovers with:

```bash
E0_PROJECT="$(gcloud config get-value project)"
if ! E0_ZONE_OUTPUT="$(gcloud compute instances list \
  --project "$E0_PROJECT" --filter='name=thesis-fedcrag' \
  --format='value(zone.basename())')"; then
  exit 3
fi
E0_ZONES=()
while IFS= read -r zone; do
  [[ -n "$zone" ]] && E0_ZONES[${#E0_ZONES[@]}]="$zone"
done <<< "$E0_ZONE_OUTPUT"
```

Keep the driver compatible with the workspace's `/bin/bash` 3.2 and execute
the hermetic suite with that interpreter. Do not use `mapfile` or associative
arrays; explicitly capture the discovery command status before splitting its
output.

- [ ] **Step 4: Run GREEN, inspect, and commit closeout tooling**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_post_e0_closeout.py -q -p no:cacheprovider
bash -n post_e0_closeout.sh
git add post_e0_closeout.sh tests/test_post_e0_closeout.py
git commit -m "ops: add fail-safe E0 preservation closeout"
git status --porcelain
```

- [ ] **Step 5: Run the exact clean-commit production command**

```bash
POST_E0_DEST=/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25 \
  bash post_e0_closeout.sh
```

Expected: exit 0; canonical `artifacts/`, `audit/`,
`SOURCE_SHA256SUMS`, and `PACKAGE_SHA256SUMS` exist; the final recorded VM
status is `TERMINATED`. Any nonzero exit preserves a uniquely named failure
attempt and still runs the bounded shutdown verification.

- [ ] **Step 6: Anchor and publish the successful closeout**

Only after Step 5 exits zero, compute the digest of `PACKAGE_SHA256SUMS`, add a
small repository closeout record and update README status from “validation
pending” to “external E0 correctness campaign strengthened-validated.” Tests
must retain the legacy timing disclosure, post-hoc trust-anchor limitation,
and prohibition on paper-scale efficacy claims. Commit these report-only
changes and publish the audit branch to the existing remote. Without a signing
key or retention-locked store, call this the first externally anchored
post-hoc inventory, not a signed historical attestation.

- [ ] **Step 7: Apply the final publication gate**

Publication promotion is allowed only when all eleven rows pass scientific
validation, all adversarial regression attacks are refused, the pre/post
remote and local inventories match, the legacy timing and historical
trust-anchor limitations are disclosed, and the VM is terminated. This gate
does not claim that E0 efficacy metrics are paper-scale results.

---

## Final acceptance checklist

- [ ] Baseline 200 tests remain green and every new test is green.
- [ ] Audit module has no production solver dependency.
- [ ] Singular cancellation and non-unique optimum cases are correct.
- [ ] Repaired-hash opposite-direction mutation is refused.
- [ ] Feasible-but-suboptimal minnorm/maxmin decisions and fabricated fallbacks are refused.
- [ ] Alternate weights on a genuinely non-unique optimal face are accepted.
- [ ] Round-1 initial-state substitution is refused.
- [ ] Repaired-hash cross-round continuity mutation is refused.
- [ ] Manifest, row metadata, configuration hash, and data fingerprints bind every execution-relevant parameter.
- [ ] Schema-v2 valid timing and the actual unbuffered pipeline pass; malformed, raw-mismatched, or pipeline-failed v2 timing fails.
- [ ] Schema-v1 completed E0 timing is preserved and explicitly unavailable per round.
- [ ] README command and eleven-row campaign text match executable behavior.
- [ ] Three adversarial review lenses report no remaining actionable finding.
- [ ] All eleven frozen E0 rows pass strengthened scientific validation, or exact failures are reported without reruns.
- [ ] Pre/post remote and local path/size/SHA-256 inventories match.
- [ ] Closeout labels the evidence post-hoc internally consistent rather than historically immutable unless an independent pre-audit anchor is found.
- [ ] VM status is confirmed `TERMINATED`.
