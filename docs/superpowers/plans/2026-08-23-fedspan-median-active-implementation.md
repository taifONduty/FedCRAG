# FedSpan Median-Active Step Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the explicit `median-active` true-step policy, preserve fixed-step compatibility, produce a clean reproducible evidence commit, and freeze—but do not launch—the ten E0 runs.

**Architecture:** Resolve the round step inside the pure FedSpan aggregation function after finite/activity gating, then pass the resolved value through the existing exact solve/application path. The driver owns explicit CLI legality, static configuration hashes, filenames, and persistence. A dedicated E0 launcher owns the immutable ten-row matrix and refuses dirty provenance.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy HiGHS, pytest, PEFT, Bash, Git.

## Global Constraints

- E0 primary policy is explicitly `median-active`: `s_t = median({r_k : k is active})` in the concatenated PEFT-scale-aware effective-B coordinate.
- `normmaxmin` has no implicit step-policy default.
- Fixed mode remains available and requires a positive finite `--fedspan_step_norm`.
- Median-active mode rejects `--fedspan_step_norm` and never silently changes policy.
- No-active, invalid derived norm, solver, cancellation, cap, or reconstruction failures apply a logged zero update.
- Preserve all historical arms, results, and the original dirty working tree.
- Paper-grade execution rejects dirty/unknown Git provenance; never use `--allow_dirty_provenance` for E0.
- Tests precede production changes and every new test is observed failing for the intended reason.
- E0 is prepared but not launched without separate authorization for the estimated 50 L4 GPU-hours.

---

### Task 1: Pure aggregation policy and diagnostics

**Files:**
- Modify: `tests/test_aggregation.py`
- Modify: `aggregation_schemes.py`

**Interfaces:**
- Consumes: existing `_fedspan_blocks(...)`, activity thresholds, `_zero_fedspan_result(...)`, and `apply_fedspan_update(...)`.
- Produces: `fedspan_delta_weights(..., step_norm=None, step_policy="fixed", ...)` returning `step_policy`, `declared_step_norm`, `resolved_step_norm`, `client_norms`, and the existing solve/application fields.

- [ ] **Step 1: Write failing odd/even median and inactive-client tests**

Construct effective client norms `[1, 3, 100]` and `[1, 3, 5, 100]`, call:

```python
result = fedspan_delta_weights(
    clients,
    broadcast,
    module_scales=scales,
    step_policy="median-active",
    step_norm=None,
)
```

Assert resolved norms `3.0` and `4.0`. Add a nonfinite client and a below-threshold client and assert neither changes the active median.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_aggregation.py -q -p no:cacheprovider -k median_active
```

Expected: FAIL because the function has no median-active policy and converts `None` to `float`.

- [ ] **Step 3: Implement explicit step resolution**

Change the public signature to:

```python
def fedspan_delta_weights(client_states, broadcast_state, module_scales,
                          step_norm=None, step_policy="fixed",
                          active_abs_tol=1e-12,
                          active_rel_tol=1e-8, mixture_norm_tol=1e-6,
                          max_abs_delta_weight=None):
```

Validate the declared relationship before geometry:

```python
if step_policy not in ("fixed", "median-active"):
    raise ValueError("step_policy must be 'fixed' or 'median-active'")
if step_policy == "fixed":
    if step_norm is None or not math.isfinite(float(step_norm)) \
            or float(step_norm) <= 0:
        raise ValueError("fixed step policy requires a positive finite step_norm")
    declared_step_norm = float(step_norm)
else:
    if step_norm is not None:
        raise ValueError("median-active step policy rejects step_norm")
    declared_step_norm = None
```

After activity gating, resolve:

```python
resolved_step_norm = (
    declared_step_norm
    if step_policy == "fixed"
    else (float(np.median([client_norms[index] for index in active]))
          if active else None)
)
```

Use the resolved value for coefficients and verification. Extend zero and successful diagnostics with the three policy fields. Keep `requested_step_norm` as a compatibility alias of the resolved value; `apply_fedspan_update` accepts `None` only for a zero fallback.

- [ ] **Step 4: Add and observe RED for no-active and legality cases**

Assert no-active median rounds return `status="no_active"`, `declared_step_norm=None`, `resolved_step_norm=None`, and `requested_step_norm=None`. Parameterize unknown policy, median plus a constant, and fixed with `None`, zero, negative, NaN, or infinity; each raises the policy-specific `ValueError`.

- [ ] **Step 5: Run aggregation tests and verify GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_aggregation.py -q -p no:cacheprovider
```

Expected: all aggregation tests PASS, including old fixed-step tests.

- [ ] **Step 6: Commit the pure aggregation unit**

```bash
git add aggregation_schemes.py tests/test_aggregation.py
git commit -m "feat: add median-active FedSpan step policy"
```

---

### Task 2: Driver legality, identifiers, and persisted audit record

**Files:**
- Modify: `tests/test_fedspan_driver.py`
- Modify: `federated_forgetting.py`

**Interfaces:**
- Consumes: Task 1's explicit policy API and diagnostics.
- Produces: `--fedspan_step_policy {fixed,median-active}`, policy-sensitive hashes and filenames, and persisted method/round records.

- [ ] **Step 1: Write failing CLI matrix tests**

Cover exactly:

```text
normmaxmin + no policy                         -> parser error
normmaxmin + fixed + no constant              -> parser error
normmaxmin + fixed + positive constant        -> legal
normmaxmin + median-active + constant          -> parser error
normmaxmin + median-active + no constant       -> legal
non-normmaxmin + either FedSpan policy option  -> parser error
```

Guard model/data work with a sentinel so illegal rows fail before external work.

- [ ] **Step 2: Run focused CLI tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_fedspan_driver.py -q -p no:cacheprovider -k "policy or normmaxmin"
```

Expected: FAIL because the parser has no policy option.

- [ ] **Step 3: Implement parser and legality rules**

Add:

```python
ap.add_argument(
    "--fedspan_step_policy",
    choices=["fixed", "median-active"],
    default=None,
    help="explicit true-step policy required for normmaxmin",
)
```

For canonical `normmaxmin`, require a policy and enforce fixed/median argument rules exactly as Task 1. For every other arm, reject either FedSpan step option. Pass both fields into `fedspan_delta_weights`.

- [ ] **Step 4: Write failing hash, filename, and persistence tests**

Extend `frozen_args` with `fedspan_step_policy="fixed"`. Assert the fixed and median configuration hashes differ. In the mocked median run assert:

```python
assert result["method_contract"]["fedspan_step_policy"] == "median-active"
assert diagnostic["step_policy"] == "median-active"
assert diagnostic["resolved_step_norm"] == pytest.approx(expected_median)
assert diagnostic["application"]["applied_step_norm"] == pytest.approx(
    diagnostic["resolved_step_norm"], abs=2e-6)
```

Assert median filenames contain `smedian-active`; fixed filenames retain the numeric step tag.

- [ ] **Step 5: Run the new persistence tests and verify RED**

Expected: FAIL because the hash, method contract, and filename do not carry the policy.

- [ ] **Step 6: Implement audit propagation**

Add `fedspan_step_policy` to the configuration hash, method contract, argument record, and aggregation call. Build the collision-safe tag with:

```python
if args.fedspan_step_policy == "fixed":
    step_tag = format(args.fedspan_step_norm, ".8g").replace(".", "p")
else:
    step_tag = "median-active"
basis += f"-s{step_tag}"
```

Persist Task 1's round diagnostic unchanged before costly evaluation.

- [ ] **Step 7: Verify the driver and full suites GREEN**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_fedspan_driver.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests -q -p no:cacheprovider
```

Expected: both PASS without collection errors.

- [ ] **Step 8: Commit the driver unit**

```bash
git add federated_forgetting.py tests/test_fedspan_driver.py
git commit -m "feat: expose auditable FedSpan step policies"
```

---

### Task 3: Freeze exact E0 execution and validation

**Files:**
- Create: `run_e0.sh`
- Create: `tests/test_e0_manifest.py`
- Modify: `README.md`
- Modify: `/Users/turjo/Desktop/FedCRAG/review_outputs/E0_E5_RUN_MANIFEST.md`
- Create: `/Users/turjo/Desktop/FedCRAG/review_outputs/E0_EXECUTION_FREEZE.md`

**Interfaces:**
- Consumes: Task 2's explicit CLI and clean-provenance refusal.
- Produces: `bash run_e0.sh verify` for a zero-GPU manifest check and `bash run_e0.sh run` for the separately authorized ten-run campaign.

- [ ] **Step 1: Write a failing static matrix test**

Parse `run_e0.sh` and assert ten unique run IDs from:

```python
coordinates = {
    "trainable-ab": ("uniform", "rawmaxmin"),
    "frozen-a": ("uniform", "rawmaxmin", "normmaxmin"),
}
regimes = {"capped-500": 500, "full": 0}
```

Assert seed 42, Contriever, LoRA rank 16, five rounds, four fixed BEIR clients, saved states, median-active only for frozen-A/normmaxmin, tolerances `1e-12`, `1e-8`, and `1e-6`, and no coefficient cap. No row contains the dirty-provenance override.

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/test_e0_manifest.py -q -p no:cacheprovider
```

Expected: FAIL because `run_e0.sh` does not exist.

- [ ] **Step 3: Implement a fail-fast launcher**

The script fixes seed 42, Contriever, five rounds, cap 500 versus full 0, the four clients, and output root `results/e0`. `verify` checks a clean tree, prints ten expanded commands, verifies Python imports, runs the CPU suite, and exits before model loading or VM creation. `run` repeats those checks, runs exactly ten rows with distinct logs, stops on first failure, and never launches E1-E5.

- [ ] **Step 4: Implement post-run correctness validation**

After each normmaxmin row, reject unless every round has matching hashes, bitwise-fixed A, finite solver residuals for solved rounds, matching solved/applied step hashes, and:

```python
abs(application["applied_step_norm"] - diagnostic["resolved_step_norm"]) \
    <= 5e-6 * max(1.0, diagnostic["resolved_step_norm"])
```

Fallback rounds instead require applied norm zero and the recorded failure status.

- [ ] **Step 5: Update the written contract**

Document flags in `README.md`. Append exact E0 constants, ten IDs, commands, thresholds, no-cap declaration, output root, and acceptance checks to the root manifest. Create `E0_EXECUTION_FREEZE.md` with status `FROZEN, NOT RUN` and state that it does not authorize cloud spend.

- [ ] **Step 6: Verify without spending**

```bash
bash -n run_e0.sh
bash run_e0.sh verify
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests -q -p no:cacheprovider
git diff --check
```

Expected: ten commands printed, all CPU checks PASS, and no cloud instance created.

- [ ] **Step 7: Commit the E0 freeze**

```bash
git add run_e0.sh tests/test_e0_manifest.py README.md
git commit -m "test: freeze the E0 attribution campaign"
```

Root reports are outside the nested repo and are hashed in the final report.

---

### Task 4: Assemble and verify clean evidence provenance

**Files:**
- Create: `/Users/turjo/Desktop/FedCRAG/review_outputs/CLEAN_EVIDENCE_BUILD.md`
- Update: `/Users/turjo/Desktop/FedCRAG/review_outputs/IMPLEMENTATION_HANDOFF.md`
- Update: clickable mirror under the Codex `outputs/FedCRAG_review_bundle` folder

**Interfaces:**
- Consumes: Tasks 1-3 and the previously reviewed implementation files.
- Produces: an immutable clean branch/worktree, source/report hashes, and an E0-readiness verdict.

- [ ] **Step 1: Create an isolated evidence worktree**

Use the worktree workflow from the campaign branch so the current dirty tree is never reset, cleaned, or overwritten. Name the evidence branch `fedspan-e0-clean` and place its worktree in a tool-created temporary directory.

- [ ] **Step 2: Transfer only reviewed implementation scope**

Transfer:

```text
aggregation_schemes.py
federated_forgetting.py
tests/test_aggregation.py
tests/test_fedspan_driver.py
tests/test_e0_manifest.py
README.md
requirements.txt
run_e0.sh
docs/superpowers/specs/2026-08-23-fedspan-step-policy-design.md
docs/superpowers/plans/2026-08-23-fedspan-median-active-implementation.md
```

Do not transfer `runs.tsv`, historical results/logs, `repair_arms.sh`, `mechanism_suite.py`, `zz_probe.txt`, or `Problem_Formulation.pdf` merely to make the tree appear clean.

- [ ] **Step 3: Commit the exact evidence implementation**

Verify the staged name list contains only declared scope, inspect the full staged diff, commit, and record the commit SHA plus source SHA-256 values.

- [ ] **Step 4: Rebuild and verify from the clean worktree**

Run the complete CPU suite with bytecode/cache disabled, Python compilation, Bash syntax, `git diff --check`, and `bash run_e0.sh verify`. Require `git status --porcelain` to be empty afterward.

- [ ] **Step 5: Write the readiness report**

Record commands and outputs, Git commit/tree status, Python/platform/package-lock hash, source hashes, test count, E0 IDs, Google Cloud project/account, and that no active instance was found. Set exactly one verdict:

```text
E0 READY — CLOUD EXECUTION NOT YET AUTHORIZED
```

or a precise blocked verdict.

- [ ] **Step 6: Stop before spending**

Do not provision, restore, start, or resize a cloud instance. Present exact estimated cost/time and request explicit authorization before `run` mode.

