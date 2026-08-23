# FedSpan E0 Step-Norm Policy Design

**Date:** 2026-08-23  
**Status:** approved design  
**Scope:** corrected `normmaxmin` FedSpan only; no change to historical arms or results

## Decision

E0 will use an explicitly selected `median-active` step-norm policy. In round
`t`, after the existing finite-client and activity checks, the server sets

```
s_t = median({r_k : k is active}),
```

where `r_k` is the norm of client `k`'s concatenated, PEFT-scale-aware effective
LoRA-B update. The server then applies the existing normalized FedSpan direction
at true effective-update norm `s_t`.

This is the design already stated in the adversarial solution review. It is
robust to unequal client magnitudes, changes naturally with convergence, and
keeps the direction decision separate from the distance moved. A uniform-
reference norm is rejected because cancellation in the uniform direction can
make the reference step arbitrarily small. A single fixed constant remains
available for controlled ablations but is not the E0 primary policy.

## Interface

Add an explicit command-line option:

```
--fedspan_step_policy {fixed,median-active}
```

There is no implicit default for `normmaxmin`. The legal configurations are:

- `fixed`: requires a positive finite `--fedspan_step_norm`.
- `median-active`: rejects `--fedspan_step_norm`, derives `s_t` independently
  each round, and records it.

Other aggregation schemes do not use either option. Existing fixed-policy
behavior remains available and unchanged.

## Data Flow

1. Compute each client's PEFT-scale-aware effective-B update norm.
2. Apply the existing finite-value and absolute/relative activity gates.
3. If the policy is `median-active`, compute the float64 median over active
   norms; if it is `fixed`, use the declared constant.
4. Pass that resolved positive norm to the existing FedSpan solver/application
   path.
5. Reconstruct the applied effective update independently and verify that its
   norm matches the resolved `s_t` within the existing numeric tolerance.
6. Persist the policy, all client norms, active set, resolved `s_t`, solver
   record, proposed/applied coefficients, and applied-state hashes.

The run-configuration hash and collision-safe filename include the policy. For
fixed mode they also include the declared constant; for median-active mode the
per-round resolved values live in round diagnostics rather than the static
configuration.

## Fail-Closed Behavior

- No active clients: apply a logged zero update; `s_t` is null because no
  median exists.
- Any nonfinite client update: exclude that client under the existing policy
  and record the reason.
- A derived median that is nonfinite or nonpositive: apply a logged zero update.
- Solver failure, near cancellation, coefficient-cap violation, reconstruction
  failure, or norm mismatch: retain the existing logged zero-update behavior.
- Dirty or unknown Git provenance remains forbidden for paper-grade E0.

No fallback silently switches from median-active to fixed, uniform-reference,
or raw averaging.

## Testing

Tests are written before production changes and must demonstrate:

1. Median selection for odd and even active-set sizes.
2. Inactive, zero, tiny, and nonfinite clients do not affect the median.
3. No-active-client rounds produce a logged zero update with no resolved norm.
4. Median-active rejects an explicit fixed norm; fixed mode still requires one.
5. The run-configuration hash changes with the policy.
6. Mocked end-to-end persistence records policy, client norms, resolved norm,
   and an applied update whose independently measured norm equals that value.
7. The full existing randomized, PEFT, persistence, and driver suite stays
   green.

## Clean Provenance and E0 Boundary

The current working tree contains unrelated and pre-existing user changes. The
evidence build will therefore be assembled on an isolated branch/worktree from
the current campaign commit, containing only the reviewed implementation,
tests, required documentation/dependencies, and this policy change. The
original dirty working tree will not be reset or cleaned.

After the clean commit passes the complete local suite, exact E0 commands and
output identifiers will be frozen in the root-level `review_outputs` folder.
E0 remains blocked until cloud execution is separately authorized because the
manifest estimates 50 L4 GPU-hours. E1-E5 remain blocked until E0's correctness
and provenance gates pass.
