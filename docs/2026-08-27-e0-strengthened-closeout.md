# E0 strengthened validation closeout

Date: 2026-08-27

## Outcome

The completed eleven-row E0 correctness campaign is strengthened-validated.
All eleven validators exited zero. The closeout copied the immutable campaign
package, independently checked its inventories and checksums, and published the
canonical local package at:

`/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/`

The package is 3.3 GiB. The SHA-256 digest of its `PACKAGE_SHA256SUMS` file is:

`b264d588ec87454a05c21c012b38d8d94b79eb5d62e61ef7bc5b8ff640eb78ac`

## Bound identities

- E0 execution commit: `7325bf56381c24c6a4af013688bdd417c95d7d7d`
- Closeout/audit commit: `f10fd43a446f604226e26b235d1df82222545b20`
- Fresh snapshot: `fedcrag-e0-closeout-20260827`
- Snapshot source: `asia-south1-c/disks/thesis-fedcrag-restored`
- Retained clone: `thesis-fedcrag-e0-closeout`
- Retained boot disk: `asia-northeast1-c/disks/thesis-fedcrag-e0-closeout-boot-jp-c`
- Validation host: `asia-northeast1-c`, `g2-standard-4`, one NVIDIA L4,
  standard provisioning
- Original final state: `TERMINATED`
- Clone final state: `TERMINATED`

The clone boot disk has auto-delete disabled. The successful clone, disk, and
fresh snapshot are retained pending an explicit cleanup decision.

## What passed

- 11/11 frozen manifest rows validated with zero exits.
- The execution source resolved to the full frozen commit and its worktree was
  clean.
- Launch arguments, dataset fingerprints, runtime provenance, archived state
  continuity, and aggregate recomputation were independently checked.
- Every row had four round-to-round continuity boundaries checked.
- The worst aggregate recomputation tolerance ratio was
  `0.0007450580596923827`, well inside the acceptance bound.
- The two frozen-A FedSpan rows replayed five applied rounds each with optimal
  status. Their largest recorded Gram delta was approximately `1.97e-11`; the
  largest scientific direction uncertainty was approximately `3.72e-10`.
- RawMaxMin simplex feasibility and objective optimality were independently
  replayed; the largest recorded optimality gap was approximately `5.28e-15`.
- Source-pre, staged, source-post, and local inventories matched, and the
  canonical package checksum manifest was verified before publication.

## Migration disclosure

The original `g2-standard-8` VM could not restart because its zone was in L4
stockout. A fresh snapshot was created from the stopped post-E0 disk. The exact
shape also stocked out across all three Singapore zones, all three Taiwan
zones, and Japan zone `asia-northeast1-a`; `g2-standard-4` also stocked out in
Japan zone `asia-northeast1-b`. The successful preservation host was therefore
a suitable smaller `g2-standard-4` with the same single 24 GiB L4 in
`asia-northeast1-c`.

This host change did not rerun training, recompute E0 outputs, or modify the
archived artifacts. It was used only to validate and transfer the stopped
disk's existing E0 results.

## Limitations and next gate

- E0 has five rounds and is a correctness/attribution grid, not a paper-scale
  efficacy experiment.
- Legacy schema-v1 per-round timing is unavailable; measured total row runtime
  remains available from launcher boundaries.
- The initial adapter trust anchor is post-hoc and self-recorded rather than an
  external signed digest.
- The interpreter invocation path is lexically bound, but no historical
  executable digest exists.
- E1–E5 have not run. This closeout establishes that the implemented methods
  and archived E0 rows satisfy their stated contracts; it does not establish
  retrieval efficacy or support a paper-scale claim.

The E0 correctness gate is closed. The next research action is E1: run the
phenomenon experiment, inspect retrieval/BWT/forgetting results, and iterate on
the experimental method if the evidence is not satisfactory.
