# Cross-region E0 closeout clone design

**Date:** 2026-08-27  
**Branch:** `fedspan-post-e0-audit`  
**Status:** user-approved operational direction; implementation gated by this
written design and its implementation plan

## Goal

Finish Task 7 despite the L4 stockout in `asia-south1-c` by validating and
exporting the completed E0 evidence from a fresh, same-spec clone in another
region. Preserve the original stopped VM and disk unchanged, retain the
recorded E0 execution identity, and make the host migration explicit in the
closeout evidence.

This operation does not launch training, alter E0 artifacts, authorize E1, or
turn E0 into paper-scale efficacy evidence.

## Current facts

- Original instance: `thesis-fedcrag`, `asia-south1-c`, `TERMINATED`.
- Original compute: `g2-standard-8`, one NVIDIA L4, standard provisioning.
- Original boot disk: `thesis-fedcrag-restored`, 200 GiB `pd-balanced`.
- The boot disk is persistent but attached with `autoDelete: true`; the
  original instance must not be deleted.
- The existing snapshot `fedcrag-w1-paused` was created on 2026-08-22, before
  E0 completed, so it is not an acceptable migration source.
- `asia-southeast1-{a,b,c}` supports `g2-standard-8` and L4. Project quota is
  one unused regional L4, at least 100 unused CPUs, and 24 unused instances;
  the default subnet exists.
- GCP exposes supported locations and quota but no trustworthy read-only live
  stock query. Capacity is established only by an allocation attempt.

## Considered approaches

### 1. Fresh same-spec cross-region clone — selected

Snapshot the stopped current disk, create a separate boot disk and
`g2-standard-8` + one-L4 instance in Singapore, and run the audited closeout
against that explicit clone. This preserves the original and keeps the host
class constant, at the cost of snapshot/disk storage, brief GPU runtime, and a
documented region/host migration.

### 2. Continue waiting in `asia-south1-c`

This has the smallest provenance deviation but has already failed three
bounded retries with zonal L4 stockout and has no completion time guarantee.

### 3. CPU-only validation clone

The post-hoc validator does not train and is expected to be CPU-capable. This
would avoid L4 stockout but changes the machine class and needs a stronger
execution-environment disclosure. Keep it as a fallback, not the selected
path.

## Resources and naming

Use explicit, collision-refusing names:

- Fresh snapshot: `fedcrag-e0-closeout-20260827`
- Candidate instance: `thesis-fedcrag-e0-closeout`
- Candidate boot disk: `thesis-fedcrag-e0-closeout-boot-<zone-suffix>`
- Candidate zones, in bounded order:
  `asia-southeast1-a`, `asia-southeast1-b`, `asia-southeast1-c`

Never overwrite or delete an existing resource with one of these names.

## Provisioning flow

1. Reconfirm the original VM is `TERMINATED`, its disk is `READY`, and the
   canonical local closeout destination is absent.
2. Refuse a pre-existing fresh-snapshot name. Snapshot the current
   `thesis-fedcrag-restored` disk into Asia snapshot storage and wait for
   snapshot status `READY`.
3. Record snapshot identity, creation time, source disk, size, storage bytes,
   storage location, and GCP command statuses locally.
4. For each Singapore zone in order:
   - refuse pre-existing candidate instance/disk names;
   - create a 200 GiB `pd-balanced` boot disk from the fresh snapshot with the
     disk retained independently of instance deletion;
   - create `thesis-fedcrag-e0-closeout` as `g2-standard-8` with its included
     one L4, standard provisioning, `TERMINATE` host maintenance, automatic
     restart, the default regional subnet, premium ephemeral external NAT,
     the original service account/scopes, vTPM and integrity monitoring;
   - if allocation fails with a capacity error, verify no instance exists,
     delete only that recoverable candidate disk, record the refusal, and try
     the next zone;
   - stop at the first successful allocation.
5. Verify the created instance descriptor and disk descriptor: exact machine
   type, one L4, standard provisioning, boot-disk size/type, source snapshot,
   service account, network, and selected zone. Never infer these from the
   create command alone.

## Closeout-driver change

The production default remains `thesis-fedcrag`. Add one explicit controller
override, `POST_E0_INSTANCE`, for the clone. Validate it against the Compute
Engine instance-name grammar before any cloud query. Discovery must retrieve
both name and zone, match the requested name exactly, and still require exactly
one matching instance.

Every start, status, SSH, SCP, stop, and audit command must use the selected
instance plus the discovered project and zone. The successful audit package
must record the selected instance name and the externally captured migration
descriptors. The override must not change remote artifact, source, interpreter,
execution-commit, destination, or scientific validation gates.

## Failure handling

- Snapshot failure: stop; original remains untouched.
- Candidate disk failure: stop or advance only when no partial instance exists.
- Capacity refusal: retain the controller log; remove only the recoverable
  failed-zone clone disk after verifying its exact name/source.
- Instance creation ambiguity or unexpected configuration: stop the candidate,
  retain it for diagnosis, and do not run closeout.
- Closeout failure: its existing trap must stop and verify the clone, preserve
  the failed attempt, and refuse canonical publication.
- Controller interruption: independently verify both original and clone states;
  any created clone must be stopped.
- Never delete the original instance/disk. Do not delete a successful clone,
  its boot disk, or the fresh snapshot without separate cleanup approval.

## Testing and review

Before production code changes:

- add a failing hermetic test that targets a non-default instance and proves
  every fake GCP call uses it;
- add refusals for empty/invalid/inexact/ambiguous instance overrides;
- retain default-instance compatibility and all existing shutdown,
  preservation, and publication tests.

Then run the closeout-focused suite, the provenance/launcher suite, the full
repository suite, Bash 3.2 syntax validation, and `git diff --check`. Obtain an
independent review of the exact change before creating or starting the clone.

## Success criteria

- Fresh snapshot is `READY` and identifies the current stopped E0 disk.
- Exactly one verified same-spec clone is created in Singapore.
- Original VM remains `TERMINATED` and its instance/disk are unchanged.
- All eleven E0 rows pass the strengthened validator from the audited commit.
- Source pre/post, staged, and local inventories match.
- Canonical `SOURCE_SHA256SUMS` and `PACKAGE_SHA256SUMS` verify.
- Clone and original VM are independently confirmed `TERMINATED`.
- Migration, legacy timing, post-hoc trust-anchor, and non-paper-scale limits
  are disclosed before the repository status is promoted or E1 is authorized.
