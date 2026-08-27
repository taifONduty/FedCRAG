# Cross-region E0 Closeout Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an audited, same-spec Singapore clone of the stopped E0 VM from a fresh snapshot and use it to complete Task 7 without modifying the original VM, disk, or E0 artifacts.

**Architecture:** Extend the closeout driver with a fail-closed target-instance override and verified cloud-configuration evidence. Add a Bash 3.2 orchestration driver that snapshots the stopped disk, tries three Singapore zones in bounded order, cleans only recoverable failed-zone clone disks, invokes closeout while the successful L4 allocation is running, and independently verifies both VMs stopped on every exit.

**Tech Stack:** GNU Bash 3.2, Python 3.13, pytest, fake `gcloud`, Google Compute Engine CLI, Git, SHA-256 inventories.

## Global Constraints

- Work only in `/Users/turjo/Desktop/FedCRAG/worktrees/fedspan-post-e0-audit` on branch `fedspan-post-e0-audit`.
- Never delete or modify original instance `thesis-fedcrag`, disk `thesis-fedcrag-restored`, or `/home/turjo/FedCRAG_E0_RESULTS`.
- Preserve E0 execution identity `7325bf56381c24c6a4af013688bdd417c95d7d7d`.
- Use fresh snapshot `fedcrag-e0-closeout-20260827`; never use pre-E0 snapshot `fedcrag-w1-paused`.
- Use clone `thesis-fedcrag-e0-closeout`; try `asia-southeast1-a`, `asia-southeast1-b`, then `asia-southeast1-c`.
- Clone compute is exactly `g2-standard-8`, one NVIDIA L4, standard provisioning, and a 200 GiB `pd-balanced` boot disk from the fresh snapshot.
- Disable clone boot-disk auto-delete. Retain a successful clone, disk, and snapshot until separately authorized cleanup.
- No training, E0 rerun, E1 launch, model change, artifact mutation, or paper-scale efficacy claim is authorized.
- Every production code change begins with a witnessed failing regression test.
- Keep shell code compatible with GNU Bash 3.2; no associative arrays or `mapfile`.
- After cloud mutation, the original must be observed `TERMINATED`; any created clone must also be observed `TERMINATED` on every exit.

---

### Task 1: Bind closeout to an explicit, verified target

**Files:**
- Modify: `post_e0_closeout.sh:11-35,245-280,354-430,639-680`
- Modify: `tests/test_post_e0_closeout.py:35-330`
- Modify: `.superpowers/sdd/task-7-report.md` (ignored execution record)

**Interfaces:**
- Consumes: optional `POST_E0_INSTANCE`; default remains `thesis-fedcrag`.
- Consumes for non-default targets: exact `POST_E0_EXPECTED_SOURCE_SNAPSHOT`.
- Produces: checked `audit/target_instance.json`, `audit/target_disk.json`, and `audit/target_snapshot.json`.
- Preserves: all source, interpreter, commit, shutdown, inventory, checksum, and publication gates.

- [ ] **Step 1: Write non-default target RED tests**

Extend fake `gcloud` so instance discovery returns name plus zone and
instance/disk/snapshot JSON descriptors. Add:

```python
def test_nondefault_instance_is_used_for_every_cloud_operation(closeout_env):
    completed = run_driver(
        closeout_env,
        POST_E0_INSTANCE="thesis-fedcrag-e0-closeout",
        POST_E0_EXPECTED_SOURCE_SNAPSHOT="fedcrag-e0-closeout-20260827",
        FAKE_STATUSES="RUNNING,TERMINATED",
    )
    assert completed.returncode == 0, completed.stderr
    calls = cloud_calls(closeout_env)
    target_calls = [line for line in calls if any(token in line for token in (
        "instances describe", "instances start", "instances stop",
        "compute ssh", "compute scp"))]
    assert target_calls
    assert all("thesis-fedcrag-e0-closeout" in line for line in target_calls)


@pytest.mark.parametrize("value", ["", "Bad_Name", "-leading", "trailing-", "a" * 64])
def test_invalid_instance_override_is_refused_before_cloud(closeout_env, value):
    completed = run_driver(closeout_env, POST_E0_INSTANCE=value)
    assert completed.returncode == 2
    assert cloud_calls(closeout_env) == []


def test_nondefault_instance_requires_expected_snapshot(closeout_env):
    completed = run_driver(
        closeout_env, POST_E0_INSTANCE="thesis-fedcrag-e0-closeout")
    assert completed.returncode == 2
    assert cloud_calls(closeout_env) == []
```

Add descriptor mutations for wrong name, machine type, L4 type/count,
provisioning model, boot disk, disk size/type/status/source snapshot, failed
descriptor command, and two exact-name discovery rows. Each must retain a
failed attempt, publish no canonical destination, and stop a touched clone.

- [ ] **Step 2: Witness RED**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest tests/test_post_e0_closeout.py -q -p no:cacheprovider \
  -k 'nondefault or invalid_instance or target_configuration'
```

Expected: failure because the driver is hardcoded and records no target
descriptors.

- [ ] **Step 3: Implement raw instance validation and exact discovery**

```bash
DEFAULT_INSTANCE=thesis-fedcrag
INSTANCE=${POST_E0_INSTANCE-$DEFAULT_INSTANCE}
case "$INSTANCE" in
    ''|*[!a-z0-9-]*|-*|*-) printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2 ;;
esac
case "$INSTANCE" in
    [a-z]*) ;;
    *) printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2 ;;
esac
[ "${#INSTANCE}" -le 63 ] || { printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2; }
EXPECTED_SOURCE_SNAPSHOT=${POST_E0_EXPECTED_SOURCE_SNAPSHOT:-}
if [ "$INSTANCE" != "$DEFAULT_INSTANCE" ] && [ -z "$EXPECTED_SOURCE_SNAPSHOT" ]; then
    printf '%s\n' "non-default instance requires POST_E0_EXPECTED_SOURCE_SNAPSHOT" >&2
    exit 2
fi
```

Discovery requests `name` and `zone.basename()`, retains only byte-equal names,
and requires exactly one nonempty zone. Every later call uses `$INSTANCE`.

- [ ] **Step 4: Capture and verify descriptors before VM activity**

After attempt creation and trap installation, capture checked JSON:

```bash
gcloud compute instances describe "$INSTANCE" --project "$E0_PROJECT" \
  --zone "$E0_ZONE" --format=json > "$ATTEMPT/audit/target_instance.json" || return 1
```

Controller Python requires exact name/zone, `g2-standard-8`, one `nvidia-l4`,
standard provisioning, and one persistent boot disk. Capture that disk and
require `READY`, size `200`, and `pd-balanced`. For a non-default target,
require its `sourceSnapshot` equals the supplied fresh snapshot, capture that
snapshot, and require `READY`, size 200, and source disk ending in
`/zones/asia-south1-c/disks/thesis-fedcrag-restored`.

- [ ] **Step 5: Run GREEN and commit**

```bash
/bin/bash -n post_e0_closeout.sh
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest tests/test_post_e0_closeout.py tests/test_validate_e0.py \
  tests/test_e0_manifest.py -q -p no:cacheprovider
git diff --check
git add post_e0_closeout.sh tests/test_post_e0_closeout.py
git commit -m "ops: bind E0 closeout to an explicit clone"
```

Expected: all tests pass and syntax/diff checks are clean.

---

### Task 2: Add the fail-safe provisioning controller

**Files:**
- Create: `e0_cross_region_closeout.sh`
- Create: `tests/test_e0_cross_region_closeout.py`
- Modify: `.superpowers/sdd/task-7-report.md` (ignored execution record)

**Interfaces:**
- Invokes `post_e0_closeout.sh` with exact clone and snapshot environment variables.
- Creates one fresh snapshot and one retained successful clone instance/disk.
- Deletes only an exact failed-zone clone disk proven recoverable from the fresh snapshot and having no users.
- Preserves the closeout status unless termination verification fails, which returns 70.

- [ ] **Step 1: Write controller tests before the script exists**

Create fake `gcloud` and fake closeout fixtures with separate original/clone
state streams. The runner begins:

```python
def run_controller(env):
    assert SCRIPT.is_file(), "RED: e0_cross_region_closeout.sh is absent"
    return subprocess.run(
        ["/bin/bash", str(SCRIPT)], cwd=ROOT, env=env,
        capture_output=True, text=True)
```

Use these concrete allocation and cleanup contracts (the fixture maps the
comma-separated capacity stream and named fault to fake `gcloud` responses):

```python
@pytest.mark.parametrize(("capacity", "zones", "expected_status"), [
    ("ready", ["asia-southeast1-a"], 0),
    ("stockout,ready", ["asia-southeast1-a", "asia-southeast1-b"], 0),
    ("stockout,stockout,stockout", [
        "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c"], 20),
])
def test_bounded_zone_search_and_shutdown(env, capacity, zones, expected_status):
    env["FAKE_CAPACITY"] = capacity
    completed = run_controller(env)
    assert completed.returncode == expected_status, completed.stderr
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    attempted = [line.split("--zone ", 1)[1].split()[0] for line in calls
                 if "instances create thesis-fedcrag-e0-closeout" in line]
    assert attempted == zones
    assert Path(env["FAKE_ORIGINAL_STATUS"]).read_text() == "TERMINATED"
    assert Path(env["FAKE_CLONE_STATUS"]).read_text() in ("ABSENT", "TERMINATED")
    if expected_status == 0:
        assert Path(env["FAKE_CLOSEOUT_LOG"]).read_text().strip() == \
            "thesis-fedcrag-e0-closeout fedcrag-e0-closeout-20260827"


@pytest.mark.parametrize("fault", [
    "original-running", "snapshot-preexists", "clone-preexists",
    "candidate-disk-preexists", "snapshot-wrong-source", "snapshot-not-ready",
])
def test_preflight_identity_faults_write_no_new_resource(env, fault):
    env["FAKE_FAULT"] = fault
    completed = run_controller(env)
    assert completed.returncode != 0
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    assert not any(token in line for line in calls for token in (
        "snapshots create", "disks create", "instances create"))
    assert Path(env["FAKE_ORIGINAL_STATUS"]).read_text() == "TERMINATED"


@pytest.mark.parametrize(("fault", "expected_status"), [
    ("failed-disk-identity-mismatch", 20),
    ("partial-instance", 20),
    ("closeout-failure", 20),
    ("clone-stop-unverified", 70),
    ("original-stop-unverified", 70),
])
def test_failure_cleanup_is_identity_bound_and_fail_closed(
        env, fault, expected_status):
    env["FAKE_FAULT"] = fault
    completed = run_controller(env)
    assert completed.returncode == expected_status
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    if fault == "failed-disk-identity-mismatch":
        assert not any("disks delete" in line for line in calls)
    if fault == "partial-instance":
        assert not any("instances delete" in line for line in calls)
    assert Path(env["FAKE_ORIGINAL_STATUS"]).read_text() == "TERMINATED"


def test_success_retains_clone_disk_snapshot_and_uses_bash_32(env):
    completed = run_controller(env)
    assert completed.returncode == 0, completed.stderr
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text()
    assert "snapshots delete" not in calls
    assert "instances delete" not in calls
    assert "disks delete thesis-fedcrag-e0-closeout-boot-a" not in calls
    source = SCRIPT.read_text()
    assert "mapfile" not in source
    assert "declare -A" not in source
```

- [ ] **Step 2: Witness missing-script RED**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest tests/test_e0_cross_region_closeout.py -q -p no:cacheprovider
```

Expected: every test fails at the explicit missing-script assertion.

- [ ] **Step 3: Implement fixed identities, preflight, and cleanup trap**

```bash
readonly ORIGINAL_INSTANCE=thesis-fedcrag
readonly ORIGINAL_ZONE=asia-south1-c
readonly ORIGINAL_DISK=thesis-fedcrag-restored
readonly SNAPSHOT=fedcrag-e0-closeout-20260827
readonly CLONE_INSTANCE=thesis-fedcrag-e0-closeout
readonly ZONES="asia-southeast1-a asia-southeast1-b asia-southeast1-c"
readonly CRITICAL_SHUTDOWN=70
```

Require clean Git, exact project `rokkh-503122`, absent canonical destination,
original `TERMINATED`, disk `READY`, and no snapshot/clone/candidate disk.
Install EXIT/INT/TERM cleanup before the first cloud write.

- [ ] **Step 4: Create and validate the fresh snapshot**

```bash
gcloud compute snapshots create "$SNAPSHOT" \
  --project "$PROJECT" --source-disk "$ORIGINAL_DISK" \
  --source-disk-zone "$ORIGINAL_ZONE" --storage-location asia
```

Poll with a bounded counter until `READY`; then require exact source disk, size
200, and Asia storage location from a captured descriptor.

- [ ] **Step 5: Implement bounded zone allocation**

For each zone derive `thesis-fedcrag-e0-closeout-boot-${zone##*-}`. Create a
200 GiB `pd-balanced` disk from the fresh snapshot and verify it, then create:

```bash
gcloud compute instances create "$CLONE_INSTANCE" \
  --project "$PROJECT" --zone "$zone" --machine-type g2-standard-8 \
  --disk "name=$disk,boot=yes,auto-delete=no,mode=rw" \
  --network default --subnet default --network-tier PREMIUM \
  --maintenance-policy TERMINATE --restart-on-failure \
  --provisioning-model STANDARD \
  --service-account 139678593638-compute@developer.gserviceaccount.com \
  --scopes storage-ro,logging-write,monitoring-write,pubsub,service-management-ro,service-control,trace \
  --shielded-vtpm --shielded-integrity-monitoring --no-shielded-secure-boot
```

On capacity refusal, require instance absence and exact recoverable disk
identity before deleting that candidate disk and advancing. Any other failure
or partial instance aborts with evidence retained.

- [ ] **Step 6: Invoke closeout while the allocation remains live**

```bash
POST_E0_INSTANCE="$CLONE_INSTANCE" \
POST_E0_EXPECTED_SOURCE_SNAPSHOT="$SNAPSHOT" \
POST_E0_DEST=/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25 \
  /bin/bash "$SCRIPT_DIR/post_e0_closeout.sh"
```

The controller trap independently stops/verifies the clone and re-verifies the
original `TERMINATED`, even though closeout also stops its target.

- [ ] **Step 7: Run GREEN, full suite, and commit**

```bash
/bin/bash -n e0_cross_region_closeout.sh
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest tests/test_e0_cross_region_closeout.py \
  tests/test_post_e0_closeout.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
git diff --check
git add e0_cross_region_closeout.sh tests/test_e0_cross_region_closeout.py
git commit -m "ops: add fail-safe cross-region E0 closeout"
```

---

### Task 3: Apply the independent pre-spend review

**Files:**
- Review: `36dba69..HEAD`
- Record: `.superpowers/sdd/e0-cross-region-closeout-review.md`
- Modify only when a concrete finding has a witnessed regression test.

**Interfaces:**
- Consumes committed Tasks 1-2.
- Produces explicit approval for snapshot/allocation or blocking findings.

- [ ] **Step 1: Review safety and specification compliance**

Attempt counterexamples for name/filter injection, wrong-zone stops, stale
snapshots, partial allocation, orphaned disks, auto-delete, source-snapshot
substitution, interruption, shutdown-status masking, premature publication,
and Bash 3.2 behavior.

- [ ] **Step 2: Cure every valid finding test-first**

For each finding, add one regression, witness failure against the pre-fix code,
apply the smallest cure, rerun focused/full tests, and commit. Repeat until no
Critical, Important, or Minor finding remains.

- [ ] **Step 3: Run the final pre-spend gate**

```bash
/bin/bash -n post_e0_closeout.sh
/bin/bash -n e0_cross_region_closeout.sh
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest tests/test_e0_cross_region_closeout.py \
  tests/test_post_e0_closeout.py tests/test_validate_e0.py \
  tests/test_e0_manifest.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
git diff --check
git status --porcelain
```

Expected: all tests pass, checks are clean, and review approves live migration.

---

### Task 4: Execute migration and Task 7 closeout

**Files/artifacts:**
- Create GCP snapshot `fedcrag-e0-closeout-20260827`.
- Create one retained candidate disk and instance in the selected Singapore zone.
- Create canonical package `/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/`.
- Preserve controller logs under `/Users/turjo/Desktop/FedCRAG/post_e0_audit/migration-2026-08-27/`.

**Interfaces:**
- Consumes clean reviewed commit and authenticated project `rokkh-503122`.
- Produces exit-zero canonical closeout or exact failure evidence with both VMs stopped.

- [ ] **Step 1: Reconfirm external preconditions**

Require original `TERMINATED`, disk `READY`, snapshot/clone/candidate disks and
canonical destination/lock absent, quota unchanged, branch clean, at least 4
GiB local free, and no running controller.

- [ ] **Step 2: Run the exact controller**

```bash
POST_E0_DEST=/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25 \
  /bin/bash e0_cross_region_closeout.sh
```

Do not run ad-hoc create/delete commands around it. Report progress at
snapshot-ready, zone-attempt, clone-ready, validation, transfer, and shutdown.

- [ ] **Step 3: Verify live success independently**

Require controller exit 0, exact canonical layout, eleven zero validator exits,
matching four inventories, verified checksums, selected target and fresh
snapshot descriptors, and final VM record. Query GCP independently and require
both original and clone `TERMINATED`. On failure, preserve evidence and stop
without README promotion or E1 authorization.

---

### Task 5: Anchor, publish, and close Task 7

**Files:**
- Modify: `README.md:7-13,305-339`
- Modify: `tests/test_e0_manifest.py`
- Create: `docs/2026-08-27-e0-strengthened-closeout.md`
- Read: `/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25/PACKAGE_SHA256SUMS`

**Interfaces:**
- Consumes successful canonical package and independently stopped VMs.
- Produces externally anchored post-hoc digest and published audit branch.

- [ ] **Step 1: Add README/report RED assertions**

Require `strengthened-validated`, eleven rows, legacy per-round timing
unavailable, cross-region same-spec disclosure, post-hoc limitation, and no
paper-scale claim. Run the focused test and witness failure against README.

- [ ] **Step 2: Write closeout record and update status**

Compute SHA-256 of `PACKAGE_SHA256SUMS`. Record audit/execution commits,
snapshot/clone/zone identities, package digest, eleven-row result, VM final
states, and limitations. Update README without silently authorizing E1.

- [ ] **Step 3: Verify, review, commit, and push**

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/turjo/Desktop/FedCRAG/FedCRAG/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
git diff --check
git status --short
git add README.md tests/test_e0_manifest.py docs/2026-08-27-e0-strengthened-closeout.md
git commit -m "docs: anchor strengthened E0 closeout"
git push -u origin fedspan-post-e0-audit
```

Run a final review of `7325bf5..HEAD`, cure findings test-first, rerun the full
suite, and reconfirm both VMs stopped. Only then report Task 7 complete and
decide whether E1 planning can advance.
