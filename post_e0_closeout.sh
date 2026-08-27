#!/bin/bash
# Fail-safe preservation driver for the completed E0 correctness campaign.
#
# This driver only reads the historical result tree.  It never invokes the E0
# launcher, never writes below the remote artifact root, and publishes its
# local package only after validation, inventory equality, and a verified VM
# shutdown.  Keep this file compatible with GNU Bash 3.2 array features.
set -u
set -o pipefail

readonly DEFAULT_INSTANCE=thesis-fedcrag
readonly APPROVED_CLONE_INSTANCE=thesis-fedcrag-e0-closeout
readonly APPROVED_CLONE_SNAPSHOT=fedcrag-e0-closeout-20260827
INSTANCE=${POST_E0_INSTANCE-$DEFAULT_INSTANCE}
case "$INSTANCE" in
    ''|*[!a-z0-9-]*|-*|*-) printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2 ;;
esac
case "$INSTANCE" in
    [a-z]*) ;;
    *) printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2 ;;
esac
[ "${#INSTANCE}" -le 63 ] || { printf '%s\n' "invalid POST_E0_INSTANCE" >&2; exit 2; }
readonly INSTANCE
EXPECTED_SOURCE_SNAPSHOT=${POST_E0_EXPECTED_SOURCE_SNAPSHOT:-}
if [ "$INSTANCE" != "$DEFAULT_INSTANCE" ] && [ -z "$EXPECTED_SOURCE_SNAPSHOT" ]; then
    printf '%s\n' "non-default instance requires POST_E0_EXPECTED_SOURCE_SNAPSHOT" >&2
    exit 2
fi
if [ "$INSTANCE" != "$DEFAULT_INSTANCE" ] && \
    { [ "$INSTANCE" != "$APPROVED_CLONE_INSTANCE" ] || [ "$EXPECTED_SOURCE_SNAPSHOT" != "$APPROVED_CLONE_SNAPSHOT" ]; }; then
    printf '%s\n' "POST_E0_INSTANCE and POST_E0_EXPECTED_SOURCE_SNAPSHOT are not an approved clone identity" >&2
    exit 2
fi
REMOTE_ARTIFACT_ROOT=/home/turjo/FedCRAG_E0_RESULTS
readonly EXECUTION_COMMIT=7325bf56381c24c6a4af013688bdd417c95d7d7d
readonly EXECUTION_COMMIT_SHORT=7325bf56381c
readonly EXPECTED_ROWS=11
readonly CRITICAL_SHUTDOWN=70

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
DEST=${POST_E0_DEST:-/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25}
RETRY_SLEEP=${POST_E0_RETRY_SLEEP:-2}
SSH_READY_LIMIT=${POST_E0_SSH_READY_LIMIT:-30}
SSH_READY_SLEEP=${POST_E0_SSH_READY_SLEEP:-2}
PYTHON_BIN=${POST_E0_PYTHON:-python3}
REMOTE_TMP_ROOT=/tmp
EXECUTION_SOURCE_ROOT=/home/turjo/fedspan-e0
EXECUTION_PYTHON=/home/turjo/FedCRAG/.venv/bin/python
EXECUTION_INTERPRETER_PATH=$EXECUTION_PYTHON
VALIDATOR_BIN=
if [ "${POST_E0_TEST_MODE:-}" = 1 ]; then
    REMOTE_ARTIFACT_ROOT=${POST_E0_TEST_REMOTE_ROOT:?test remote root required}
    REMOTE_TMP_ROOT=${POST_E0_REMOTE_TMP:?test remote tmp required}
    EXECUTION_SOURCE_ROOT=${POST_E0_TEST_EXECUTION_SOURCE_ROOT:?test execution source root required}
    EXECUTION_PYTHON=${POST_E0_PYTHON:-python3}
    EXECUTION_INTERPRETER_PATH=${POST_E0_TEST_EXECUTION_INTERPRETER_PATH:?test execution interpreter path required}
fi
VALIDATOR_BIN=$EXECUTION_PYTHON
if [ "${POST_E0_TEST_MODE:-}" = 1 ]; then
    VALIDATOR_BIN=${POST_E0_VALIDATOR:?test validator required}
elif [ -n "${POST_E0_VALIDATOR:-}" ]; then
    printf '%s\n' "POST_E0_VALIDATOR is test-only" >&2; exit 2
fi
RETRY_LIMIT=3
ATTEMPT=
E0_PROJECT=
E0_ZONE=
CLEANUP_ARMED=0
CLONE_CLEANUP_ARMED=0
CLEANUP_DONE=0
VM_TOUCHED=0
PUBLICATION_LOCK=
PUBLICATION_LOCK_TOKEN=

say() { printf '%s\n' "$*" >&2; }

hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }

new_failure_path() {
    local stamp counter candidate
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    counter=0
    while [ "$counter" -lt 100 ]; do
        candidate="${DEST}.failed-${stamp}-${counter}"
        if [ ! -e "$candidate" ] && [ ! -L "$candidate" ] && mkdir "$candidate"; then
            printf '%s\n' "$candidate"; return
        fi
        if [ -e "$candidate" ] || [ -L "$candidate" ]; then
            counter=$((counter + 1))
        else
            return 1
        fi
    done
    return 1
}

preserve_attempt() {
    local failure
    [ -n "$ATTEMPT" ] && [ -d "$ATTEMPT" ] || return 0
    failure=$(new_failure_path) || { say "CRITICAL: cannot reserve failure path"; return 1; }
    mv "$ATTEMPT"/* "$failure/" && rmdir "$ATTEMPT" || {
        say "CRITICAL: cannot preserve failed attempt at $failure"
        return 1
    }
    ATTEMPT=
}

preserve_preflight_failure() {
    local message=$1 failure
    mkdir -p "$(dirname "$DEST")" || return 0
    failure=$(new_failure_path) || return 0
    mkdir "$failure/audit" || return 0
    printf '%s\n' "$message" > "$failure/audit/preflight_failure.txt"
}

create_attempt() {
    local stamp counter candidate
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    counter=0
    while [ "$counter" -lt 100 ]; do
        candidate="${DEST}.attempt-${stamp}-${counter}"
        if [ ! -e "$candidate" ] && [ ! -L "$candidate" ] && mkdir "$candidate"; then
            mkdir "$candidate/artifacts" "$candidate/audit" || return 1
            ATTEMPT=$candidate
            return 0
        fi
        counter=$((counter + 1))
    done
    return 1
}

verify_canonical_layout() {
    "$PYTHON_BIN" - "$DEST" <<'PY'
import os, sys
root = sys.argv[1]
if set(os.listdir(root)) != {"artifacts", "audit", "SOURCE_SHA256SUMS", "PACKAGE_SHA256SUMS"}:
    raise SystemExit("canonical top-level layout is not exact")
if not os.path.isdir(os.path.join(root, "artifacts")) or not os.path.isdir(os.path.join(root, "audit")):
    raise SystemExit("canonical directories are missing")
PY
}

acquire_publication_lock() {
    local candidate token
    candidate="${DEST}.publication-lock"
    token="$$-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir "$candidate" || return 1
    if ! (umask 077; printf '%s\n' "$token" > "$candidate/owner"); then
        rmdir "$candidate" 2>/dev/null || :
        return 1
    fi
    PUBLICATION_LOCK=$candidate
    PUBLICATION_LOCK_TOKEN=$token
}

release_publication_lock() {
    local owner
    [ -n "$PUBLICATION_LOCK" ] || return 0
    [ -f "$PUBLICATION_LOCK/owner" ] && [ ! -L "$PUBLICATION_LOCK/owner" ] || return 1
    IFS= read -r owner < "$PUBLICATION_LOCK/owner" || return 1
    [ "$owner" = "$PUBLICATION_LOCK_TOKEN" ] || return 1
    rm "$PUBLICATION_LOCK/owner" && rmdir "$PUBLICATION_LOCK" || return 1
    PUBLICATION_LOCK=
    PUBLICATION_LOCK_TOKEN=
}

atomic_promote() {
    "$PYTHON_BIN" - "$ATTEMPT" "$DEST" <<'PY'
import ctypes, os, sys
source, destination = [os.fsencode(value) for value in sys.argv[1:]]
if os.path.exists(destination) or os.path.islink(destination):
    raise SystemExit("destination already exists")
libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
function = libc.renameatx_np
function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
function.restype = ctypes.c_int
if function(-2, source, -2, destination, 0x00000004):
    raise OSError(ctypes.get_errno(), "renameatx_np RENAME_EXCL failed")
PY
}

get_status() {
    local output status
    output=$(gcloud compute instances describe "$INSTANCE" --project "$E0_PROJECT" \
        --zone "$E0_ZONE" --format='value(status)')
    status=$?
    [ "$status" -eq 0 ] || return 1
    case "$output" in
        RUNNING|TERMINATED) printf '%s\n' "$output" ;;
        *) return 1 ;;
    esac
}

wait_for_ssh_ready() {
    local attempt
    case "$SSH_READY_LIMIT" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$SSH_READY_LIMIT" -gt 0 ] || return 1
    attempt=1
    while [ "$attempt" -le "$SSH_READY_LIMIT" ]; do
        # A successful VM status only establishes that the instance is running;
        # this explicit noninteractive SSH probe establishes that SCP can use it.
        if gcloud compute ssh "$INSTANCE" --project "$E0_PROJECT" --zone "$E0_ZONE" \
            --quiet --ssh-flag=-oBatchMode=yes --ssh-flag=-oConnectTimeout=10 \
            --command true >/dev/null 2>&1; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -le "$SSH_READY_LIMIT" ] && sleep "$SSH_READY_SLEEP"
    done
    return 1
}

ensure_terminated() {
    local attempt status
    attempt=1
    while [ "$attempt" -le "$RETRY_LIMIT" ]; do
        gcloud compute instances stop "$INSTANCE" --project "$E0_PROJECT" \
            --zone "$E0_ZONE" --quiet >/dev/null 2>&1 || :
        status=$(get_status) || status=
        if [ "$status" = TERMINATED ]; then
            return 0
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -le "$RETRY_LIMIT" ] && sleep "$RETRY_SLEEP"
    done
    return 1
}

is_approved_clone_zone() {
    case "$1" in asia-southeast1-a|asia-southeast1-b|asia-southeast1-c) return 0 ;; esac
    return 1
}

get_status_in_zone() {
    local zone=$1 output status
    output=$(gcloud compute instances describe "$APPROVED_CLONE_INSTANCE" --project "$E0_PROJECT" \
        --zone "$zone" --format='value(status)')
    status=$?
    [ "$status" -eq 0 ] || return 1
    case "$output" in RUNNING|TERMINATED) printf '%s\n' "$output" ;; *) return 1 ;; esac
}

ensure_terminated_in_zone() {
    local zone=$1 attempt status
    attempt=1
    while [ "$attempt" -le "$RETRY_LIMIT" ]; do
        gcloud compute instances stop "$APPROVED_CLONE_INSTANCE" --project "$E0_PROJECT" \
            --zone "$zone" --quiet >/dev/null 2>&1 || return 1
        status=$(get_status_in_zone "$zone") || return 1
        [ "$status" = TERMINATED ] && return 0
        attempt=$((attempt + 1))
        [ "$attempt" -le "$RETRY_LIMIT" ] && sleep "$RETRY_SLEEP"
    done
    return 1
}

approved_clone_exists_in_zone() {
    local zone=$1 output status name found_zone
    output=$(gcloud compute instances list --project "$E0_PROJECT" --zones "$zone" \
        --filter="name=$APPROVED_CLONE_INSTANCE" --format='csv[no-heading](name,zone.basename())')
    status=$?
    [ "$status" -eq 0 ] || return 2
    while IFS=, read -r name found_zone; do
        [ "$name" = "$APPROVED_CLONE_INSTANCE" ] && [ "$found_zone" = "$zone" ] && return 0
    done <<EOF
$output
EOF
    return 1
}

ensure_approved_clone_terminated() {
    local zone result
    for zone in asia-southeast1-a asia-southeast1-b asia-southeast1-c; do
        approved_clone_exists_in_zone "$zone"
        result=$?
        [ "$result" -eq 1 ] && continue
        [ "$result" -eq 0 ] || return 1
        ensure_terminated_in_zone "$zone" || return 1
    done
}

cleanup() {
    local original=$?
    trap - EXIT INT TERM
    if [ "$CLONE_CLEANUP_ARMED" -eq 1 ] && [ "$CLEANUP_DONE" -eq 0 ]; then
        if ensure_approved_clone_terminated; then
            CLEANUP_DONE=1
        else
            say "CRITICAL: approved clone termination could not be verified"
            preserve_attempt
            exit "$CRITICAL_SHUTDOWN"
        fi
    elif [ "$CLEANUP_ARMED" -eq 1 ] && [ "$CLEANUP_DONE" -eq 0 ]; then
        if ensure_terminated; then
            CLEANUP_DONE=1
        else
            say "CRITICAL: VM termination could not be verified"
            preserve_attempt
            exit "$CRITICAL_SHUTDOWN"
        fi
    fi
    if [ "$original" -ne 0 ]; then
        preserve_attempt
    fi
    release_publication_lock || :
    exit "$original"
}

on_signal() { exit 130; }

fail() { say "closeout refused: $1"; exit "${2:-20}"; }

require_clean_worktree() {
    # The test-only escape is intentionally explicit and cannot be used by the
    # production command documented in the task brief.
    if [ "${POST_E0_TEST_MODE:-}" = 1 ]; then
        return 0
    fi
    local output status
    output=$(git -C "$SCRIPT_DIR" status --porcelain)
    status=$?
    [ "$status" -eq 0 ] || fail "cannot determine worktree status" 2
    [ -z "$output" ] || \
        fail "worktree is not clean" 2
}

discover_project() {
    local project_status
    E0_PROJECT="$(gcloud config get-value project)"
    project_status=$?
    [ "$project_status" -eq 0 ] || return 3
    [ -n "$E0_PROJECT" ] && [ "$E0_PROJECT" != "(unset)" ] || return 4
}

discover_target() {
    local discovery_status name zone

    # Deliberately capture the command status before interpreting stdout.
    E0_ZONE_OUTPUT="$(gcloud compute instances list --project "$E0_PROJECT" \
        --filter="name=$INSTANCE" --format='csv[no-heading](name,zone.basename())')"
    discovery_status=$?
    [ "$discovery_status" -eq 0 ] || return 3
    E0_ZONES=()
    while IFS=, read -r name zone; do
        [ "$name" = "$INSTANCE" ] && [ -n "$zone" ] || continue
        if [ "$INSTANCE" = "$APPROVED_CLONE_INSTANCE" ] && ! is_approved_clone_zone "$zone"; then
            return 4
        fi
        E0_ZONES[${#E0_ZONES[@]}]="$zone"
    done <<EOF
$E0_ZONE_OUTPUT
EOF
    [ "${#E0_ZONES[@]}" -eq 1 ] || return 4
    E0_ZONE=${E0_ZONES[0]}
    return 0
}

capture_and_verify_target() {
    local disk snapshot
    gcloud compute instances describe "$INSTANCE" --project "$E0_PROJECT" \
        --zone "$E0_ZONE" --format=json > "$ATTEMPT/audit/target_instance.json" || return 1
    disk=$("$PYTHON_BIN" - "$ATTEMPT/audit/target_instance.json" "$INSTANCE" "$E0_ZONE" <<'PY'
import json, sys

path, instance, zone = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))

def basename(value):
    if not isinstance(value, str) or not value:
        raise SystemExit("target descriptor has invalid resource name")
    return value.rstrip("/").rsplit("/", 1)[-1]

if data.get("name") != instance or basename(data.get("zone")) != zone:
    raise SystemExit("target instance name or zone does not match discovery")
if instance == "thesis-fedcrag-e0-closeout" and zone not in {"asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c"}:
    raise SystemExit("approved clone is not in an approved Singapore zone")
if basename(data.get("machineType")) != "g2-standard-8":
    raise SystemExit("target instance machine type is not g2-standard-8")
accelerators = data.get("guestAccelerators")
if not isinstance(accelerators, list) or len(accelerators) != 1:
    raise SystemExit("target instance does not have one L4 accelerator")
accelerator = accelerators[0]
if not isinstance(accelerator, dict) or basename(accelerator.get("acceleratorType")) != "nvidia-l4" or accelerator.get("acceleratorCount") != 1:
    raise SystemExit("target instance does not have one nvidia-l4")
if data.get("scheduling", {}).get("provisioningModel") != "STANDARD":
    raise SystemExit("target instance provisioning model is not STANDARD")
disks = data.get("disks")
boot_disks = [item for item in disks if isinstance(item, dict) and item.get("boot") is True and item.get("type") == "PERSISTENT"] if isinstance(disks, list) else []
if len(boot_disks) != 1 or boot_disks[0].get("autoDelete") is not False:
    raise SystemExit("target instance does not have one persistent boot disk")
print(basename(boot_disks[0].get("source")))
PY
) || return 1
    [ -n "$disk" ] || return 1
    gcloud compute disks describe "$disk" --project "$E0_PROJECT" --zone "$E0_ZONE" \
        --format=json > "$ATTEMPT/audit/target_disk.json" || return 1
    snapshot=$("$PYTHON_BIN" - "$ATTEMPT/audit/target_disk.json" "$EXPECTED_SOURCE_SNAPSHOT" <<'PY'
import json, sys

path, expected_snapshot = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))

def basename(value):
    if not isinstance(value, str) or not value:
        raise SystemExit("target descriptor has invalid resource name")
    return value.rstrip("/").rsplit("/", 1)[-1]

if data.get("status") != "READY" or str(data.get("sizeGb")) != "200" or basename(data.get("type")) != "pd-balanced":
    raise SystemExit("target boot disk is not READY 200GB pd-balanced")
if expected_snapshot:
    source_snapshot = basename(data.get("sourceSnapshot"))
    if source_snapshot != expected_snapshot:
        raise SystemExit("target boot disk source snapshot does not match expected clone snapshot")
    print(source_snapshot)
PY
) || return 1
    if [ -n "$EXPECTED_SOURCE_SNAPSHOT" ]; then
        [ -n "$snapshot" ] || return 1
        gcloud compute snapshots describe "$snapshot" --project "$E0_PROJECT" \
            --format=json > "$ATTEMPT/audit/target_snapshot.json" || return 1
        "$PYTHON_BIN" - "$ATTEMPT/audit/target_snapshot.json" "$snapshot" <<'PY' || return 1
import json, sys

path, expected_snapshot = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
if data.get("name") != expected_snapshot or data.get("status") != "READY" or str(data.get("diskSizeGb")) != "200":
    raise SystemExit("target source snapshot is not the expected READY 200GB snapshot")
source_disk = data.get("sourceDisk")
if not isinstance(source_disk, str) or not source_disk.endswith("/zones/asia-south1-c/disks/thesis-fedcrag-restored"):
    raise SystemExit("target source snapshot does not originate from thesis-fedcrag-restored")
PY
    fi
}

assert_regular_tree() {
    local root=$1
    "$PYTHON_BIN" - "$root" <<'PY'
import os, stat, sys
root = sys.argv[1]
root_mode = os.lstat(root).st_mode
if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
    raise SystemExit("unsafe artifact root: " + root)
for base, dirs, files in os.walk(root, followlinks=False):
    for name in dirs + files:
        path = os.path.join(base, name)
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise SystemExit("unsafe artifact entry: " + path)
PY
}

inventory_tree() {
    # Python path handling and byte-sort avoid newline/whitespace ambiguity;
    # NUL cannot occur in a POSIX filename.  The JSON form preserves all other
    # characters and includes the exact relative path, byte size, and digest.
    "$PYTHON_BIN" - "$1" <<'PY'
import hashlib, json, os, stat, sys
root = os.path.abspath(sys.argv[1])
entries = []
for base, dirs, files in os.walk(root, followlinks=False):
    dirs.sort(key=os.fsencode); files.sort(key=os.fsencode)
    for name in files:
        path = os.path.join(base, name)
        item = os.lstat(path)
        if not stat.S_ISREG(item.st_mode):
            raise SystemExit("unsafe inventory entry: " + path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        entries.append((os.fsencode(os.path.relpath(path, root)), {
            "path": os.path.relpath(path, root), "size": item.st_size,
            "sha256": digest.hexdigest()}))
for _, entry in sorted(entries):
    print(json.dumps(entry, sort_keys=True, ensure_ascii=True))
PY
}

copy_regular_tree() {
    local source=$1 target=$2
    assert_regular_tree "$source" || return 1
    mkdir -p "$target" || return 1
    (cd "$source" && find . -type f -print0) | while IFS= read -r -d '' relative; do
        mkdir -p "$target/$(dirname "$relative")" || exit 1
        cp -p "$source/$relative" "$target/$relative" || exit 1
    done
}

check_identity_and_rows() {
    local root=$1
    [ -f "$root/COMPLETE.json" ] && [ -f "$root/manifest.json" ] || return 1
    "$PYTHON_BIN" - "$root" "$EXECUTION_COMMIT_SHORT" "$EXPECTED_ROWS" <<'PY'
import glob, json, os, sys
root, expected_commit, expected_rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
complete = json.load(open(os.path.join(root, "COMPLETE.json"), encoding="utf-8"))
manifest = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
rows = manifest.get("rows")
if complete.get("commit") != expected_commit or manifest.get("commit") != expected_commit:
    raise SystemExit("completion or manifest commit is not the frozen short identity")
if not isinstance(rows, list) or len(rows) != expected_rows:
    raise SystemExit("manifest does not contain eleven rows")
run_ids = [row.get("run_id") for row in rows if isinstance(row, dict)]
if len(run_ids) != expected_rows or len(set(run_ids)) != expected_rows or not all(run_ids):
    raise SystemExit("manifest row identities are invalid")
validated = complete.get("validated_rows")
if not isinstance(validated, list) or validated != run_ids:
    raise SystemExit("completion validated rows do not exactly bind manifest rows")
for run_id in run_ids:
    paths = glob.glob(os.path.join(root, run_id, "federated_*.json"))
    if len(paths) != 1:
        raise SystemExit("row lacks one federated result: " + run_id)
    result = json.load(open(paths[0], encoding="utf-8"))
    if result.get("commit") != expected_commit:
        raise SystemExit("result commit is not the frozen short identity: " + run_id)
PY
}

resolve_execution_commit() {
    local repository=$1 resolved
    resolved=$(git -C "$repository" rev-parse "${EXECUTION_COMMIT_SHORT}^{commit}") || return 1
    [ "$resolved" = "$EXECUTION_COMMIT" ]
}

record_and_validate_execution_source() {
    local head head_status source_status source_status_status
    head=$(git -C "$EXECUTION_SOURCE_ROOT" rev-parse HEAD)
    head_status=$?
    source_status=$(git -C "$EXECUTION_SOURCE_ROOT" status --porcelain)
    source_status_status=$?
    {
        printf 'execution_source_root=%s\n' "$EXECUTION_SOURCE_ROOT"
        printf 'execution_interpreter_path=%s\n' "$EXECUTION_INTERPRETER_PATH"
        printf 'execution_source_head_status=%s\n' "$head_status"
        printf 'execution_source_head=%s\n' "$head"
        printf 'execution_source_status_status=%s\n' "$source_status_status"
        printf 'execution_source_status_porcelain=%s\n' "$source_status"
    } >> "$1" || return 1
    [ "$head_status" -eq 0 ] || return 1
    [ "$source_status_status" -eq 0 ] || return 1
    [ "$head" = "$EXECUTION_COMMIT" ] || return 1
    [ -z "$source_status" ] || return 1
}

write_remote_worker() {
    # The production worker is this same audited script in an explicit remote
    # mode.  It is copied to /tmp and never checked out over the E0 execution
    # tree.  The test mode above avoids SSH entirely.
    cp "$SCRIPT_DIR/post_e0_closeout.sh" "$ATTEMPT/audit/post_e0_closeout.sh"
    chmod 700 "$ATTEMPT/audit/post_e0_closeout.sh"
}

run_production_snapshot() {
    local audit_commit bundle bundle_hash remote_stage worker_status fetch_status
    audit_commit=$(git -C "$SCRIPT_DIR" rev-parse HEAD) || return 1
    bundle="$ATTEMPT/audit/fedspan-post-e0-audit-${audit_commit}.bundle"
    [ "$(git -C "$SCRIPT_DIR" rev-parse HEAD)" = "$audit_commit" ] || return 1
    git -C "$SCRIPT_DIR" bundle create "$bundle" HEAD || return 1
    bundle_hash=$(hash_file "$bundle") || return 1
    remote_stage="$REMOTE_TMP_ROOT/fedspan-post-e0-stage-${audit_commit}"
    write_remote_worker || return 1
    {
        printf 'audit_commit=%s\n' "$audit_commit"
        printf 'bundle_sha256=%s\n' "$bundle_hash"
        printf 'command=%s\n' "validate_e0.py (remote clean clone only)"
        /bin/bash --version
    } > "$ATTEMPT/audit/environment.txt"
    gcloud compute scp "$bundle" "$ATTEMPT/audit/post_e0_closeout.sh" \
        "$INSTANCE:$REMOTE_TMP_ROOT/" --project "$E0_PROJECT" --zone "$E0_ZONE" || return 1
    # The remote subcommand refuses unsafe entries, copies only inventoried
    # regular files to /tmp, and runs the strengthened validator from the
    # fresh audit-commit clone.  Its outputs remain in /tmp until copied back.
    printf '%s\n' "POST_E0_REMOTE_WORKER=1 bash $REMOTE_TMP_ROOT/post_e0_closeout.sh '$audit_commit' '$bundle_hash' '$EXECUTION_COMMIT'" \
        > "$ATTEMPT/audit/remote_command.txt" || return 1
    gcloud compute ssh "$INSTANCE" --project "$E0_PROJECT" --zone "$E0_ZONE" \
        --command "$(cat "$ATTEMPT/audit/remote_command.txt")"
    worker_status=$?
    # Pull audit evidence even after a scientific refusal, before preserving it.
    mkdir "$ATTEMPT/remote_transfer" || return 1
    gcloud compute scp --recurse "$INSTANCE:$remote_stage/audit" "$ATTEMPT/remote_transfer" \
        --project "$E0_PROJECT" --zone "$E0_ZONE"
    fetch_status=$?
    [ "$fetch_status" -eq 0 ] || return 1
    [ -d "$ATTEMPT/remote_transfer/audit" ] || return 1
    assert_regular_tree "$ATTEMPT/remote_transfer/audit" || return 1
    mv "$ATTEMPT/remote_transfer/audit/"* "$ATTEMPT/audit/" || return 1
    rmdir "$ATTEMPT/remote_transfer/audit" "$ATTEMPT/remote_transfer" || return 1
    [ "$worker_status" -eq 0 ] || return "$worker_status"
    mkdir "$ATTEMPT/remote_transfer" || return 1
    gcloud compute scp --recurse "$INSTANCE:$remote_stage/artifacts" "$ATTEMPT/remote_transfer" \
        --project "$E0_PROJECT" --zone "$E0_ZONE" || return 1
    [ -d "$ATTEMPT/remote_transfer/artifacts" ] && [ -d "$ATTEMPT/audit" ] || return 1
    assert_regular_tree "$ATTEMPT/remote_transfer/artifacts" || return 1
    mv "$ATTEMPT/remote_transfer/artifacts/"* "$ATTEMPT/artifacts/" || return 1
    rmdir "$ATTEMPT/remote_transfer/artifacts" "$ATTEMPT/remote_transfer" || return 1
    inventory_tree "$ATTEMPT/artifacts" > "$ATTEMPT/audit/local_inventory.jsonl" || return 1
    cmp "$ATTEMPT/audit/source_pre_inventory.jsonl" "$ATTEMPT/audit/source_post_inventory.jsonl" || return 1
    cmp "$ATTEMPT/audit/source_pre_inventory.jsonl" "$ATTEMPT/audit/staged_inventory.jsonl" || return 1
    cmp "$ATTEMPT/audit/source_pre_inventory.jsonl" "$ATTEMPT/audit/local_inventory.jsonl" || return 1
}

remote_worker() {
    local audit_commit bundle_hash execution_commit stage repo bundle rows run validator_status
    [ "$#" -eq 3 ] || return 2
    audit_commit=$1
    bundle_hash=$2
    execution_commit=$3
    case "$audit_commit" in *[!0-9a-f]*|???????????????????????????????????????) return 2 ;; esac
    [ "${#audit_commit}" -eq 40 ] || return 2
    case "$bundle_hash" in *[!0-9a-f]*|"") return 2 ;; esac
    [ "${#bundle_hash}" -eq 64 ] || return 2
    [ "$execution_commit" = "$EXECUTION_COMMIT" ] || return 2
    case "$REMOTE_TMP_ROOT" in /|""|*".."*) return 2 ;; esac
    [ -d "$REMOTE_TMP_ROOT" ] && [ ! -L "$REMOTE_TMP_ROOT" ] || return 2
    stage="$REMOTE_TMP_ROOT/fedspan-post-e0-stage-${audit_commit}"
    repo="$REMOTE_TMP_ROOT/fedspan-post-e0-clone-${audit_commit}"
    bundle="$REMOTE_TMP_ROOT/fedspan-post-e0-audit-${audit_commit}.bundle"
    [ ! -e "$stage" ] && [ ! -L "$stage" ] && [ ! -e "$repo" ] && [ ! -L "$repo" ] || return 2
    mkdir "$stage" && mkdir "$stage/artifacts" "$stage/audit" || return 1
    [ -f "$bundle" ] && [ ! -L "$bundle" ] || return 1
    [ "$(hash_file "$bundle")" = "$bundle_hash" ] || return 1
    assert_regular_tree "$REMOTE_ARTIFACT_ROOT" || return 1
    inventory_tree "$REMOTE_ARTIFACT_ROOT" > "$stage/audit/source_pre_inventory.jsonl" || return 1
    [ "${POST_E0_TEST_FAIL:-}" != copy ] || return 1
    copy_regular_tree "$REMOTE_ARTIFACT_ROOT" "$stage/artifacts" || return 1
    check_identity_and_rows "$stage/artifacts" || return 1
    inventory_tree "$REMOTE_ARTIFACT_ROOT" > "$stage/audit/source_post_inventory.jsonl" || return 1
    inventory_tree "$stage/artifacts" > "$stage/audit/staged_inventory.jsonl" || return 1
    cmp "$stage/audit/source_pre_inventory.jsonl" "$stage/audit/source_post_inventory.jsonl" || return 1
    cmp "$stage/audit/source_pre_inventory.jsonl" "$stage/audit/staged_inventory.jsonl" || return 1
    [ "${POST_E0_TEST_FAIL:-}" != checksum ] || return 1
    git clone "$bundle" "$repo" || return 1
    [ "$(git -C "$repo" rev-parse HEAD)" = "$audit_commit" ] || return 1
    resolve_execution_commit "$repo" || return 1
    printf 'audit_clone=%s\n' "$repo" \
        > "$stage/audit/execution_source_identity.txt" || return 1
    record_and_validate_execution_source \
        "$stage/audit/execution_source_identity.txt" || return 1
    rows=$("$PYTHON_BIN" - "$stage/artifacts/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for row in value["rows"]:
    print(row["run_id"])
PY
) || return 1
    : > "$stage/audit/validator_output.jsonl"
    : > "$stage/audit/validator_command.txt"
    : > "$stage/audit/validator_exit_status.tsv"
    while IFS= read -r run; do
        [ -n "$run" ] || continue
        printf '%q ' "$VALIDATOR_BIN" "$repo/validate_e0.py" "$stage/artifacts/$run" \
            --manifest "$stage/artifacts/manifest.json" --run_id "$run" \
            --execution_source_root "$EXECUTION_SOURCE_ROOT" \
            --execution_interpreter_path "$EXECUTION_INTERPRETER_PATH" >> "$stage/audit/validator_command.txt"
        printf '\n' >> "$stage/audit/validator_command.txt"
        "$VALIDATOR_BIN" "$repo/validate_e0.py" "$stage/artifacts/$run" \
            --manifest "$stage/artifacts/manifest.json" --run_id "$run" \
            --execution_source_root "$EXECUTION_SOURCE_ROOT" \
            --execution_interpreter_path "$EXECUTION_INTERPRETER_PATH" \
            >> "$stage/audit/validator_output.jsonl" 2> "$stage/audit/validator-${run}.stderr"
        validator_status=$?
        printf '%s\t%s\n' "$run" "$validator_status" >> "$stage/audit/validator_exit_status.tsv"
        [ "$validator_status" -eq 0 ] || {
                printf 'scientific validation failed for %s (exit %s)\n' "$run" "$validator_status" \
                    > "$stage/audit/validation_failure.txt"
                return 1
            }
    done <<EOF
$rows
EOF
    "$EXECUTION_PYTHON" - <<'PY' > "$stage/audit/runtime_versions.txt" || return 1
import numpy, sys, torch
print("python=" + sys.version.replace("\n", " "))
print("torch=" + torch.__version__)
print("numpy=" + numpy.__version__)
PY
    "$PYTHON_BIN" - "$stage/audit/validator_output.jsonl" "$stage/artifacts/status.tsv" \
        "$stage/artifacts/manifest.json" \
        "$stage/audit/validation_summary.json" <<'PY' || return 1
import datetime, json, math, os, sys
reports_path, status_path, manifest_path, output = sys.argv[1:]
reports = [json.loads(line) for line in open(reports_path, encoding="utf-8") if line.strip()]
if len(reports) != 11:
    raise SystemExit("validator did not produce eleven reports")
required = ("manifest_verified", "dataset_content_verified", "runtime_provenance_verified",
            "fedspan_direction_residuals", "rawmaxmin_direction_residuals",
            "continuity_boundaries_checked", "aggregate_recomputation_worst_tolerance_ratio")
if any(not all(key in report for key in required) for report in reports):
    raise SystemExit("validator report schema is incomplete")
if any(not (report["manifest_verified"] and report["dataset_content_verified"]
            and report["runtime_provenance_verified"]) for report in reports):
    raise SystemExit("validator verification flags are not all true")
rows = [row["run_id"] for row in json.load(open(manifest_path))["rows"]]
seen = {}
for line in open(status_path, encoding="utf-8"):
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 4 or fields[1] != "VALIDATED":
        continue
    run_id, _, seconds, timestamp = fields
    if run_id not in rows or run_id in seen:
        raise SystemExit("status records do not uniquely bind validated rows")
    try: seen[run_id] = float(seconds)
    except ValueError: raise SystemExit("status runtime is not numeric")
    if not math.isfinite(seen[run_id]) or seen[run_id] <= 0:
        raise SystemExit("status runtime must be finite and positive")
    try:
        parsed = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        if parsed.tzinfo is not None: raise ValueError
    except ValueError: raise SystemExit("status timestamp is not UTC RFC3339 seconds")
if set(seen) != set(rows) or len(seen) != 11:
    raise SystemExit("status records do not cover every validated row")
summary = {"validated_rows": len(reports), "reports": reports,
           "fedspan_direction_residuals": [r["fedspan_direction_residuals"] for r in reports],
           "rawmaxmin_direction_residuals": [r["rawmaxmin_direction_residuals"] for r in reports],
           "continuity_boundaries_checked": [r["continuity_boundaries_checked"] for r in reports],
           "aggregate_recomputation_worst_tolerance_ratio": [r["aggregate_recomputation_worst_tolerance_ratio"] for r in reports],
           "measured_total_row_runtimes": [{"run_id": run_id, "seconds": seen[run_id]} for run_id in rows],
           "legacy_schema_v1_per_round": "unavailable"}
json.dump(summary, open(output, "w", encoding="utf-8"), sort_keys=True)
PY
    "$PYTHON_BIN" - "$stage/audit/validation_summary.json" \
        "$stage/audit/2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md" <<'PY' || return 1
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as out:
    out.write("# E0 strengthened validation closeout\n\n")
    out.write("Validated rows: %d\n\n" % summary["validated_rows"])
    out.write("FedSpan direction residuals: %s\n\n" % summary["fedspan_direction_residuals"])
    out.write("RawMaxMin direction residuals: %s\n\n" % summary["rawmaxmin_direction_residuals"])
    out.write("Continuity boundaries checked: %s\n\n" % summary["continuity_boundaries_checked"])
    out.write("Aggregate tolerance ratios: %s\n\n" % summary["aggregate_recomputation_worst_tolerance_ratio"])
    out.write("Measured total row runtimes: %s\n\n" % summary["measured_total_row_runtimes"])
    out.write("Schema-v1 per-round timing is unavailable. This is a post-hoc internally consistent inventory, not a signed historical attestation. It does not support paper-scale efficacy claims.\n")
PY
}

write_reports() {
    "$PYTHON_BIN" - "$ATTEMPT/audit/source_pre_inventory.jsonl" \
        "$ATTEMPT/audit/local_inventory.jsonl" "$ATTEMPT/SOURCE_SHA256SUMS" <<'PY' || return 1
import json, sys
remote_path, local_path, output = sys.argv[1:]
remote = [json.loads(line) for line in open(remote_path, encoding="utf-8")]
local = [json.loads(line) for line in open(local_path, encoding="utf-8")]
if remote != local:
    raise SystemExit("remote/local source inventories differ")
with open(output, "w", encoding="utf-8") as handle:
    for item in remote:
        handle.write("{sha256}  {size}  {path}\n".format(**item))
PY
    [ -f "$ATTEMPT/audit/validation_summary.json" ] || return 1
    [ -f "$ATTEMPT/audit/2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md" ] || \
        printf '%s\n' "Post-hoc internally consistent preservation; not a signed historical attestation." \
            > "$ATTEMPT/audit/2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md"
    /bin/bash --version > "$ATTEMPT/audit/bash_version.txt" || return 1
    "$PYTHON_BIN" - "$ATTEMPT/artifacts/manifest.json" \
        "$ATTEMPT/audit/staged_manifest_digest.json" <<'PY' || return 1
import hashlib, json, os, sys
path, output = sys.argv[1:]
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump({"path": "artifacts/manifest.json", "size": os.path.getsize(path),
           "sha256": digest}, open(output, "w", encoding="utf-8"),
          sort_keys=True)
PY
}

verify_manifest_digest() {
    "$PYTHON_BIN" - "$ATTEMPT/artifacts/manifest.json" \
        "$ATTEMPT/audit/staged_manifest_digest.json" <<'PY'
import hashlib, json, os, sys
path, record_path = sys.argv[1:]
record = json.load(open(record_path, encoding="utf-8"))
actual = {"path": "artifacts/manifest.json", "size": os.path.getsize(path),
          "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
raise SystemExit(0 if actual == record else "exported manifest digest changed")
PY
}

package_and_verify() {
    "$PYTHON_BIN" - "$ATTEMPT" <<'PY' || return 1
import hashlib, json, os, sys
root = sys.argv[1]
paths = ["SOURCE_SHA256SUMS"]
for base, dirs, files in os.walk(os.path.join(root, "audit")):
    dirs.sort(key=os.fsencode); files.sort(key=os.fsencode)
    for name in files:
        paths.append(os.path.relpath(os.path.join(base, name), root))
with open(os.path.join(root, "PACKAGE_SHA256SUMS"), "w", encoding="utf-8") as output:
    for relative in sorted(paths, key=os.fsencode):
        path = os.path.join(root, relative)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        output.write(json.dumps({"path": relative, "sha256": digest}, sort_keys=True) + "\n")
PY
    "$PYTHON_BIN" - "$ATTEMPT" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
for line in open(os.path.join(root, "PACKAGE_SHA256SUMS"), encoding="utf-8"):
    record = json.loads(line)
    path = os.path.join(root, record["path"])
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != record["sha256"]:
        raise SystemExit("package checksum mismatch: " + record["path"])
PY
}

main() {
    local discovery_code before after
    require_clean_worktree
    discover_project
    discovery_code=$?
    if [ "$discovery_code" -ne 0 ]; then
        preserve_preflight_failure "project discovery failed"
        exit "$discovery_code"
    fi

    if [ "$INSTANCE" = "$APPROVED_CLONE_INSTANCE" ]; then
        trap cleanup EXIT
        trap on_signal INT TERM
        [ ! -e "$DEST" ] && [ ! -L "$DEST" ] || fail "canonical destination already exists" 5
        mkdir -p "$(dirname "$DEST")" || fail "cannot create preservation parent"
        acquire_publication_lock || fail "cannot reserve sibling publication lock" 5
        create_attempt || fail "cannot create exclusive attempt"
        CLONE_CLEANUP_ARMED=1
        discover_target
        discovery_code=$?
        [ "$discovery_code" -eq 0 ] || fail "project/zone discovery failed" "$discovery_code"
        CLEANUP_ARMED=1
    else
    discover_target
    discovery_code=$?
    if [ "$discovery_code" -ne 0 ]; then
        preserve_preflight_failure "project/zone discovery failed"
        exit "$discovery_code"
    fi
    trap cleanup EXIT
    trap on_signal INT TERM
    [ ! -e "$DEST" ] && [ ! -L "$DEST" ] || fail "canonical destination already exists" 5
    mkdir -p "$(dirname "$DEST")" || fail "cannot create preservation parent"
    acquire_publication_lock || fail "cannot reserve sibling publication lock" 5
    create_attempt || fail "cannot create exclusive attempt"
    CLEANUP_ARMED=1
    fi
    capture_and_verify_target || fail "target configuration descriptor gate failed"

    before=$(get_status) || fail "initial VM status query was malformed or failed"
    printf 'before=%s\n' "$before" > "$ATTEMPT/audit/vm_state.txt" || fail "cannot record initial VM state"
    if [ "$before" = TERMINATED ]; then
        gcloud compute instances start "$INSTANCE" --project "$E0_PROJECT" \
            --zone "$E0_ZONE" --quiet || fail "could not start terminated VM"
        VM_TOUCHED=1
        after=$(get_status) || fail "post-start VM status query was malformed or failed"
        [ "$after" = RUNNING ] || fail "terminated VM did not become RUNNING"
    else
        after=$before
    fi
    printf 'after=%s\n' "$after" >> "$ATTEMPT/audit/vm_state.txt" || fail "cannot record post-start VM state"
    if [ "$before" = TERMINATED ]; then
        wait_for_ssh_ready || fail "VM SSH did not become ready after bounded probes"
    fi

    run_production_snapshot || fail "remote snapshot/validation gate failed"
    write_reports || fail "report gate failed"
    if ensure_terminated; then
        CLEANUP_DONE=1
    else
        # The bounded attempt has already happened; do not repeat it in EXIT.
        CLEANUP_DONE=1
        fail "explicit shutdown could not be verified" "$CRITICAL_SHUTDOWN"
    fi
    printf 'final=TERMINATED\n' >> "$ATTEMPT/audit/vm_state.txt" || fail "cannot record final VM state"
    package_and_verify || fail "package checksum gate failed"
    verify_manifest_digest || fail "exported manifest digest gate failed"
    [ ! -e "$DEST" ] && [ ! -L "$DEST" ] || fail "canonical destination appeared before promotion"
    atomic_promote || fail "atomic no-replace publication failed"
    ATTEMPT=
    verify_canonical_layout || fail "canonical layout verification failed"
    release_publication_lock || fail "cannot release publication lock"
}

if [ "${POST_E0_REMOTE_WORKER:-}" = 1 ]; then
    remote_worker "$@"
else
    main "$@"
fi
