#!/bin/bash
# Fail-safe preservation driver for the completed E0 correctness campaign.
#
# This driver only reads the historical result tree.  It never invokes the E0
# launcher, never writes below the remote artifact root, and publishes its
# local package only after validation, inventory equality, and a verified VM
# shutdown.  Keep this file compatible with GNU Bash 3.2 array features.
set -u
set -o pipefail

readonly INSTANCE=thesis-fedcrag
readonly REMOTE_ARTIFACT_ROOT=/home/turjo/FedCRAG_E0_RESULTS
readonly EXECUTION_COMMIT=7325bf56381c24c6a4af013688bdd417c95d7d7d
readonly EXECUTION_COMMIT_SHORT=7325bf56381c
readonly EXPECTED_ROWS=11
readonly CRITICAL_SHUTDOWN=70

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
DEST=${POST_E0_DEST:-/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25}
RETRY_SLEEP=${POST_E0_RETRY_SLEEP:-2}
PYTHON_BIN=${POST_E0_PYTHON:-python3}
RETRY_LIMIT=3
ATTEMPT=
E0_PROJECT=
E0_ZONE=
CLEANUP_ARMED=0
CLEANUP_DONE=0
VM_TOUCHED=0

say() { printf '%s\n' "$*" >&2; }

hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }

new_failure_path() {
    local stamp counter candidate
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    counter=0
    while :; do
        candidate="${DEST}.failed-${stamp}-${counter}"
        [ ! -e "$candidate" ] && { printf '%s\n' "$candidate"; return; }
        counter=$((counter + 1))
    done
}

preserve_attempt() {
    local failure
    [ -n "$ATTEMPT" ] && [ -d "$ATTEMPT" ] || return 0
    failure=$(new_failure_path)
    mv "$ATTEMPT" "$failure"
    ATTEMPT=
}

preserve_preflight_failure() {
    local message=$1 failure
    mkdir -p "$(dirname "$DEST")" || return 0
    failure=$(new_failure_path)
    mkdir -p "$failure/audit" || return 0
    printf '%s\n' "$message" > "$failure/audit/preflight_failure.txt"
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

cleanup() {
    local original=$?
    trap - EXIT INT TERM
    if [ "$CLEANUP_ARMED" -eq 1 ] && [ "$CLEANUP_DONE" -eq 0 ]; then
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
    [ -z "$(git -C "$SCRIPT_DIR" status --porcelain)" ] || \
        fail "worktree is not clean" 2
}

discover_target() {
    local discovery_status zone
    E0_PROJECT="$(gcloud config get-value project)"
    [ -n "$E0_PROJECT" ] && [ "$E0_PROJECT" != "(unset)" ] || return 4

    # Deliberately capture the command status before interpreting stdout.
    E0_ZONE_OUTPUT="$(gcloud compute instances list --project "$E0_PROJECT" \
        --filter='name=thesis-fedcrag' --format='value(zone.basename())')"
    discovery_status=$?
    [ "$discovery_status" -eq 0 ] || return 3
    E0_ZONES=()
    while IFS= read -r zone; do
        [ -n "$zone" ] && E0_ZONES[${#E0_ZONES[@]}]="$zone"
    done <<EOF
$E0_ZONE_OUTPUT
EOF
    [ "${#E0_ZONES[@]}" -eq 1 ] || return 4
    E0_ZONE=${E0_ZONES[0]}
    return 0
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
    local root=$1 rows
    [ -f "$root/COMPLETE.json" ] && [ -f "$root/manifest.json" ] || return 1
    grep -F "$EXECUTION_COMMIT_SHORT" "$root/COMPLETE.json" >/dev/null || return 1
    grep -F "$EXECUTION_COMMIT" "$root/COMPLETE.json" >/dev/null || return 1
    grep -F "$EXECUTION_COMMIT_SHORT" "$root/manifest.json" >/dev/null || return 1
    grep -F "$EXECUTION_COMMIT" "$root/manifest.json" >/dev/null || return 1
    rows=$("$PYTHON_BIN" - "$root/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
rows = value.get("rows", value if isinstance(value, list) else [])
print(len(rows))
PY
)
    [ "$rows" = "$EXPECTED_ROWS" ]
}

run_test_snapshot() {
    local source=$1 validator=$2
    [ -n "$source" ] && [ -d "$source" ] || return 1
    assert_regular_tree "$source" || return 1
    inventory_tree "$source" > "$ATTEMPT/audit/remote_pre_inventory.jsonl" || return 1
    if [ "${POST_E0_TEST_FAIL:-}" = copy ]; then return 1; fi
    copy_regular_tree "$source" "$ATTEMPT/artifacts" || return 1
    check_identity_and_rows "$ATTEMPT/artifacts" || return 1
    inventory_tree "$ATTEMPT/artifacts" > "$ATTEMPT/audit/local_inventory.jsonl" || return 1
    cmp "$ATTEMPT/audit/remote_pre_inventory.jsonl" \
        "$ATTEMPT/audit/local_inventory.jsonl" || return 1
    [ -x "$validator" ] || return 1
    "$validator" > "$ATTEMPT/audit/validation_summary.json" || return 1
    [ "${POST_E0_TEST_FAIL:-}" != checksum ] || return 1
    return 0
}

write_remote_worker() {
    # The production worker is this same audited script in an explicit remote
    # mode.  It is copied to /tmp and never checked out over the E0 execution
    # tree.  The test mode above avoids SSH entirely.
    cp "$SCRIPT_DIR/post_e0_closeout.sh" "$ATTEMPT/audit/post_e0_closeout.sh"
    chmod 700 "$ATTEMPT/audit/post_e0_closeout.sh"
}

run_production_snapshot() {
    local audit_commit bundle bundle_hash remote_base
    audit_commit=$(git -C "$SCRIPT_DIR" rev-parse HEAD) || return 1
    bundle="$ATTEMPT/audit/fedspan-post-e0-audit-${audit_commit}.bundle"
    git -C "$SCRIPT_DIR" bundle create "$bundle" "$audit_commit" || return 1
    bundle_hash=$(hash_file "$bundle") || return 1
    remote_base="/tmp/fedspan-post-e0-audit-${audit_commit}"
    write_remote_worker || return 1
    {
        printf 'audit_commit=%s\n' "$audit_commit"
        printf 'bundle_sha256=%s\n' "$bundle_hash"
        printf 'command=%s\n' "validate_e0.py (remote clean clone only)"
        /bin/bash --version
    } > "$ATTEMPT/audit/environment.txt"
    gcloud compute scp "$bundle" "$ATTEMPT/audit/post_e0_closeout.sh" \
        "$INSTANCE:/tmp/" --project "$E0_PROJECT" --zone "$E0_ZONE" || return 1
    # The remote subcommand refuses unsafe entries, copies only inventoried
    # regular files to /tmp, and runs the strengthened validator from the
    # fresh audit-commit clone.  Its outputs remain in /tmp until copied back.
    gcloud compute ssh "$INSTANCE" --project "$E0_PROJECT" --zone "$E0_ZONE" \
        --command "POST_E0_REMOTE_WORKER=1 bash /tmp/post_e0_closeout.sh '$REMOTE_ARTIFACT_ROOT' '$remote_base' '$audit_commit' '$EXECUTION_COMMIT'" || return 1
    gcloud compute scp --recurse "$INSTANCE:$remote_base/artifacts" \
        "$INSTANCE:$remote_base/audit" "$ATTEMPT/" --project "$E0_PROJECT" \
        --zone "$E0_ZONE" || return 1
    [ -d "$ATTEMPT/artifacts" ] && [ -d "$ATTEMPT/audit" ] || return 1
    inventory_tree "$ATTEMPT/artifacts" > "$ATTEMPT/audit/local_inventory.jsonl" || return 1
    cmp "$ATTEMPT/audit/remote_pre_inventory.jsonl" \
        "$ATTEMPT/audit/local_inventory.jsonl" || return 1
}

remote_worker() {
    local source=$1 stage=$2 audit_commit=$3 execution_commit=$4 repo rows run
    [ "$execution_commit" = "$EXECUTION_COMMIT" ] || return 1
    rm -rf "$stage"
    mkdir -p "$stage/artifacts" "$stage/audit" || return 1
    assert_regular_tree "$source" || return 1
    inventory_tree "$source" > "$stage/audit/remote_pre_inventory.jsonl" || return 1
    copy_regular_tree "$source" "$stage/artifacts" || return 1
    check_identity_and_rows "$stage/artifacts" || return 1
    inventory_tree "$stage/artifacts" > "$stage/audit/remote_post_inventory.jsonl" || return 1
    cmp "$stage/audit/remote_pre_inventory.jsonl" "$stage/audit/remote_post_inventory.jsonl" || return 1
    repo="/tmp/fedspan-post-e0-audit-${audit_commit}"
    rm -rf "$repo"
    git clone "/tmp/fedspan-post-e0-audit-${audit_commit}.bundle" "$repo" || return 1
    rows=$("$PYTHON_BIN" - "$stage/artifacts/manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for row in value["rows"]:
    print(row["run_id"])
PY
) || return 1
    : > "$stage/audit/validator_output.jsonl"
    for run in $rows; do
        /home/turjo/FedCRAG/.venv/bin/python "$repo/validate_e0.py" "$stage/artifacts/$run" \
            --manifest "$stage/artifacts/manifest.json" --run_id "$run" \
            --execution_source_root /home/turjo/FedCRAG \
            >> "$stage/audit/validator_output.jsonl" || {
                printf 'scientific validation failed for %s\n' "$run" \
                    > "$stage/audit/validation_failure.txt"
                return 1
            }
    done
    /home/turjo/FedCRAG/.venv/bin/python - <<'PY' > "$stage/audit/runtime_versions.txt"
import numpy, sys, torch
print("python=" + sys.version.replace("\n", " "))
print("torch=" + torch.__version__)
print("numpy=" + numpy.__version__)
PY
    printf '{"validated_rows": %s, "status": "pass"}\n' "$EXPECTED_ROWS" \
        > "$stage/audit/validation_summary.json"
    printf '%s\n' "Post-hoc internally consistent preservation; not a signed historical attestation." \
        > "$stage/audit/2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md"
}

write_reports() {
    "$PYTHON_BIN" - "$ATTEMPT/audit/remote_pre_inventory.jsonl" \
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
    /bin/bash --version > "$ATTEMPT/audit/bash_version.txt"
}

package_and_verify() {
    (cd "$ATTEMPT" && shasum -a 256 SOURCE_SHA256SUMS audit/* > PACKAGE_SHA256SUMS) || return 1
    (cd "$ATTEMPT" && shasum -a 256 -c PACKAGE_SHA256SUMS >/dev/null) || return 1
}

main() {
    local discovery_code before after
    require_clean_worktree
    discover_target
    discovery_code=$?
    if [ "$discovery_code" -ne 0 ]; then
        preserve_preflight_failure "project/zone discovery failed"
        exit "$discovery_code"
    fi
    trap cleanup EXIT
    trap on_signal INT TERM
    [ ! -e "$DEST" ] || fail "canonical destination already exists" 5
    mkdir -p "$(dirname "$DEST")" || fail "cannot create preservation parent"
    ATTEMPT="${DEST}.attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir "$ATTEMPT" && mkdir "$ATTEMPT/artifacts" "$ATTEMPT/audit" || fail "cannot create attempt"
    CLEANUP_ARMED=1

    before=$(get_status) || fail "initial VM status query was malformed or failed"
    printf 'before=%s\n' "$before" > "$ATTEMPT/audit/vm_state.txt"
    if [ "$before" = TERMINATED ]; then
        gcloud compute instances start "$INSTANCE" --project "$E0_PROJECT" \
            --zone "$E0_ZONE" --quiet || fail "could not start terminated VM"
        VM_TOUCHED=1
        after=$(get_status) || fail "post-start VM status query was malformed or failed"
        [ "$after" = RUNNING ] || fail "terminated VM did not become RUNNING"
    else
        after=$before
    fi
    printf 'after=%s\n' "$after" >> "$ATTEMPT/audit/vm_state.txt"

    if [ "${POST_E0_TEST_MODE:-}" = 1 ]; then
        run_test_snapshot "${POST_E0_TEST_REMOTE_ROOT:-}" \
            "${POST_E0_TEST_VALIDATOR:-}" || fail "test snapshot gate failed"
    else
        run_production_snapshot || fail "remote snapshot/validation gate failed"
    fi
    write_reports || fail "report gate failed"
    if ensure_terminated; then
        CLEANUP_DONE=1
    else
        # The bounded attempt has already happened; do not repeat it in EXIT.
        CLEANUP_DONE=1
        fail "explicit shutdown could not be verified" "$CRITICAL_SHUTDOWN"
    fi
    printf 'final=TERMINATED\n' >> "$ATTEMPT/audit/vm_state.txt"
    package_and_verify || fail "package checksum gate failed"
    mv "$ATTEMPT" "$DEST" || fail "atomic publication rename failed"
    ATTEMPT=
}

if [ "${POST_E0_REMOTE_WORKER:-}" = 1 ]; then
    remote_worker "$@"
else
    main "$@"
fi
