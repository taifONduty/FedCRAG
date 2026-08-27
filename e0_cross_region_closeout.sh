#!/bin/bash
set -u
set -o pipefail
readonly PROJECT=rokkh-503122 ORIGINAL_INSTANCE=thesis-fedcrag ORIGINAL_ZONE=asia-south1-c ORIGINAL_DISK=thesis-fedcrag-restored SNAPSHOT=fedcrag-e0-closeout-20260827 CLONE_INSTANCE=thesis-fedcrag-e0-closeout ZONES="asia-southeast1-a asia-southeast1-b asia-southeast1-c" CRITICAL_SHUTDOWN=70 DISK_SIZE=200 TEST_HANDSHAKE=e0-cross-region-fake-v1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
DEST=/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25
CLOSEOUT_SCRIPT=$SCRIPT_DIR/post_e0_closeout.sh
CLOUD=gcloud; POLL_SLEEP=2; POLL_LIMIT=30; TEST_MODE=${E0_CROSS_REGION_TEST_MODE:-}; CLONE_ZONE=
say() { printf '%s\n' "$*" >&2; }
cloud() { "$CLOUD" "$@"; }
configure_test_mode() {
    [ "$TEST_MODE" = 1 ] || return 0
    case ${E0_CROSS_REGION_TRANSPORT:-} in /*) ;; *) return 1;; esac
    [ -x "$E0_CROSS_REGION_TRANSPORT" ] || return 1; CLOUD=$E0_CROSS_REGION_TRANSPORT
    [ "$(cloud __e0_cross_region_test_handshake 2>/dev/null)" = "$TEST_HANDSHAKE" ] || return 1
    DEST=${E0_CROSS_REGION_DEST:?}; CLOSEOUT_SCRIPT=${E0_CROSS_REGION_CLOSEOUT_SCRIPT:?}
    case "$CLOSEOUT_SCRIPT" in /*) ;; *) return 1;; esac
    [ -x "$CLOSEOUT_SCRIPT" ] || return 1
    POLL_SLEEP=${E0_CROSS_REGION_POLL_SLEEP:-0}; POLL_LIMIT=${E0_CROSS_REGION_POLL_LIMIT:-30}
}
require_clean_worktree() {
    local output status
    [ "$TEST_MODE" = 1 ] && [ "${E0_CROSS_REGION_SKIP_GIT_CLEAN:-}" = 1 ] && return 0
    output=$(git -C "$SCRIPT_DIR" status --porcelain); status=$?
    [ "$status" -eq 0 ] && [ -z "$output" ]
}
status_of() { cloud compute instances describe "$1" --project "$PROJECT" --zone "$2" --format='value(status)'; }
original_terminated() { [ "$(status_of "$ORIGINAL_INSTANCE" "$ORIGINAL_ZONE")" = TERMINATED ]; }
clone_presence() {
    local output
    output=$(cloud compute instances list --project "$PROJECT" --zones "$1" --filter="name=($CLONE_INSTANCE)" --format='value(name)') || return 2
    [ -z "$output" ] && return 0; [ "$output" = "$CLONE_INSTANCE" ] && return 1; return 2
}
clone_absent() { clone_presence "$1"; [ "$?" -eq 0 ]; }
named_absent() {
    local kind=$1 zone=$2 name=$3 output
    if [ "$kind" = snapshots ]; then output=$(cloud compute snapshots list --project "$PROJECT" --filter="name=($name)" --format='value(name)') || return 1
    else output=$(cloud compute disks list --project "$PROJECT" --zones "$zone" --filter="name=($name)" --format='value(name)') || return 1; fi
    [ -z "$output" ]
}
disk_descriptor() { cloud compute disks describe "$1" --project "$PROJECT" --zone "$2" --format=json; }
snapshot_descriptor() { cloud compute snapshots describe "$SNAPSHOT" --project "$PROJECT" --format=json; }
instance_descriptor() { cloud compute instances describe "$CLONE_INSTANCE" --project "$PROJECT" --zone "$1" --format=json; }
validate_json() {
    local kind=$1 raw=$2 zone=${3:-} disk=${4:-}
    python3 -c 'import json,sys; k,r,z,d=sys.argv[1:]; v=json.loads(r); p="https://www.googleapis.com/compute/v1/projects/rokkh-503122"; b=p+"/zones/"+z; eq=lambda a,b:a==b; a=v.get("guestAccelerators"); x=v.get("disks"); q={"snapshot":eq(v.get("name"),"fedcrag-e0-closeout-20260827") and eq(v.get("status"),"READY") and str(v.get("diskSizeGb"))=="200" and eq(v.get("sourceDisk"),p+"/zones/asia-south1-c/disks/thesis-fedcrag-restored") and eq(v.get("storageLocations"),["asia"]),"original":eq(v.get("name"),"thesis-fedcrag-restored") and eq(v.get("status"),"READY") and str(v.get("sizeGb"))=="200" and eq(v.get("type"),p+"/zones/asia-south1-c/diskTypes/pd-balanced"),"disk":eq(v.get("name"),d) and eq(v.get("status"),"READY") and str(v.get("sizeGb"))=="200" and eq(v.get("type"),b+"/diskTypes/pd-balanced") and eq(v.get("sourceSnapshot"),p+"/global/snapshots/fedcrag-e0-closeout-20260827") and eq(v.get("users"),[]),"instance":eq(v.get("name"),"thesis-fedcrag-e0-closeout") and eq(v.get("zone"),b) and eq(v.get("machineType"),b+"/machineTypes/g2-standard-8") and eq(v.get("scheduling",{}).get("provisioningModel"),"STANDARD") and isinstance(a,list) and len(a)==1 and eq(a[0].get("acceleratorType"),b+"/acceleratorTypes/nvidia-l4") and eq(a[0].get("acceleratorCount"),1) and isinstance(x,list) and len(x)==1 and eq(x[0].get("source"),b+"/disks/"+d) and eq(x[0].get("boot"),True) and eq(x[0].get("autoDelete"),False) and eq(x[0].get("type"),"PERSISTENT")}; raise SystemExit(0 if q.get(k,False) else 1)' "$kind" "$raw" "$zone" "$disk"
}
stop_and_verify_clone() {
    local presence status; [ -n "$CLONE_ZONE" ] || return 0
    clone_presence "$CLONE_ZONE"; presence=$?; [ "$presence" -eq 0 ] && return 0; [ "$presence" -eq 1 ] || return 1
    cloud compute instances stop "$CLONE_INSTANCE" --project "$PROJECT" --zone "$CLONE_ZONE" --quiet >/dev/null 2>&1 || return 1
    status=$(status_of "$CLONE_INSTANCE" "$CLONE_ZONE") || return 1; [ "$status" = TERMINATED ]
}
exit_cleanup() {
    local result=$? clone_status original_status
    trap - EXIT; trap '' INT TERM
    stop_and_verify_clone; clone_status=$?
    original_terminated; original_status=$?
    if [ "$clone_status" -ne 0 ] || [ "$original_status" -ne 0 ]; then say 'CRITICAL: termination verification failed'; exit "$CRITICAL_SHUTDOWN"; fi
    exit "$result"
}
on_signal() { trap '' INT TERM; exit 20; }
recover_failed_candidate() {
    local zone=$1 disk=$2 descriptor
    clone_absent "$zone" || return 1; descriptor=$(disk_descriptor "$disk" "$zone") || return 1
    validate_json disk "$descriptor" "$zone" "$disk" || return 1
    cloud compute disks delete "$disk" --project "$PROJECT" --zone "$zone" --quiet
}
preflight() {
    local project zone disk descriptor output
    require_clean_worktree || return 1; project=$(cloud config get-value project 2>/dev/null) || return 1
    [ "$project" = "$PROJECT" ] && [ ! -e "$DEST" ] && [ ! -L "$DEST" ] && [ ! -e "${DEST}.publication-lock" ] || return 1
    original_terminated || return 1; descriptor=$(disk_descriptor "$ORIGINAL_DISK" "$ORIGINAL_ZONE") || return 1
    validate_json original "$descriptor" "$ORIGINAL_ZONE" "$ORIGINAL_DISK" || return 1; named_absent snapshots '' "$SNAPSHOT" || return 1
    output=$(cloud compute instances list --project "$PROJECT" --filter="name=($CLONE_INSTANCE)" --format='value(name)') || return 1; [ -z "$output" ] || return 1
    for zone in $ZONES; do disk="$CLONE_INSTANCE-boot-${zone##*-}"; clone_absent "$zone" && named_absent disks "$zone" "$disk" || return 1; done
}
create_and_validate_snapshot() {
    local count=1 descriptor
    cloud compute snapshots create "$SNAPSHOT" --project "$PROJECT" --source-disk "$ORIGINAL_DISK" --source-disk-zone "$ORIGINAL_ZONE" --storage-location asia || return 1
    while [ "$count" -le "$POLL_LIMIT" ]; do descriptor=$(snapshot_descriptor) || return 1; validate_json snapshot "$descriptor" && return 0; count=$((count + 1)); [ "$count" -le "$POLL_LIMIT" ] && sleep "$POLL_SLEEP"; done
    return 1
}
allocate_clone() {
    local zone disk descriptor instance_error
    for zone in $ZONES; do
        disk="$CLONE_INSTANCE-boot-${zone##*-}"; CLONE_ZONE=$zone
        cloud compute disks create "$disk" --project "$PROJECT" --zone "$zone" --size "${DISK_SIZE}GB" --type pd-balanced --source-snapshot "$SNAPSHOT" || return 1
        descriptor=$(disk_descriptor "$disk" "$zone") || return 1; validate_json disk "$descriptor" "$zone" "$disk" || return 1
        if instance_error=$(cloud compute instances create "$CLONE_INSTANCE" --project "$PROJECT" --zone "$zone" --machine-type g2-standard-8 --disk "name=$disk,boot=yes,auto-delete=no,mode=rw" --network default --subnet default --network-tier PREMIUM --maintenance-policy TERMINATE --restart-on-failure --provisioning-model STANDARD --service-account 139678593638-compute@developer.gserviceaccount.com --scopes storage-ro,logging-write,monitoring-write,pubsub,service-management-ro,service-control,trace --shielded-vtpm --shielded-integrity-monitoring --no-shielded-secure-boot 2>&1); then
            descriptor=$(instance_descriptor "$zone") || return 1; validate_json instance "$descriptor" "$zone" "$disk" || return 1; return 0
        fi
        clone_absent "$zone" || return 1; printf '%s' "$instance_error" | grep -F ZONE_RESOURCE_POOL_EXHAUSTED >/dev/null || return 1; recover_failed_candidate "$zone" "$disk" || return 1
    done
    return 20
}
run_closeout() { env -u POST_E0_TEST_MODE -u POST_E0_TEST_REMOTE_ROOT -u POST_E0_REMOTE_TMP -u POST_E0_VALIDATOR -u POST_E0_PYTHON -u POST_E0_TEST_EXECUTION_SOURCE_ROOT -u POST_E0_TEST_EXECUTION_INTERPRETER_PATH -u POST_E0_REMOTE_WORKER POST_E0_INSTANCE="$CLONE_INSTANCE" POST_E0_EXPECTED_SOURCE_SNAPSHOT="$SNAPSHOT" POST_E0_DEST="$DEST" /bin/bash "$CLOSEOUT_SCRIPT"; }
main() {
    local allocation_status closeout_status
    configure_test_mode || { say 'test transport refused'; return 20; }; preflight || { say 'preflight refused'; return 20; }
    trap exit_cleanup EXIT; trap on_signal INT TERM
    create_and_validate_snapshot || return 20; allocate_clone; allocation_status=$?; [ "$allocation_status" -eq 0 ] || return 20
    run_closeout; closeout_status=$?; return "$closeout_status"
}
main "$@"
