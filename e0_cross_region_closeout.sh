#!/bin/bash
# Fail-safe, bounded cross-region E0 closeout controller.  GNU Bash 3.2 only.
set -u
set -o pipefail

readonly PROJECT=rokkh-503122
readonly ORIGINAL_INSTANCE=thesis-fedcrag
readonly ORIGINAL_ZONE=asia-south1-c
readonly ORIGINAL_DISK=thesis-fedcrag-restored
readonly SNAPSHOT=fedcrag-e0-closeout-20260827
readonly CLONE_INSTANCE=thesis-fedcrag-e0-closeout
readonly ZONES="asia-southeast1-a asia-southeast1-b asia-southeast1-c"
readonly CRITICAL_SHUTDOWN=70
readonly DISK_SIZE=200

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
DEST=/Users/turjo/Desktop/FedCRAG/post_e0_audit/2026-08-25
CLOSEOUT_SCRIPT=$SCRIPT_DIR/post_e0_closeout.sh
POLL_SLEEP=2
POLL_LIMIT=30
if [ "${E0_CROSS_REGION_TEST_MODE:-}" = 1 ]; then
    DEST=${E0_CROSS_REGION_DEST:?test destination required}
    CLOSEOUT_SCRIPT=${E0_CROSS_REGION_CLOSEOUT_SCRIPT:?test closeout required}
    POLL_SLEEP=${E0_CROSS_REGION_POLL_SLEEP:-0}
fi

CLONE_ZONE=
CLONE_EXISTS=0

say() { printf '%s\n' "$*" >&2; }

require_clean_worktree() {
    [ "${E0_CROSS_REGION_SKIP_GIT_CLEAN:-}" = 1 ] && return 0
    [ -z "$(git status --porcelain)" ]
}

status_of() {
    gcloud compute instances describe "$1" --project "$PROJECT" --zone "$2" \
        --format='value(status)'
}

original_terminated() {
    [ "$(status_of "$ORIGINAL_INSTANCE" "$ORIGINAL_ZONE")" = TERMINATED ]
}

clone_absent() {
    local output
    output=$(gcloud compute instances list --project "$PROJECT" --zones "$1" \
        --filter="name=($CLONE_INSTANCE)" --format='value(name)') || return 1
    [ -z "$output" ]
}

snapshot_absent() {
    local output
    output=$(gcloud compute snapshots list --project "$PROJECT" \
        --filter="name=($SNAPSHOT)" --format='value(name)') || return 1
    [ -z "$output" ]
}

candidate_disk_absent() {
    local zone=$1 disk=$2 output
    output=$(gcloud compute disks list --project "$PROJECT" --zones "$zone" \
        --filter="name=($disk)" --format='value(name)') || return 1
    [ -z "$output" ]
}

disk_descriptor() {
    gcloud compute disks describe "$1" --project "$PROJECT" --zone "$2" --format=json
}

snapshot_descriptor() {
    gcloud compute snapshots describe "$SNAPSHOT" --project "$PROJECT" --format=json
}

descriptor_has() { printf '%s' "$1" | grep -F "$2" >/dev/null; }

valid_snapshot() {
    local descriptor=$1
    descriptor_has "$descriptor" "\"name\":\"$SNAPSHOT\"" &&
        descriptor_has "$descriptor" '"status":"READY"' &&
        descriptor_has "$descriptor" '"diskSizeGb":"200"' &&
        descriptor_has "$descriptor" "/zones/$ORIGINAL_ZONE/disks/$ORIGINAL_DISK" &&
        descriptor_has "$descriptor" '"storageLocations":["asia"]'
}

valid_candidate_disk() {
    local descriptor=$1 disk=$2
    descriptor_has "$descriptor" "\"name\":\"$disk\"" &&
        descriptor_has "$descriptor" '"status":"READY"' &&
        descriptor_has "$descriptor" '"sizeGb":"200"' &&
        descriptor_has "$descriptor" 'diskTypes/pd-balanced' &&
        descriptor_has "$descriptor" "/snapshots/$SNAPSHOT" &&
        descriptor_has "$descriptor" '"users":[]'
}

stop_and_verify_clone() {
    [ -n "$CLONE_ZONE" ] || return 0
    gcloud compute instances stop "$CLONE_INSTANCE" --project "$PROJECT" --zone "$CLONE_ZONE" --quiet >/dev/null 2>&1 || return 1
    [ "$(status_of "$CLONE_INSTANCE" "$CLONE_ZONE")" = TERMINATED ]
}

exit_cleanup() {
    local result=$?
    trap - EXIT INT TERM
    if ! stop_and_verify_clone || ! original_terminated; then
        say "CRITICAL: termination verification failed"
        exit "$CRITICAL_SHUTDOWN"
    fi
    exit "$result"
}

on_signal() { exit 20; }

ensure_candidate_absent() {
    local zone=$1 disk=$2
    clone_absent "$zone" || return 1
    candidate_disk_absent "$zone" "$disk"
}

recover_failed_candidate() {
    local zone=$1 disk=$2 descriptor
    clone_absent "$zone" || return 1
    descriptor=$(disk_descriptor "$disk" "$zone") || return 1
    valid_candidate_disk "$descriptor" "$disk" || return 1
    gcloud compute disks delete "$disk" --project "$PROJECT" --zone "$zone" --quiet
}

preflight() {
    local project zone disk descriptor
    require_clean_worktree || return 1
    project=$(gcloud config get-value project 2>/dev/null) || return 1
    [ "$project" = "$PROJECT" ] || return 1
    [ ! -e "$DEST" ] && [ ! -L "$DEST" ] || return 1
    original_terminated || return 1
    descriptor=$(disk_descriptor "$ORIGINAL_DISK" "$ORIGINAL_ZONE") || return 1
    descriptor_has "$descriptor" "\"name\":\"$ORIGINAL_DISK\"" &&
        descriptor_has "$descriptor" '"status":"READY"' &&
        descriptor_has "$descriptor" '"sizeGb":"200"' &&
        descriptor_has "$descriptor" 'diskTypes/pd-balanced' || return 1
    snapshot_absent || return 1
    output=$(gcloud compute instances list --project "$PROJECT" --filter="name=($CLONE_INSTANCE)" --format='value(name)') || return 1
    [ -z "$output" ] || return 1
    for zone in $ZONES; do
        disk="$CLONE_INSTANCE-boot-${zone##*-}"
        ensure_candidate_absent "$zone" "$disk" || return 1
    done
}

create_and_validate_snapshot() {
    local count descriptor
    gcloud compute snapshots create "$SNAPSHOT" --project "$PROJECT" \
        --source-disk "$ORIGINAL_DISK" --source-disk-zone "$ORIGINAL_ZONE" \
        --storage-location asia || return 1
    count=1
    while [ "$count" -le "$POLL_LIMIT" ]; do
        descriptor=$(snapshot_descriptor) || return 1
        if descriptor_has "$descriptor" '"status":"READY"'; then
            valid_snapshot "$descriptor" && return 0
            return 1
        fi
        count=$((count + 1))
        sleep "$POLL_SLEEP"
    done
    return 1
}

allocate_clone() {
    local zone disk descriptor instance_error
    for zone in $ZONES; do
        disk="$CLONE_INSTANCE-boot-${zone##*-}"
        gcloud compute disks create "$disk" --project "$PROJECT" --zone "$zone" \
            --size "${DISK_SIZE}GB" --type pd-balanced --source-snapshot "$SNAPSHOT" || return 1
        descriptor=$(disk_descriptor "$disk" "$zone") || return 1
        valid_candidate_disk "$descriptor" "$disk" || return 1
        if instance_error=$(gcloud compute instances create "$CLONE_INSTANCE" --project "$PROJECT" --zone "$zone" \
            --machine-type g2-standard-8 --disk "name=$disk,boot=yes,auto-delete=no,mode=rw" \
            --network default --subnet default --network-tier PREMIUM \
            --maintenance-policy TERMINATE --restart-on-failure --provisioning-model STANDARD \
            --service-account 139678593638-compute@developer.gserviceaccount.com \
            --scopes storage-ro,logging-write,monitoring-write,pubsub,service-management-ro,service-control,trace \
            --shielded-vtpm --shielded-integrity-monitoring --no-shielded-secure-boot 2>&1); then
            CLONE_ZONE=$zone
            CLONE_EXISTS=1
            return 0
        fi
        if ! clone_absent "$zone"; then
            CLONE_ZONE=$zone
            CLONE_EXISTS=1
            return 1
        fi
        printf '%s' "$instance_error" | grep -F 'ZONE_RESOURCE_POOL_EXHAUSTED' >/dev/null || return 1
        recover_failed_candidate "$zone" "$disk" || return 1
    done
    return 20
}

main() {
    local allocation_status closeout_status
    preflight || { say "preflight refused"; return 20; }
    trap exit_cleanup EXIT
    trap on_signal INT TERM
    create_and_validate_snapshot || { say "snapshot validation failed"; return 20; }
    allocate_clone
    allocation_status=$?
    [ "$allocation_status" -eq 0 ] || return 20
    POST_E0_INSTANCE="$CLONE_INSTANCE" \
    POST_E0_EXPECTED_SOURCE_SNAPSHOT="$SNAPSHOT" \
    POST_E0_DEST="$DEST" /bin/bash "$CLOSEOUT_SCRIPT"
    closeout_status=$?
    return "$closeout_status"
}

main "$@"
