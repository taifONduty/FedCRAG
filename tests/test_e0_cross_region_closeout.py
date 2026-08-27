"""Hermetic contracts for the cross-region E0 closeout controller."""
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "e0_cross_region_closeout.sh"


@pytest.fixture
def env(tmp_path):
    """Install fake cloud and closeout transports with independent VM states."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gcloud.log"
    original_status = tmp_path / "original-status"
    clone_status = tmp_path / "clone-status"
    disk_state = tmp_path / "disk-state"
    snapshot_state = tmp_path / "snapshot-state"
    capacity_state = tmp_path / "capacity-state"
    original_status.write_text("TERMINATED")
    clone_status.write_text("ABSENT")
    disk_state.write_text("")
    snapshot_state.write_text("ABSENT")
    capacity_state.write_text("ready")

    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text(r'''#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_GCLOUD_LOG"
fault=${FAKE_FAULT:-}
arg_after() {
    previous=
    for arg in "$@"; do
        [ "$previous" = "$1" ] && { printf '%s\n' "$arg"; return; }
        previous=$arg
    done
    return 1
}
if [ "$1" = config ]; then printf '%s\n' rokkh-503122; exit 0; fi
if [ "$1" = compute ] && [ "$2" = instances ] && [ "$3" = list ]; then
    [ "$fault" = clone-preexists ] && printf '%s\n' thesis-fedcrag-e0-closeout
    [ "$(cat "$FAKE_CLONE_STATUS")" = ABSENT ] || printf '%s\n' thesis-fedcrag-e0-closeout
    exit 0
fi
if [ "$1" = compute ] && [ "$2" = disks ] && [ "$3" = list ]; then
    [ "$fault" = candidate-disk-query-failure ] && exit 1
    [ "$fault" = candidate-disk-preexists ] && printf '%s\n' thesis-fedcrag-e0-closeout-boot-a
    cat "$FAKE_DISK_STATE"
    exit 0
fi
if [ "$1" = compute ] && [ "$2" = snapshots ] && [ "$3" = list ]; then
    [ "$fault" = snapshot-query-failure ] && exit 1
    if [ "$(cat "$FAKE_SNAPSHOT_STATE")" = READY ] || [ "$fault" = snapshot-preexists ] || [ "$fault" = snapshot-wrong-source ] || [ "$fault" = snapshot-not-ready ]; then
        printf '%s\n' fedcrag-e0-closeout-20260827
    fi
    exit 0
fi
if [ "$1" = compute ] && [ "$2" = instances ] && [ "$3" = describe ]; then
    name=$4
    if [ "$name" = thesis-fedcrag ]; then
        responses=
        [ -f "$FAKE_ORIGINAL_RESPONSES_FILE" ] && responses=$(cat "$FAKE_ORIGINAL_RESPONSES_FILE")
        if [ -n "$responses" ]; then
            value=${responses%%,*}
            if [ "$responses" != "$value" ]; then printf '%s' "${responses#*,}" > "$FAKE_ORIGINAL_RESPONSES_FILE"; fi
            printf '%s\n' "$value"; exit 0
        fi
        cat "$FAKE_ORIGINAL_STATUS"; exit 0
    fi
    status=$(cat "$FAKE_CLONE_STATUS")
    [ "$status" != ABSENT ] || exit 1
    printf '%s\n' "$status"; exit 0
fi
if [ "$1" = compute ] && [ "$2" = disks ] && [ "$3" = describe ]; then
    name=$4
    if [ "$name" = thesis-fedcrag-restored ]; then
        printf '%s\n' '{"name":"thesis-fedcrag-restored","status":"READY","sizeGb":"200","type":"https://example/diskTypes/pd-balanced"}'
        exit 0
    fi
    [ "$fault" = candidate-disk-query-failure ] && exit 1
    if [ "$fault" = candidate-disk-preexists ] || grep -qx "$name" "$FAKE_DISK_STATE"; then
        source=fedcrag-e0-closeout-20260827
        [ "$fault" = failed-disk-identity-mismatch ] && source=wrong-snapshot
        printf '{"name":"%s","status":"READY","sizeGb":"200","type":"https://example/diskTypes/pd-balanced","sourceSnapshot":"https://example/global/snapshots/%s","users":[]}' "${name}" "$source"
        exit 0
    fi
    exit 1
fi
if [ "$1" = compute ] && [ "$2" = snapshots ] && [ "$3" = describe ]; then
    [ "$fault" = snapshot-query-failure ] && exit 1
    [ "$(cat "$FAKE_SNAPSHOT_STATE")" = READY ] || [ "$fault" = snapshot-preexists ] || [ "$fault" = snapshot-wrong-source ] || [ "$fault" = snapshot-not-ready ] || exit 1
    source=thesis-fedcrag-restored; status=READY
    [ "$fault" = snapshot-wrong-source ] && source=wrong-disk
    [ "$fault" = snapshot-not-ready ] && status=CREATING
    printf '{"name":"fedcrag-e0-closeout-20260827","status":"%s","diskSizeGb":"200","sourceDisk":"https://example/zones/asia-south1-c/disks/%s","storageLocations":["asia"]}' "$status" "$source"
    exit 0
fi
if [ "$1" = compute ] && [ "$2" = snapshots ] && [ "$3" = create ]; then
    printf READY > "$FAKE_SNAPSHOT_STATE"; exit 0
fi
if [ "$1" = compute ] && [ "$2" = disks ] && [ "$3" = create ]; then
    printf '%s\n' "$4" >> "$FAKE_DISK_STATE"; exit 0
fi
if [ "$1" = compute ] && [ "$2" = disks ] && [ "$3" = delete ]; then
    : > "$FAKE_DISK_STATE"; exit 0
fi
if [ "$1" = compute ] && [ "$2" = instances ] && [ "$3" = create ]; then
    values=$(cat "$FAKE_CAPACITY_STATE"); value=${values%%,*}
    [ "$values" = "$value" ] || printf '%s' "${values#*,}" > "$FAKE_CAPACITY_STATE"
    if [ "$fault" = partial-instance ]; then printf RUNNING > "$FAKE_CLONE_STATUS"; exit 1; fi
    [ "$fault" = unexpected-instance-failure ] && exit 1
    [ "$value" = ready ] || { printf '%s\n' ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS >&2; exit 1; }
    printf RUNNING > "$FAKE_CLONE_STATUS"; exit 0
fi
if [ "$1" = compute ] && [ "$2" = instances ] && [ "$3" = stop ]; then
    [ "$4" = thesis-fedcrag-e0-closeout ] || exit 91
    [ "$fault" = clone-stop-unverified ] || printf TERMINATED > "$FAKE_CLONE_STATUS"
    exit 0
fi
printf '%s\n' "unexpected fake gcloud call: $*" >&2; exit 99
''')
    fake_gcloud.chmod(fake_gcloud.stat().st_mode | stat.S_IXUSR)

    closeout = tmp_path / "fake-closeout.sh"
    closeout.write_text("#!/bin/sh\nprintf '%s %s\\n' \"$POST_E0_INSTANCE\" \"$POST_E0_EXPECTED_SOURCE_SNAPSHOT\" > \"$FAKE_CLOSEOUT_LOG\"\n[ \"${FAKE_FAULT:-}\" = closeout-failure ] && exit 20\nexit 0\n")
    closeout.chmod(closeout.stat().st_mode | stat.S_IXUSR)
    closeout_log = tmp_path / "closeout.log"
    controller_env = os.environ.copy()
    controller_env.update({
        "PATH": str(bin_dir) + os.pathsep + controller_env["PATH"],
        "FAKE_GCLOUD_LOG": str(log),
        "FAKE_ORIGINAL_STATUS": str(original_status),
        "FAKE_CLONE_STATUS": str(clone_status),
        "FAKE_DISK_STATE": str(disk_state),
        "FAKE_SNAPSHOT_STATE": str(snapshot_state),
        "FAKE_CAPACITY_STATE": str(capacity_state),
        "FAKE_ORIGINAL_RESPONSES_FILE": str(tmp_path / "original-responses"),
        "FAKE_CLOSEOUT_LOG": str(closeout_log),
        "E0_CROSS_REGION_TEST_MODE": "1",
        "E0_CROSS_REGION_CLOSEOUT_SCRIPT": str(closeout),
        "E0_CROSS_REGION_DEST": str(tmp_path / "post_e0_audit" / "2026-08-25"),
        "E0_CROSS_REGION_SKIP_GIT_CLEAN": "1",
        "E0_CROSS_REGION_POLL_SLEEP": "0",
    })
    return controller_env


def run_controller(env):
    assert SCRIPT.is_file(), "RED: e0_cross_region_closeout.sh is absent"
    return subprocess.run(["/bin/bash", str(SCRIPT)], cwd=ROOT, env=env,
                          capture_output=True, text=True)


@pytest.mark.parametrize(("capacity", "zones", "expected_status"), [
    ("ready", ["asia-southeast1-a"], 0),
    ("stockout,ready", ["asia-southeast1-a", "asia-southeast1-b"], 0),
    ("stockout,stockout,stockout", ["asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c"], 20),
])
def test_bounded_zone_search_and_shutdown(env, capacity, zones, expected_status):
    env["FAKE_CAPACITY"] = capacity
    Path(env["FAKE_CAPACITY_STATE"]).write_text(capacity)
    completed = run_controller(env)
    assert completed.returncode == expected_status, completed.stderr
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    attempted = [line.split("--zone ", 1)[1].split()[0] for line in calls
                 if "instances create thesis-fedcrag-e0-closeout" in line]
    assert attempted == zones
    assert Path(env["FAKE_ORIGINAL_STATUS"]).read_text() == "TERMINATED"
    assert Path(env["FAKE_CLONE_STATUS"]).read_text() in ("ABSENT", "TERMINATED")
    if expected_status == 0:
        assert Path(env["FAKE_CLOSEOUT_LOG"]).read_text().strip() == "thesis-fedcrag-e0-closeout fedcrag-e0-closeout-20260827"


@pytest.mark.parametrize("fault", ["original-running", "snapshot-preexists", "clone-preexists", "candidate-disk-preexists", "snapshot-wrong-source", "snapshot-not-ready", "snapshot-query-failure", "candidate-disk-query-failure"])
def test_preflight_identity_faults_write_no_new_resource(env, fault):
    env["FAKE_FAULT"] = fault
    if fault == "original-running":
        Path(env["FAKE_ORIGINAL_RESPONSES_FILE"]).write_text("RUNNING")
    completed = run_controller(env)
    assert completed.returncode != 0
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    assert not any(token in line for line in calls for token in ("snapshots create", "disks create", "instances create"))
    assert Path(env["FAKE_ORIGINAL_STATUS"]).read_text() == "TERMINATED"


@pytest.mark.parametrize(("fault", "expected_status"), [("failed-disk-identity-mismatch", 20), ("partial-instance", 20), ("unexpected-instance-failure", 20), ("closeout-failure", 20), ("clone-stop-unverified", 70), ("original-stop-unverified", 70)])
def test_failure_cleanup_is_identity_bound_and_fail_closed(env, fault, expected_status):
    env["FAKE_FAULT"] = fault
    if fault == "original-stop-unverified":
        env["FAKE_ORIGINAL_RESPONSES"] = "TERMINATED,RUNNING"
        Path(env["FAKE_ORIGINAL_RESPONSES_FILE"]).write_text("TERMINATED,RUNNING")
    completed = run_controller(env)
    assert completed.returncode == expected_status
    calls = Path(env["FAKE_GCLOUD_LOG"]).read_text().splitlines()
    if fault == "failed-disk-identity-mismatch": assert not any("disks delete" in line for line in calls)
    if fault == "partial-instance": assert not any("instances delete" in line for line in calls)
    if fault == "unexpected-instance-failure": assert not any("disks delete" in line for line in calls)
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
