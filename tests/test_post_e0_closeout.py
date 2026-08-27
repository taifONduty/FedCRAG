"""Hermetic contract tests for the fail-safe E0 preservation driver.

The tests install a fake ``gcloud`` ahead of PATH.  The driver is always run
with /bin/bash, never the user's login shell, and the fake records every cloud
operation so a regression cannot silently reach GCP during CI.
"""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "post_e0_closeout.sh"
EXPECTED_COMMIT = "7325bf56381c24c6a4af013688bdd417c95d7d7d"
EXPECTED_COMMIT_SHORT = "7325bf56381c"


@pytest.fixture
def closeout_env(tmp_path):
    """Build a local-only E0 source and a logged fake gcloud executable."""
    source = tmp_path / "remote-e0-results"
    source.mkdir()
    (source / "COMPLETE.json").write_text(json.dumps({
        "commit": EXPECTED_COMMIT_SHORT,
        "validated_rows": [f"row-{index}" for index in range(11)],
    }))
    (source / "manifest.json").write_text(json.dumps({
        "commit": EXPECTED_COMMIT_SHORT,
        "rows": [
            {"run_id": f"row-{index}", "commit": EXPECTED_COMMIT_SHORT}
            for index in range(11)
        ],
    }))
    for index in range(11):
        row = source / f"row-{index}"
        row.mkdir()
        (row / "federated_result.json").write_text(json.dumps({
            "commit": EXPECTED_COMMIT_SHORT,
        }))
    (source / "status.tsv").write_text("".join(
        f"row-{index}\tVALIDATED\t{index + 1}.5\t2026-08-25T12:00:00Z\n" for index in range(11)))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    remote_tmp = tmp_path / "remote-tmp"
    remote_tmp.mkdir()
    execution_source = tmp_path / "execution-source"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(ROOT), str(execution_source)],
        check=True)
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", EXPECTED_COMMIT],
        cwd=execution_source, check=True)
    status_file = tmp_path / "statuses"
    status_file.write_text("RUNNING")
    ssh_readiness_file = tmp_path / "ssh-readiness"
    ssh_readiness_file.write_text("ready")
    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text("""#!/bin/sh
set -eu
printf '%s\\n' \"$*\" >> \"$FAKE_GCLOUD_LOG\"
if [ \"$1\" = config ]; then
  [ \"${FAKE_CONFIG_FAIL:-0}\" = 0 ] || exit 1
  printf '%s\\n' \"${FAKE_PROJECT:-project-e0}\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = list ]; then
  [ \"${FAKE_LIST_FAIL:-0}\" = 0 ] || exit 1
  if [ -n \"${FAKE_DISCOVERY_ROWS:-}\" ]; then
    printf '%s\\n' \"$FAKE_DISCOVERY_ROWS\"
  else
    printf '%s\\n' \"${FAKE_ZONES-zone-e0}\" | tr ',' '\\n' | while IFS= read -r zone; do
      if [ -n \"$zone\" ]; then
        printf '%s,%s\\n' \"${FAKE_DISCOVERY_NAME-thesis-fedcrag}\" \"$zone\"
      fi
    done
  fi
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = describe ]; then
  case \" $* \" in
    *\" --format=json \"*)
      [ \"${FAKE_INSTANCE_DESCRIPTOR_FAIL:-0}\" = 0 ] || exit 1
      printf '%s\\n' \"$FAKE_INSTANCE_DESCRIPTOR\"
      exit 0
      ;;
  esac
  [ \"${FAKE_STATUS_FAIL:-0}\" = 0 ] || exit 1
  values=$(cat \"$FAKE_STATUS_FILE\")
  value=${values%%,*}
  if [ \"$values\" != \"$value\" ]; then
    printf '%s' \"${values#*,}\" > \"$FAKE_STATUS_FILE\"
  fi
  printf '%s\\n' \"$value\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = disks ] && [ \"$3\" = describe ]; then
  [ \"${FAKE_DISK_DESCRIPTOR_FAIL:-0}\" = 0 ] || exit 1
  printf '%s\\n' \"$FAKE_DISK_DESCRIPTOR\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = snapshots ] && [ \"$3\" = describe ]; then
  [ \"${FAKE_SNAPSHOT_DESCRIPTOR_FAIL:-0}\" = 0 ] || exit 1
  printf '%s\\n' \"$FAKE_SNAPSHOT_DESCRIPTOR\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = start ]; then
  [ \"${FAKE_START_FAIL:-0}\" = 0 ] || exit 1
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = stop ]; then
  if [ \"${FAKE_CREATE_DEST_ON_STOP:-0}\" = 1 ]; then mkdir -p \"$POST_E0_DEST\"; printf foreign > \"$POST_E0_DEST/marker\"; fi
  [ \"${FAKE_STOP_FAIL:-0}\" = 0 ] || exit 1
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = scp ]; then
  first=$3
  [ \"$first\" = --recurse ] && first=$4
  case \"$first\" in *:*)
    destination=
    for item in \"$@\"; do
      [ \"$item\" = --project ] && break
      destination=$item
    done
    mkdir -p \"$destination\"
    for item in \"$@\"; do
      case \"$item\" in *:*) cp -R \"${item#*:}\" \"$destination/\" ;; esac
    done
    ;;
  *)
    target=
    for item in \"$@\"; do case \"$item\" in *:*) target=$item ;; esac; done
    destination=${target#*:}
    mkdir -p \"$destination\"
    for item in \"$@\"; do
      case \"$item\" in
        -*|compute|scp|\"$target\"|*:* ) ;;
        *) [ -f \"$item\" ] && cp \"$item\" \"$destination/\" ;;
      esac
    done
    destination=
    ;;
  esac
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = ssh ]; then
  command=
  shift 2
  while [ \"$#\" -gt 0 ]; do
    if [ \"$1\" = --command ]; then command=$2; break; fi
    shift
  done
  [ -n \"$command\" ] || exit 98
  if [ \"$command\" = true ]; then
    values=$(cat \"$FAKE_SSH_READINESS_FILE\")
    value=${values%%,*}
    if [ \"$values\" != \"$value\" ]; then
      printf '%s' \"${values#*,}\" > \"$FAKE_SSH_READINESS_FILE\"
    fi
    [ \"$value\" = ready ] && exit 0
    exit 1
  fi
  /bin/bash -c \"$command\"
  exit $?
fi
printf '%s\\n' \"unexpected fake gcloud command: $*\" >&2
exit 99
""")
    fake_gcloud.chmod(fake_gcloud.stat().st_mode | stat.S_IXUSR)

    validator = tmp_path / "validator"
    validator.write_text("""#!/bin/sh
if [ \"${POST_E0_TEST_FAIL:-}\" = validation ]; then echo 'scientific refusal: synthetic validator failure' >&2; exit 41; fi
source_anchor=
interpreter_anchor=
while [ \"$#\" -gt 0 ]; do
  case \"$1\" in
    --execution_source_root) shift; source_anchor=$1 ;;
    --execution_interpreter_path) shift; interpreter_anchor=$1 ;;
  esac
  shift
done
[ \"$source_anchor\" = \"$EXPECTED_EXECUTION_SOURCE_ROOT\" ] || exit 42
[ \"$interpreter_anchor\" = \"$EXPECTED_EXECUTION_INTERPRETER_PATH\" ] || exit 43
printf '{"manifest_verified": true, "dataset_content_verified": true, "runtime_provenance_verified": true, "continuity_boundaries_checked": 4, "aggregate_recomputation_worst_tolerance_ratio": 0.1, "fedspan_direction_residuals": {}, "rawmaxmin_direction_residuals": {}}\\n'
""")
    validator.chmod(validator.stat().st_mode | stat.S_IXUSR)

    destination = tmp_path / "post_e0_audit" / "2026-08-25"
    env = os.environ.copy()
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "FAKE_GCLOUD_LOG": str(gcloud_log),
        "FAKE_STATUS_FILE": str(status_file),
        "FAKE_SSH_READINESS_FILE": str(ssh_readiness_file),
        "POST_E0_DEST": str(destination),
        "POST_E0_TEST_MODE": "1",
        "POST_E0_TEST_REMOTE_ROOT": str(source),
        "POST_E0_REMOTE_TMP": str(remote_tmp),
        "POST_E0_TEST_EXECUTION_SOURCE_ROOT": str(execution_source),
        "POST_E0_TEST_EXECUTION_INTERPRETER_PATH": "/shared/FedCRAG/.venv/bin/python",
        "EXPECTED_EXECUTION_SOURCE_ROOT": str(execution_source),
        "EXPECTED_EXECUTION_INTERPRETER_PATH": "/shared/FedCRAG/.venv/bin/python",
        "POST_E0_VALIDATOR": str(validator),
        "POST_E0_RETRY_SLEEP": "0",
        "FAKE_INSTANCE_DESCRIPTOR": json.dumps({
            "name": "thesis-fedcrag",
            "zone": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/zone-e0",
            "machineType": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/zone-e0/machineTypes/g2-standard-8",
            "guestAccelerators": [{
                "acceleratorType": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/zone-e0/acceleratorTypes/nvidia-l4",
                "acceleratorCount": 1,
            }],
            "scheduling": {"provisioningModel": "STANDARD"},
            "disks": [{
                "boot": True, "type": "PERSISTENT",
                "source": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/zone-e0/disks/thesis-fedcrag-restored",
            }],
        }),
        "FAKE_DISK_DESCRIPTOR": json.dumps({
            "name": "thesis-fedcrag-restored", "status": "READY", "sizeGb": "200",
            "type": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/zone-e0/diskTypes/pd-balanced",
            "sourceSnapshot": "https://www.googleapis.com/compute/v1/projects/project-e0/global/snapshots/fedcrag-e0-closeout-20260827",
        }),
        "FAKE_SNAPSHOT_DESCRIPTOR": json.dumps({
            "name": "fedcrag-e0-closeout-20260827", "status": "READY", "diskSizeGb": "200",
            "sourceDisk": "https://www.googleapis.com/compute/v1/projects/project-e0/zones/asia-south1-c/disks/thesis-fedcrag-restored",
        }),
    })
    return {"env": env, "destination": destination, "log": gcloud_log,
            "source": source, "remote_tmp": remote_tmp,
            "execution_source": execution_source}


def run_driver(closeout_env, **extra_env):
    assert SCRIPT.is_file(), "RED: post_e0_closeout.sh has not been created"
    env = closeout_env["env"].copy()
    env.update({key: str(value) for key, value in extra_env.items()})
    if "FAKE_STATUSES" in extra_env:
        Path(env["FAKE_STATUS_FILE"]).write_text(str(extra_env["FAKE_STATUSES"]))
    if "FAKE_SSH_READINESS" in extra_env:
        Path(env["FAKE_SSH_READINESS_FILE"]).write_text(str(extra_env["FAKE_SSH_READINESS"]))
    return subprocess.run(
        ["/bin/bash", str(SCRIPT)], cwd=ROOT, env=env,
        capture_output=True, text=True)


def cloud_calls(closeout_env):
    path = closeout_env["log"]
    return path.read_text().splitlines() if path.exists() else []


def failed_attempts(closeout_env):
    parent = closeout_env["destination"].parent
    return sorted(parent.glob("2026-08-25.failed-*"))


def assert_not_published(closeout_env):
    assert not closeout_env["destination"].exists()
    assert failed_attempts(closeout_env), "failure must retain a unique attempt"


def target_descriptor_env(closeout_env, mutation):
    """Return one malformed descriptor set for the target-binding gate."""
    instance = json.loads(closeout_env["env"]["FAKE_INSTANCE_DESCRIPTOR"])
    disk = json.loads(closeout_env["env"]["FAKE_DISK_DESCRIPTOR"])
    snapshot = json.loads(closeout_env["env"]["FAKE_SNAPSHOT_DESCRIPTOR"])
    if mutation == "wrong-name":
        instance["name"] = "not-the-clone"
    elif mutation == "machine-type":
        instance["machineType"] = instance["machineType"].replace("g2-standard-8", "n1-standard-8")
    elif mutation == "l4-type":
        instance["guestAccelerators"][0]["acceleratorType"] = "https://example/acceleratorTypes/nvidia-tesla-t4"
    elif mutation == "l4-count":
        instance["guestAccelerators"][0]["acceleratorCount"] = 2
    elif mutation == "provisioning":
        instance["scheduling"]["provisioningModel"] = "SPOT"
    elif mutation == "boot-disk":
        instance["disks"][0]["boot"] = False
    elif mutation == "disk-size":
        disk["sizeGb"] = "199"
    elif mutation == "disk-type":
        disk["type"] = disk["type"].replace("pd-balanced", "pd-standard")
    elif mutation == "disk-status":
        disk["status"] = "CREATING"
    elif mutation == "source-snapshot":
        disk["sourceSnapshot"] = disk["sourceSnapshot"].replace("20260827", "wrong")
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    return {"FAKE_INSTANCE_DESCRIPTOR": json.dumps(instance),
            "FAKE_DISK_DESCRIPTOR": json.dumps(disk),
            "FAKE_SNAPSHOT_DESCRIPTOR": json.dumps(snapshot)}


def assert_target_refusal_preserves_and_stops(closeout_env, completed):
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag-e0-closeout" in call
               for call in cloud_calls(closeout_env))


def test_nondefault_instance_is_used_for_every_cloud_operation(closeout_env):
    completed = run_driver(
        closeout_env,
        POST_E0_INSTANCE="thesis-fedcrag-e0-closeout",
        POST_E0_EXPECTED_SOURCE_SNAPSHOT="fedcrag-e0-closeout-20260827",
        FAKE_DISCOVERY_NAME="thesis-fedcrag-e0-closeout",
        FAKE_INSTANCE_DESCRIPTOR=json.dumps({
            **json.loads(closeout_env["env"]["FAKE_INSTANCE_DESCRIPTOR"]),
            "name": "thesis-fedcrag-e0-closeout",
        }),
        FAKE_STATUSES="RUNNING,TERMINATED",
    )
    assert completed.returncode == 0, completed.stderr
    calls = cloud_calls(closeout_env)
    target_calls = [line for line in calls if any(token in line for token in (
        "instances describe", "instances start", "instances stop",
        "compute ssh", "compute scp"))]
    assert target_calls
    assert all("thesis-fedcrag-e0-closeout" in line for line in target_calls)
    for name in ("target_instance.json", "target_disk.json", "target_snapshot.json"):
        assert (closeout_env["destination"] / "audit" / name).is_file()


@pytest.mark.parametrize("value", ["", "Bad_Name", "-leading", "trailing-", "a" * 64])
def test_invalid_instance_override_is_refused_before_cloud(closeout_env, value):
    completed = run_driver(closeout_env, POST_E0_INSTANCE=value)
    assert completed.returncode == 2
    assert cloud_calls(closeout_env) == []


def test_nondefault_instance_requires_expected_snapshot(closeout_env):
    completed = run_driver(closeout_env, POST_E0_INSTANCE="thesis-fedcrag-e0-closeout")
    assert completed.returncode == 2
    assert cloud_calls(closeout_env) == []


@pytest.mark.parametrize("mutation", [
    "wrong-name", "machine-type", "l4-type", "l4-count", "provisioning", "boot-disk",
    "disk-size", "disk-type", "disk-status", "source-snapshot",
])
def test_target_configuration_descriptor_mutations_refuse_and_stop(closeout_env, mutation):
    completed = run_driver(
        closeout_env,
        POST_E0_INSTANCE="thesis-fedcrag-e0-closeout",
        POST_E0_EXPECTED_SOURCE_SNAPSHOT="fedcrag-e0-closeout-20260827",
        FAKE_DISCOVERY_NAME="thesis-fedcrag-e0-closeout",
        FAKE_STATUSES="RUNNING,TERMINATED",
        **target_descriptor_env(closeout_env, mutation),
    )
    assert_target_refusal_preserves_and_stops(closeout_env, completed)


@pytest.mark.parametrize("failure", [
    {"FAKE_INSTANCE_DESCRIPTOR_FAIL": "1"},
    {"FAKE_DISK_DESCRIPTOR_FAIL": "1"},
    {"FAKE_SNAPSHOT_DESCRIPTOR_FAIL": "1"},
])
def test_target_configuration_failed_descriptor_refuses_and_stops(closeout_env, failure):
    completed = run_driver(
        closeout_env,
        POST_E0_INSTANCE="thesis-fedcrag-e0-closeout",
        POST_E0_EXPECTED_SOURCE_SNAPSHOT="fedcrag-e0-closeout-20260827",
        FAKE_DISCOVERY_NAME="thesis-fedcrag-e0-closeout",
        FAKE_STATUSES="RUNNING,TERMINATED",
        **failure,
    )
    assert_target_refusal_preserves_and_stops(closeout_env, completed)


def test_target_configuration_ambiguous_exact_name_discovery_refuses_before_vm_activity(closeout_env):
    completed = run_driver(
        closeout_env,
        POST_E0_INSTANCE="thesis-fedcrag-e0-closeout",
        POST_E0_EXPECTED_SOURCE_SNAPSHOT="fedcrag-e0-closeout-20260827",
        FAKE_DISCOVERY_ROWS="thesis-fedcrag-e0-closeout,zone-e0\nthesis-fedcrag-e0-closeout,zone-e1",
    )
    assert completed.returncode == 4
    assert_not_published(closeout_env)
    assert not any("instances start" in call or "instances stop" in call
                   for call in cloud_calls(closeout_env))


def test_records_workspace_bash_and_requires_bash_32_syntax(closeout_env):
    version = subprocess.run(["/bin/bash", "--version"], check=True,
                             capture_output=True, text=True).stdout
    assert "GNU bash" in version
    completed = run_driver(closeout_env, FAKE_STATUSES="TERMINATED,RUNNING,TERMINATED")
    assert completed.returncode == 0, completed.stderr
    copied_version = (closeout_env["destination"] / "audit" / "bash_version.txt").read_text()
    assert copied_version == version
    source = SCRIPT.read_text()
    assert "mapfile" not in source
    assert "declare -A" not in source


def test_success_snapshots_exactly_eleven_regular_files_and_stops_vm(closeout_env):
    completed = run_driver(closeout_env, FAKE_STATUSES="TERMINATED,RUNNING,TERMINATED")
    assert completed.returncode == 0, completed.stderr
    destination = closeout_env["destination"]
    for relative in ("artifacts/COMPLETE.json", "audit/validation_summary.json",
                     "audit/2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md",
                     "SOURCE_SHA256SUMS", "PACKAGE_SHA256SUMS"):
        assert (destination / relative).is_file()
    assert "instances start thesis-fedcrag" in "\n".join(cloud_calls(closeout_env))
    assert "instances stop thesis-fedcrag" in "\n".join(cloud_calls(closeout_env))
    assert any("compute ssh thesis-fedcrag" in call for call in cloud_calls(closeout_env))
    assert sum("compute scp" in call for call in cloud_calls(closeout_env)) >= 2
    assert (destination / "audit" / "source_pre_inventory.jsonl").is_file()
    assert (destination / "audit" / "source_post_inventory.jsonl").is_file()
    assert (destination / "audit" / "staged_inventory.jsonl").is_file()
    summary = json.loads((destination / "audit" / "validation_summary.json").read_text())
    assert summary["validated_rows"] == 11
    assert summary["legacy_schema_v1_per_round"] == "unavailable"
    assert len(summary["measured_total_row_runtimes"]) == 11
    validator_command = (destination / "audit" / "validator_command.txt").read_text()
    assert f"--execution_source_root {closeout_env['execution_source']}" in validator_command
    assert "--execution_interpreter_path /shared/FedCRAG/.venv/bin/python" in validator_command
    closeout = (destination / "audit" / "2026-08-25_E0_STRENGTHENED_VALIDATION_CLOSEOUT.md").read_text()
    for required in ("direction residuals", "Continuity boundaries", "tolerance ratios",
                     "Measured total row runtimes", "Schema-v1", "post-hoc", "paper-scale"):
        assert required in closeout
    assert not failed_attempts(closeout_env)


@pytest.mark.parametrize("state", ["different-head", "dirty"])
def test_remote_worker_refuses_nonexact_or_dirty_execution_source(closeout_env, state):
    execution_source = closeout_env["execution_source"]
    if state == "different-head":
        subprocess.run(
            ["git", "checkout", "--quiet", "--detach", "HEAD^"],
            cwd=execution_source, check=True)
    else:
        (execution_source / "untracked-closeout.txt").write_text("dirty\n")

    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")

    assert completed.returncode == 20
    assert_not_published(closeout_env)
    attempts = failed_attempts(closeout_env)
    assert (attempts[0] / "audit" / "execution_source_identity.txt").is_file()
    identity = (attempts[0] / "audit" / "execution_source_identity.txt").read_text()
    assert "execution_source_head_status=0" in identity
    assert "execution_source_status_status=0" in identity
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))


def test_started_vm_waits_for_ssh_before_first_scp(closeout_env):
    completed = run_driver(
        closeout_env,
        FAKE_STATUSES="TERMINATED,RUNNING,TERMINATED",
        FAKE_SSH_READINESS="unready,ready",
        POST_E0_SSH_READY_LIMIT="2",
        POST_E0_SSH_READY_SLEEP="0",
    )
    assert completed.returncode == 0, completed.stderr
    calls = cloud_calls(closeout_env)
    readiness = [index for index, call in enumerate(calls)
                 if "compute ssh thesis-fedcrag" in call and "--command true" in call]
    first_scp = next(index for index, call in enumerate(calls) if "compute scp" in call)
    assert len(readiness) == 2
    assert readiness[-1] < first_scp
    assert all("--project project-e0" in calls[index] and "--zone zone-e0" in calls[index]
               for index in readiness)
    assert all("--ssh-flag=-oBatchMode=yes" in calls[index]
               and "--ssh-flag=-oConnectTimeout=10" in calls[index]
               for index in readiness)


def test_permanently_unready_started_vm_preserves_attempt_stops_and_never_scps(closeout_env):
    completed = run_driver(
        closeout_env,
        FAKE_STATUSES="TERMINATED,RUNNING,TERMINATED",
        FAKE_SSH_READINESS="unready,unready",
        POST_E0_SSH_READY_LIMIT="2",
        POST_E0_SSH_READY_SLEEP="0",
    )
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    calls = cloud_calls(closeout_env)
    assert sum("compute ssh thesis-fedcrag" in call and "--command true" in call
               for call in calls) == 2
    assert not any("compute scp" in call for call in calls)
    assert any("instances stop thesis-fedcrag" in call for call in calls)


@pytest.mark.parametrize("values", ["", "zone-a,zone-b"])
def test_refuses_zero_or_two_matching_zones(closeout_env, values):
    completed = run_driver(closeout_env, FAKE_ZONES=values)
    assert completed.returncode == 4
    assert_not_published(closeout_env)
    assert not any("instances start" in call for call in cloud_calls(closeout_env))


def test_refuses_literal_unset_project(closeout_env):
    completed = run_driver(closeout_env, FAKE_PROJECT="(unset)")
    assert completed.returncode == 4
    assert_not_published(closeout_env)
    assert not cloud_calls(closeout_env)[1:]


def test_refuses_failed_project_query_before_any_instance_call(closeout_env):
    completed = run_driver(closeout_env, FAKE_CONFIG_FAIL="1")
    assert completed.returncode == 3
    assert_not_published(closeout_env)
    assert cloud_calls(closeout_env) == ["config get-value project"]


def test_refuses_validator_override_outside_explicit_test_mode(closeout_env):
    env = closeout_env["env"].copy()
    for key in ("POST_E0_TEST_MODE", "POST_E0_TEST_REMOTE_ROOT",
                "POST_E0_REMOTE_TMP", "POST_E0_TEST_EXECUTION_SOURCE_ROOT"):
        env.pop(key, None)
    env["POST_E0_VALIDATOR"] = "/tmp/not-a-validator"
    completed = subprocess.run(["/bin/bash", str(SCRIPT)], cwd=ROOT, env=env,
                               capture_output=True, text=True)
    assert completed.returncode == 2
    assert "test-only" in completed.stderr
    assert not cloud_calls(closeout_env)


def test_refuses_failed_zone_query_before_start(closeout_env):
    completed = run_driver(closeout_env, FAKE_LIST_FAIL="1")
    assert completed.returncode == 3
    assert_not_published(closeout_env)
    assert not any("instances start" in call for call in cloud_calls(closeout_env))


@pytest.mark.parametrize("validated_rows", [None, [], ["row-0"] * 11,
                         [f"wrong-{index}" for index in range(11)]])
def test_refuses_missing_empty_duplicate_or_wrong_completion_rows(
        closeout_env, validated_rows):
    complete = closeout_env["source"] / "COMPLETE.json"
    payload = json.loads(complete.read_text())
    if validated_rows is None:
        payload.pop("validated_rows")
    else:
        payload["validated_rows"] = validated_rows
    complete.write_text(json.dumps(payload))
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)


@pytest.mark.parametrize("status_rows", [
    "row-0\tVALIDATED\t1\n",
    "".join(f"row-{index}\tVALIDATED\t1\n" for index in range(11)) + "row-0\tVALIDATED\t1\n",
    "".join(f"row-{index}\tFAILED\t1\n" for index in range(11)),
])
def test_refuses_status_rows_that_do_not_bind_each_validated_run(closeout_env, status_rows):
    (closeout_env["source"] / "status.tsv").write_text(status_rows)
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)


def test_refuses_malformed_real_status_timestamp(closeout_env):
    status = closeout_env["source"] / "status.tsv"
    status.write_text(status.read_text().replace("2026-08-25T12:00:00Z", "not-a-timestamp", 1))
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)


@pytest.mark.parametrize("runtime", ["NaN", "inf", "0", "-1"])
def test_refuses_nonpositive_or_nonfinite_runtime(closeout_env, runtime):
    status = closeout_env["source"] / "status.tsv"
    status.write_text(status.read_text().replace("1.5", runtime, 1))
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)


@pytest.mark.parametrize("failure", ["validation", "copy", "checksum"])
def test_failure_gates_preserve_unique_attempt_and_refuse_publication(closeout_env, failure):
    completed = run_driver(
        closeout_env, POST_E0_TEST_FAIL=failure,
        FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))


def test_scientific_failure_retrieves_remote_audit_before_preservation(closeout_env):
    completed = run_driver(closeout_env, POST_E0_TEST_FAIL="validation",
                           FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    attempts = failed_attempts(closeout_env)
    assert len(attempts) == 1
    assert (attempts[0] / "audit" / "validation_failure.txt").is_file()
    assert (attempts[0] / "audit" / "source_pre_inventory.jsonl").is_file()
    assert "scientific refusal" in (attempts[0] / "audit" / "validator-row-0.stderr").read_text()
    assert (attempts[0] / "audit" / "validator_exit_status.tsv").read_text().startswith("row-0\t41")


def test_existing_canonical_destination_is_refused_before_start(closeout_env):
    closeout_env["destination"].mkdir(parents=True)
    completed = run_driver(closeout_env)
    assert completed.returncode == 5
    assert closeout_env["destination"].is_dir()
    assert not failed_attempts(closeout_env)
    assert not any("instances start" in call for call in cloud_calls(closeout_env))


def test_lock_loser_never_removes_held_sibling_lock(closeout_env):
    lock = Path(str(closeout_env["destination"]) + ".publication-lock")
    lock.mkdir(parents=True)
    completed = run_driver(closeout_env)
    assert completed.returncode == 5
    assert lock.is_dir()


def test_refuses_concurrent_destination_before_promotion_without_nesting(closeout_env):
    completed = run_driver(closeout_env, FAKE_CREATE_DEST_ON_STOP="1",
                           FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert (closeout_env["destination"] / "marker").read_text() == "foreign"
    assert not (closeout_env["destination"] / "2026-08-25.attempt").exists()
    assert failed_attempts(closeout_env)


@pytest.mark.parametrize("extra", [
    {"FAKE_STATUSES": "GARBAGE"},
    {"FAKE_STATUS_FAIL": "1"},
])
def test_malformed_or_failed_status_is_critical_but_never_publishes(closeout_env, extra):
    completed = run_driver(closeout_env, **extra)
    assert completed.returncode == 70
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))


def test_stop_command_failure_overrides_original_failure_as_critical(closeout_env):
    completed = run_driver(
        closeout_env, POST_E0_TEST_FAIL="validation", FAKE_STOP_FAIL="1",
        FAKE_STATUSES="RUNNING,RUNNING,RUNNING,RUNNING")
    assert completed.returncode == 70
    assert_not_published(closeout_env)


def test_bounded_shutdown_retries_never_publish_when_termination_not_observed(closeout_env):
    completed = run_driver(
        closeout_env, FAKE_STATUSES="RUNNING,RUNNING,RUNNING,RUNNING,RUNNING")
    assert completed.returncode == 70
    assert_not_published(closeout_env)
    calls = cloud_calls(closeout_env)
    assert sum("instances describe" in call and "--format=value(status)" in call
               for call in calls) <= 4
    assert sum("instances stop" in call for call in calls) <= 3


def test_symlink_source_is_refused_before_copy_and_cleanup_runs(closeout_env):
    shutil.rmtree(closeout_env["source"] / "row-0")
    (closeout_env["source"] / "row-0").symlink_to("row-1")
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))


def test_symlink_artifact_root_is_refused_before_copy_and_cleanup_runs(closeout_env, tmp_path):
    source = closeout_env["source"]
    replacement = tmp_path / "replacement-results"
    source.rename(replacement)
    source.symlink_to(replacement, target_is_directory=True)
    completed = run_driver(closeout_env, FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))
