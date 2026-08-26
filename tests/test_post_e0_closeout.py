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


@pytest.fixture
def closeout_env(tmp_path):
    """Build a local-only E0 source and a logged fake gcloud executable."""
    source = tmp_path / "remote-e0-results"
    source.mkdir()
    (source / "COMPLETE.json").write_text(json.dumps({
        "commit": EXPECTED_COMMIT,
        "rows": [f"row-{index}" for index in range(11)],
    }))
    (source / "manifest.json").write_text(json.dumps({
        "commit": EXPECTED_COMMIT,
        "rows": [
            {"run_id": f"row-{index}", "commit": EXPECTED_COMMIT}
            for index in range(11)
        ],
    }))
    for index in range(11):
        row = source / f"row-{index}"
        row.mkdir()
        (row / "result.json").write_text("{}\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    status_file = tmp_path / "statuses"
    status_file.write_text("RUNNING")
    fake_gcloud = bin_dir / "gcloud"
    fake_gcloud.write_text("""#!/bin/sh
set -eu
printf '%s\\n' \"$*\" >> \"$FAKE_GCLOUD_LOG\"
if [ \"$1\" = config ]; then
  printf '%s\\n' \"${FAKE_PROJECT:-project-e0}\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = list ]; then
  [ \"${FAKE_LIST_FAIL:-0}\" = 0 ] || exit 1
  printf '%s' \"${FAKE_ZONES-zone-e0}\" | tr ',' '\\n'
  printf '\\n'
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = describe ]; then
  [ \"${FAKE_STATUS_FAIL:-0}\" = 0 ] || exit 1
  values=$(cat \"$FAKE_STATUS_FILE\")
  value=${values%%,*}
  if [ \"$values\" != \"$value\" ]; then
    printf '%s' \"${values#*,}\" > \"$FAKE_STATUS_FILE\"
  fi
  printf '%s\\n' \"$value\"
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = start ]; then
  [ \"${FAKE_START_FAIL:-0}\" = 0 ] || exit 1
  exit 0
fi
if [ \"$1\" = compute ] && [ \"$2\" = instances ] && [ \"$3\" = stop ]; then
  [ \"${FAKE_STOP_FAIL:-0}\" = 0 ] || exit 1
  exit 0
fi
printf '%s\\n' \"unexpected fake gcloud command: $*\" >&2
exit 99
""")
    fake_gcloud.chmod(fake_gcloud.stat().st_mode | stat.S_IXUSR)

    validator = tmp_path / "validator"
    validator.write_text("""#!/bin/sh
if [ \"${POST_E0_TEST_FAIL:-}\" = validation ]; then exit 41; fi
printf '{"validated_rows": 11, "status": "pass"}\\n'
""")
    validator.chmod(validator.stat().st_mode | stat.S_IXUSR)

    destination = tmp_path / "post_e0_audit" / "2026-08-25"
    env = os.environ.copy()
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "FAKE_GCLOUD_LOG": str(gcloud_log),
        "FAKE_STATUS_FILE": str(status_file),
        "POST_E0_DEST": str(destination),
        "POST_E0_TEST_MODE": "1",
        "POST_E0_TEST_REMOTE_ROOT": str(source),
        "POST_E0_TEST_VALIDATOR": str(validator),
        "POST_E0_RETRY_SLEEP": "0",
    })
    return {"env": env, "destination": destination, "log": gcloud_log,
            "source": source}


def run_driver(closeout_env, **extra_env):
    assert SCRIPT.is_file(), "RED: post_e0_closeout.sh has not been created"
    env = closeout_env["env"].copy()
    env.update({key: str(value) for key, value in extra_env.items()})
    if "FAKE_STATUSES" in extra_env:
        Path(env["FAKE_STATUS_FILE"]).write_text(str(extra_env["FAKE_STATUSES"]))
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
    assert not failed_attempts(closeout_env)


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


@pytest.mark.parametrize("failure", ["validation", "copy", "checksum"])
def test_failure_gates_preserve_unique_attempt_and_refuse_publication(closeout_env, failure):
    completed = run_driver(
        closeout_env, POST_E0_TEST_FAIL=failure,
        FAKE_STATUSES="RUNNING,TERMINATED")
    assert completed.returncode == 20
    assert_not_published(closeout_env)
    assert any("instances stop thesis-fedcrag" in call for call in cloud_calls(closeout_env))


def test_existing_canonical_destination_is_refused_before_start(closeout_env):
    closeout_env["destination"].mkdir(parents=True)
    completed = run_driver(closeout_env)
    assert completed.returncode == 5
    assert closeout_env["destination"].is_dir()
    assert not failed_attempts(closeout_env)
    assert not any("instances start" in call for call in cloud_calls(closeout_env))


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
    assert sum("instances describe" in call for call in calls) <= 4
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
