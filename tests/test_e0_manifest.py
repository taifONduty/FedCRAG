"""Contract tests for the frozen E0 attribution launcher.

``manifest`` and ``bash -n`` cannot see a guard that aborts before any row is
launched, so ``verify`` — the subcommand ``run`` and ``resume`` both begin
with — is exercised here for real, in a hermetic clean worktree.
"""
import os
import re
import select
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_e0.sh"


def manifest_rows():
    completed = subprocess.run(
        ["bash", str(SCRIPT), "manifest"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        run_id, coordinate, arm, regime, max_steps, command = line.split(
            "\t", maxsplit=5)
        rows.append({
            "run_id": run_id,
            "coordinate": coordinate,
            "arm": arm,
            "regime": regime,
            "max_steps": int(max_steps),
            "command": command,
        })
    return rows


def test_e0_manifest_is_the_attribution_grid_plus_row_scale_control():
    rows = manifest_rows()
    expected = {
        (coordinate, arm, regime, max_steps)
        for coordinate, arms in {
            "trainable-ab": ("uniform", "rawmaxmin"),
            "frozen-a": ("uniform", "rawmaxmin", "normmaxmin"),
        }.items()
        for arm in arms
        for regime, max_steps in (("capped-500", 500), ("full", 0))
    }
    observed = {
        (row["coordinate"], row["arm"], row["regime"], row["max_steps"])
        for row in rows
    }
    assert len(rows) == 11
    assert len({row["run_id"] for row in rows}) == 11
    assert observed == expected


def test_frozen_a_rows_declare_a_row_scale_and_isolate_the_rescale():
    # frozen-A rows must be step-scale matched to trainable-A+B (peft-init),
    # or they confound the freeze with a ~1.73x B->dW rescale; exactly one
    # unit-scale row exists to measure that rescale on its own.
    rows = manifest_rows()
    frozen = [row for row in rows if row["coordinate"] == "frozen-a"]
    trainable = [row for row in rows if row["coordinate"] == "trainable-ab"]
    assert frozen and trainable
    for row in trainable:
        assert "--frozen_a_row_scale" not in row["command"]
    scales = [
        "unit" if "--frozen_a_row_scale unit" in row["command"] else
        "peft-init" if "--frozen_a_row_scale peft-init" in row["command"]
        else None
        for row in frozen
    ]
    assert None not in scales
    assert scales.count("unit") == 1
    assert scales.count("peft-init") == len(frozen) - 1


def test_e0_commands_freeze_shared_scientific_contract():
    for row in manifest_rows():
        command = row["command"]
        for required in (
            "--model contriever",
            "--slices nfcorpus fiqa scifact arguana",
            "--seed 42",
            "--num_rounds 5",
            "--local_epochs 1",
            "--lora_rank 16",
            "--batch_size 32",
            "--eval_batch_size 256",
            "--save_states",
            "--no_grad_ckpt",
            f"--max_steps_per_round {row['max_steps']}",
            f"--lora_mode {row['coordinate']}",
        ):
            assert required in command
        assert "--allow_dirty_provenance" not in command
        assert "--fedspan_max_abs_delta_weight" not in command
        assert "--out " in command

        if row["arm"] == "uniform":
            assert "--weighted" not in command
            assert "--weight_by" not in command
        else:
            assert "--weighted" in command
            assert f"--weight_by {row['arm']}" in command

        if row["arm"] == "normmaxmin":
            for required in (
                "--fedspan_step_policy median-active",
                # D1: the direction solver is a declared part of the method,
                # and the driver refuses normmaxmin without it. Omitting it
                # here would abort the campaign at CLI parse.
                "--fedspan_direction_policy minnorm",
                "--fedspan_active_abs_tol 1e-12",
                "--fedspan_active_rel_tol 1e-8",
                "--fedspan_mixture_norm_tol 1e-6",
            ):
                assert required in command
            assert row["coordinate"] == "frozen-a"
            assert "--fedspan_step_norm" not in command
        else:
            assert "--fedspan_step_policy" not in command
            assert "--fedspan_step_norm" not in command
            # The driver rejects the direction policy on any other arm.
            assert "--fedspan_direction_policy" not in command


# ------------------------------------------------ documentation contract


def _readme_corrected_fedspan_command():
    readme = (ROOT / "README.md").read_text()
    match = re.search(
        r"Corrected FedSpan requires all of:\s*```bash\s*(.*?)\s*```",
        readme, flags=re.DOTALL)
    assert match, "README has no canonical corrected-FedSpan command"
    return shlex.split(match.group(1).replace("\\\n", " "))


def _flag_values(tokens, flags):
    values = {}
    for flag in flags:
        occurrences = [index for index, token in enumerate(tokens)
                       if token == flag]
        assert len(occurrences) == 1, f"expected exactly one {flag}"
        index = occurrences[0]
        assert index + 1 < len(tokens), f"{flag} has no value"
        values[flag] = tokens[index + 1]
    return values


def _prose(text):
    """Normalize Markdown wrapping without weakening wording assertions."""
    return " ".join(text.replace(">", " ").split()).lower()


def _method_flags(tokens):
    return {
        token for token in tokens
        if token.startswith(("--lora_", "--frozen_a_", "--fedspan_"))
        or token in {"--weighted", "--weight_by"}
    }


def test_readme_canonical_fedspan_command_matches_frozen_manifest_method():
    readme_tokens = _readme_corrected_fedspan_command()
    frozen = next(
        row for row in manifest_rows()
        if row["run_id"] == "e0-frozen-a-normmaxmin-full")
    manifest_tokens = shlex.split(frozen["command"])
    value_flags = {
        "--lora_rank",
        "--lora_mode",
        "--frozen_a_row_scale",
        "--weight_by",
        "--fedspan_step_policy",
        "--fedspan_direction_policy",
        "--fedspan_active_abs_tol",
        "--fedspan_active_rel_tol",
        "--fedspan_mixture_norm_tol",
    }

    expected = _flag_values(manifest_tokens, value_flags)
    assert expected == {
        "--lora_rank": "16",
        "--lora_mode": "frozen-a",
        "--frozen_a_row_scale": "peft-init",
        "--weight_by": "normmaxmin",
        "--fedspan_step_policy": "median-active",
        "--fedspan_direction_policy": "minnorm",
        "--fedspan_active_abs_tol": "1e-12",
        "--fedspan_active_rel_tol": "1e-8",
        "--fedspan_mixture_norm_tol": "1e-6",
    }
    assert _flag_values(readme_tokens, value_flags) == expected
    assert _method_flags(readme_tokens) == _method_flags(manifest_tokens)
    assert "--weighted" in readme_tokens
    assert "--save_states" in readme_tokens
    assert "--fedspan_max_abs_delta_weight" not in readme_tokens
    assert "--fedspan_max_abs_delta_weight" not in manifest_tokens


def test_readme_reports_completed_eleven_row_correctness_campaign_only():
    readme = (ROOT / "README.md").read_text()
    e0 = readme.split("### E0 correctness grid", 1)[1].split(
        "### API retrievers", 1)[0]
    status = readme.split("> ## Status:", 1)[1].split("---", 1)[0]

    status_prose = _prose(status)
    assert "eleven-row" in _prose(status + e0)
    assert "correctness campaign completed" in status_prose
    assert "strengthened post-hoc validation is pending" in status_prose
    assert "no paper-scale efficacy claim" in status_prose
    assert "ten-row" not in e0.lower()
    assert "exact ten commands" not in e0.lower()
    assert "no E0 run exists" not in readme


def test_readme_discloses_legacy_timing_limit_without_discarding_total_runtime():
    readme = _prose((ROOT / "README.md").read_text())

    assert "legacy e0 per-round timings are unavailable" in readme
    assert "total row runtime remains usable" in readme


def test_help_and_readme_require_an_explicit_frozen_a_row_scale():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "federated_forgetting.py"), "--help"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    help_text = completed.stdout.lower()
    readme = _prose((ROOT / "README.md").read_text())

    for text in (help_text, readme):
        assert "unit is the default" not in text
        assert "unit is an implicit" not in text
        assert "default 'unit'" not in text
        assert 'default "unit"' not in text
        assert not re.search(r"\bunit\b.{0,60}\b(?:default|implicit)\b", text)
        assert not re.search(r"\b(?:default|implicit)\b.{0,60}\bunit\b", text)
    assert "required" in help_text
    assert "choose explicitly" in help_text
    assert "unit" in help_text and "peft-init" in help_text
    assert "separate explicit choices" in readme
    assert "no safe implicit row-scale default" in readme


# -------------------------------------------------- the verify subcommand


def _clean_launcher_tree(tmp_path, source=None):
    """A one-commit Git worktree holding only the launcher.

    ``verify`` refuses a dirty tree, and a development checkout normally is
    one, so the subcommand can only be exercised somewhere hermetic.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "run_e0.sh").write_text(
        SCRIPT.read_text() if source is None else source)
    identity = ["-c", "user.email=e0@test", "-c", "user.name=E0 Test"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git"] + identity + ["add", "-A"], cwd=root, check=True)
    subprocess.run(["git"] + identity + ["commit", "-qm", "launcher"],
                   cwd=root, check=True)
    return root


def _stub_python(tmp_path):
    """Production Python for every call except the nested suite invocation.

    ``verify`` runs the whole test suite; letting it do so from inside that
    same suite would recurse without bound.
    """
    path = tmp_path / "python-stub"
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then\n'
        '  echo "stub: nested suite not re-run"\n'
        "  exit 0\n"
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n')
    path.chmod(0o755)
    return path


def _run_verify(tmp_path, source=None):
    root = _clean_launcher_tree(tmp_path, source=source)
    environment = dict(
        os.environ,
        PYTHON=str(_stub_python(tmp_path)),
        REAL_PYTHON=sys.executable,
        E0_OUT=str(tmp_path / "out"))
    return subprocess.run(
        ["bash", "run_e0.sh", "verify"], cwd=root,
        capture_output=True, text=True, env=environment)


def test_verify_accepts_the_grid_the_launcher_actually_ships(tmp_path):
    """The launcher and the grid must agree, or E0 cannot start at all."""
    expected = len(manifest_rows())
    completed = _run_verify(tmp_path)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert f"{expected} frozen commands" in completed.stdout
    printed = [line for line in completed.stdout.splitlines() if "\t" in line]
    assert len(printed) == expected
    assert not (tmp_path / "out").exists()


def test_verify_reports_a_row_count_drift_instead_of_exiting_silently(
        tmp_path):
    """A bare ``[[ ]]`` guard under ``set -e`` aborts with no output at all."""
    drifted = SCRIPT.read_text().replace(
        "E0_ROWS=(\n",
        'E0_ROWS=(\n  "e0-drift|frozen-a|uniform|full|0|unit"\n', 1)
    assert drifted != SCRIPT.read_text()
    completed = _run_verify(tmp_path, source=drifted)

    assert completed.returncode != 0
    assert completed.stderr.strip(), "the abort explained nothing"
    assert str(len(manifest_rows()) + 1) in completed.stderr


def test_launcher_never_hardcodes_the_grid_size_in_operator_text():
    """The COMPLETE record is a claim about the campaign; it must be true."""
    source = SCRIPT.read_text()
    assert "all ten rows" not in source
    assert "10 frozen commands" not in source
    assert '"${#E0_ROWS[@]}"' in source


def test_e0_launcher_is_syntax_clean_and_never_provisions_cloud():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = SCRIPT.read_text()
    assert "git status --porcelain" in source
    assert "gcloud compute" not in source
    assert "E0_OUT" in source


def test_launcher_control_artifact_guards_include_dangling_symlinks():
    source = SCRIPT.read_text()
    assert '[[ -e "$E0_COMPLETE" || -L "$E0_COMPLETE" ]]' in source
    assert '[[ -e "$E0_MANIFEST" || -L "$E0_MANIFEST" ]]' in source


def test_e0_launcher_timing_contract_uses_the_auditable_module():
    source = SCRIPT.read_text()
    assert "export PYTHONUNBUFFERED=1" in source
    assert '"$PYTHON" -u e0_resources.py timestamp' in source
    assert 'e0_resources.py clock' in source
    assert 'e0_resources.py boundary' not in source
    assert 'e0_resources.py write' in source
    assert '--boundaries "$boundaries"' not in source
    for scalar in (
        "--started-wall-ns", "--finished-wall-ns",
        "--started-mono-ns", "--finished-mono-ns",
    ):
        assert scalar in source
    assert "fedcrag-e0-resources/1" not in source
    assert "PIPESTATUS[@]" in source
    campaign = source.split("campaign() {", 1)[1].split("main() {", 1)[0]
    acquire = campaign.index('acquire_row_lock "$run_id"')
    precheck = campaign.index('require_row_artifacts_absent "$run_id"')
    execute = campaign.index(
        'execute_row "$run_id" "$coordinate" "$arm" "$max_steps"')
    final_release = campaign.rindex("release_row_lock")
    assert acquire < precheck < execute < final_release


def test_failed_row_retry_refuses_every_surviving_artifact_until_removed(
        tmp_path):
    run_id = "e0-failed-row"
    output = tmp_path / "out"
    logs = output / "logs"
    logs.mkdir(parents=True)
    status = output / "status.tsv"
    status.write_text(f"{run_id}\tFAILED\t1\t2026-08-26T00:00:00Z\n")
    artifacts = [
        output / run_id,
        logs / f"{run_id}.log",
        logs / f"{run_id}.gpu",
        logs / f"{run_id}.boundaries",
    ]
    artifacts[0].mkdir()
    dangling_target = tmp_path / "missing-failed-log"
    artifacts[1].symlink_to(dangling_target)
    for artifact in artifacts[2:]:
        artifact.write_text("preserved failed evidence\n")
    command = 'source "$1"; require_row_artifacts_absent "$2"'
    environment = dict(os.environ, E0_OUT=str(output), PYTHON=sys.executable)

    rejected = subprocess.run(
        ["bash", "-c", command, "retry-check", str(SCRIPT), run_id],
        capture_output=True, text=True, env=environment)

    assert rejected.returncode != 0
    for artifact in artifacts:
        assert str(artifact) in rejected.stderr
        assert artifact.exists() or artifact.is_symlink(), \
            "failed evidence was deleted automatically"
    assert "remove every listed artifact before resume" in rejected.stderr

    for artifact in reversed(artifacts):
        artifact.rmdir() if artifact.is_dir() else artifact.unlink()
    accepted = subprocess.run(
        ["bash", "-c", command, "retry-check", str(SCRIPT), run_id],
        capture_output=True, text=True, env=environment)
    assert accepted.returncode == 0, accepted.stderr
    assert "\tFAILED\t" in status.read_text()
    assert 'require_row_artifacts_absent "$run_id"' in SCRIPT.read_text()


def _row_lock_command():
    return (
        'source "$1"; acquire_row_lock "$2"; '
        'printf "LOCK_READY\\n"; IFS= read -r _; release_row_lock'
    )


def test_concurrent_row_lock_acquisition_has_exactly_one_owner(tmp_path):
    run_id = "e0-concurrent-row"
    output = tmp_path / "out"
    environment = dict(os.environ, E0_OUT=str(output), PYTHON=sys.executable)
    argv = [
        "bash", "-c", _row_lock_command(), "row-lock",
        str(SCRIPT), run_id,
    ]
    owner = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=environment, bufsize=1)
    assert owner.stdout is not None
    ready, _, _ = select.select([owner.stdout], [], [], 3)
    assert ready, "first row-lock process never reported ownership"
    assert owner.stdout.readline() == "LOCK_READY\n"

    refused = subprocess.run(
        argv, input="release\n", capture_output=True, text=True,
        env=environment, timeout=5)

    lock_path = output / "locks" / f"{run_id}.lock"
    assert refused.returncode != 0
    assert run_id in refused.stderr
    assert str(lock_path) in refused.stderr
    assert "explicitly remove" in refused.stderr
    assert owner.stdin is not None
    owner.stdin.write("release\n")
    owner.stdin.flush()
    stdout, stderr = owner.communicate(timeout=5)
    assert owner.returncode == 0, (stdout, stderr)
    assert not lock_path.exists() and not lock_path.is_symlink()


@pytest.mark.parametrize("stale_kind", ["directory", "dangling-symlink"])
def test_stale_or_symlink_row_lock_is_refused_without_overwrite(
        tmp_path, stale_kind):
    run_id = "e0-stale-row"
    output = tmp_path / "out"
    lock_root = output / "locks"
    lock_root.mkdir(parents=True)
    lock_path = lock_root / f"{run_id}.lock"
    if stale_kind == "directory":
        lock_path.mkdir()
        (lock_path / "owner").write_text("stale-owner\n")
    else:
        lock_path.symlink_to(tmp_path / "missing-lock-target")
    environment = dict(os.environ, E0_OUT=str(output), PYTHON=sys.executable)

    completed = subprocess.run([
        "bash", "-c", 'source "$1"; acquire_row_lock "$2"',
        "row-lock", str(SCRIPT), run_id,
    ], capture_output=True, text=True, env=environment, timeout=5)

    assert completed.returncode != 0
    assert run_id in completed.stderr
    assert str(lock_path) in completed.stderr
    assert "explicitly remove" in completed.stderr
    assert lock_path.exists() or lock_path.is_symlink()


def test_row_lock_exit_trap_releases_only_the_owned_lock(tmp_path):
    run_id = "e0-trap-row"
    output = tmp_path / "out"
    lock_path = output / "locks" / f"{run_id}.lock"
    logs = output / "logs"
    logs.mkdir(parents=True)
    scientific_artifact = logs / f"{run_id}.log"
    scientific_artifact.write_text("preserve me\n")
    environment = dict(os.environ, E0_OUT=str(output), PYTHON=sys.executable)

    completed = subprocess.run([
        "bash", "-c", 'source "$1"; acquire_row_lock "$2"; exit 7',
        "row-lock", str(SCRIPT), run_id,
    ], capture_output=True, text=True, env=environment, timeout=5)

    assert completed.returncode == 7
    assert not lock_path.exists() and not lock_path.is_symlink()
    assert scientific_artifact.read_text() == "preserve me\n"


def test_e0_launcher_timing_selftest_exercises_every_pipeline_stage():
    environment = dict(os.environ, PYTHON=sys.executable)
    completed = subprocess.run(
        ["bash", str(SCRIPT), "timing-selftest"], cwd=ROOT,
        capture_output=True, text=True, timeout=20, env=environment)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "timing self-test passed" in completed.stdout
    for stage in ("producer", "filter", "tee"):
        assert f"refused {stage} failure" in completed.stdout


def test_e0_launcher_timing_selftest_streams_before_process_completion():
    environment = dict(os.environ, PYTHON=sys.executable)
    process = subprocess.Popen(
        ["bash", str(SCRIPT), "timing-selftest"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, bufsize=1)
    assert process.stdout is not None
    # The producer sleeps for one second immediately after this first marker.
    # A filter or tee that waits for producer EOF cannot satisfy this bound.
    ready, _, _ = select.select([process.stdout], [], [], 0.6)
    assert ready, "producer/filter/tee buffered the first marker"
    first = process.stdout.readline()
    assert "E0_ROUND_START e0-timing-selftest 1/2" in first
    assert process.poll() is None, "output arrived only after process exit"

    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, (first + stdout, stderr)
    assert "timing self-test passed" in stdout
