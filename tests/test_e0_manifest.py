"""Contract tests for the frozen E0 attribution launcher.

``manifest`` and ``bash -n`` cannot see a guard that aborts before any row is
launched, so ``verify`` — the subcommand ``run`` and ``resume`` both begin
with — is exercised here for real, in a hermetic clean worktree.
"""
import os
import subprocess
import sys
from pathlib import Path

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

