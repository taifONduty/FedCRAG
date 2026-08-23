"""Static contract tests for the frozen E0 attribution launcher."""
import subprocess
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


def test_e0_launcher_is_syntax_clean_and_never_provisions_cloud():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    source = SCRIPT.read_text()
    assert "git status --porcelain" in source
    assert "gcloud compute" not in source
    assert "E0_OUT" in source

