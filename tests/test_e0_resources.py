"""Auditable, monotonic resource telemetry for the E0 launcher."""
import copy
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e0_resources as resources  # noqa: E402


RUN_ID = "e0-test-row"


def _legacy_record():
    return {
        "schema": "fedcrag-e0-resources/1",
        "run_id": RUN_ID,
        "started_utc": "2026-08-26T00:00:00Z",
        "finished_utc": "2026-08-26T00:00:12Z",
        "elapsed_seconds": 12.0,
        "round_elapsed_seconds": [5.0, 7.0],
        "determinism_probe": "separate interpreter, same environment",
        "deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": None,
        "python_hash_seed": None,
        "torch_version": "test",
        "gpu_available": False,
        "peak_gpu_memory_mib": None,
        "gpu_memory_samples": 0,
    }


def _write_log(path, events, wall_values=None):
    walls = wall_values or list(range(10_000, 10_000 + len(events)))
    path.write_text("".join(
        f"{wall}\t{mono}\t{text}\n"
        for wall, (mono, text) in zip(walls, events)))
    return path


def _write_boundaries(path, run_id=RUN_ID, events=None):
    events = events or [
        ("start", 1_700_000_000_000_000_000, 100),
        ("finish", 1_700_000_001_000_000_000, 1_000),
    ]
    path.write_text("".join(
        f"E0_BOUNDARY\t{event}\t{run_id}\t{wall_ns}\t{mono_ns}\n"
        for event, wall_ns, mono_ns in events))
    return path


def _valid_evidence(tmp_path):
    log = _write_log(tmp_path / "row.log", [
        (200, f"E0_ROUND_START {RUN_ID} 1/2"),
        (350, f"E0_ROUND_END {RUN_ID} 1/2"),
        (500, f"E0_ROUND_START {RUN_ID} 2/2"),
        (900, f"E0_ROUND_END {RUN_ID} 2/2"),
    ])
    samples = tmp_path / "row.gpu"
    samples.write_text("100\n250\n175\n")
    boundaries_path = _write_boundaries(tmp_path / "row.boundaries")
    boundaries = dict(
        started_wall_ns=1_700_000_000_000_000_000,
        finished_wall_ns=1_700_000_001_000_000_000,
        started_mono_ns=100,
        finished_mono_ns=1_000,
        num_rounds=2,
    )
    return log, samples, boundaries_path, boundaries


def _record(tmp_path):
    log, samples, boundaries_path, boundaries = _valid_evidence(tmp_path)
    record = resources.build_resource_record(
        RUN_ID, log, samples, boundaries_path=boundaries_path,
        num_rounds=boundaries["num_rounds"])
    return record, log, samples, boundaries_path, boundaries


def test_timestamped_rounds_partition_monotonic_runtime_exactly(tmp_path):
    log, _, _, boundaries = _valid_evidence(tmp_path)

    timing = resources.parse_timestamped_rounds(
        log, RUN_ID, 2,
        boundaries["started_mono_ns"], boundaries["finished_mono_ns"])

    assert timing == {
        "pre_ns": 100,
        "round_ns": [150, 400],
        "between_round_ns": [150],
        "post_ns": 100,
    }
    assert (timing["pre_ns"] + sum(timing["round_ns"])
            + sum(timing["between_round_ns"]) + timing["post_ns"]
            == boundaries["finished_mono_ns"]
            - boundaries["started_mono_ns"])
    assert all(value > 0 for value in timing["round_ns"])


def test_wall_clock_jumps_never_change_monotonic_round_durations(tmp_path):
    events = [
        (200, f"E0_ROUND_START {RUN_ID} 1/2"),
        (350, f"E0_ROUND_END {RUN_ID} 1/2"),
        (500, f"E0_ROUND_START {RUN_ID} 2/2"),
        (900, f"E0_ROUND_END {RUN_ID} 2/2"),
    ]
    forward = _write_log(
        tmp_path / "forward.log", events,
        [10, 10**18, 10**18 + 1, 20])
    backward = _write_log(
        tmp_path / "backward.log", events,
        [10**18, 1, 10**17, 0])

    expected = resources.parse_timestamped_rounds(
        forward, RUN_ID, 2, 100, 1_000)
    assert resources.parse_timestamped_rounds(
        backward, RUN_ID, 2, 100, 1_000) == expected


@pytest.mark.parametrize("events", [
    [(200, f"E0_ROUND_START {RUN_ID} 1/2")],
    [(200, f"E0_ROUND_START {RUN_ID} 1/2"),
     (210, f"E0_ROUND_START {RUN_ID} 1/2"),
     (350, f"E0_ROUND_END {RUN_ID} 1/2"),
     (500, f"E0_ROUND_START {RUN_ID} 2/2"),
     (900, f"E0_ROUND_END {RUN_ID} 2/2")],
    [(200, f"E0_ROUND_START {RUN_ID} 1/2"),
     (350, f"E0_ROUND_START {RUN_ID} 2/2"),
     (500, f"E0_ROUND_END {RUN_ID} 1/2"),
     (900, f"E0_ROUND_END {RUN_ID} 2/2")],
    [(200, f"E0_ROUND_START {RUN_ID} 1/3"),
     (350, f"E0_ROUND_END {RUN_ID} 1/3"),
     (500, f"E0_ROUND_START {RUN_ID} 2/3"),
     (900, f"E0_ROUND_END {RUN_ID} 2/3")],
    [(200, "E0_ROUND_START another-row 1/2"),
     (350, "E0_ROUND_END another-row 1/2"),
     (500, "E0_ROUND_START another-row 2/2"),
     (900, "E0_ROUND_END another-row 2/2")],
    [(500, f"E0_ROUND_START {RUN_ID} 2/2"),
     (900, f"E0_ROUND_END {RUN_ID} 2/2"),
     (200, f"E0_ROUND_START {RUN_ID} 1/2"),
     (350, f"E0_ROUND_END {RUN_ID} 1/2")],
    [(200, f"E0_ROUND_START {RUN_ID} 1/2"),
     (200, f"E0_ROUND_END {RUN_ID} 1/2"),
     (500, f"E0_ROUND_START {RUN_ID} 2/2"),
     (900, f"E0_ROUND_END {RUN_ID} 2/2")],
    [(50, f"E0_ROUND_START {RUN_ID} 1/2"),
     (350, f"E0_ROUND_END {RUN_ID} 1/2"),
     (500, f"E0_ROUND_START {RUN_ID} 2/2"),
     (900, f"E0_ROUND_END {RUN_ID} 2/2")],
])
def test_schema_v2_refuses_ambiguous_or_impossible_marker_streams(
        tmp_path, events):
    log = _write_log(tmp_path / "bad.log", events)

    with pytest.raises(resources.ResourceValidationError):
        resources.parse_timestamped_rounds(log, RUN_ID, 2, 100, 1_000)


def test_schema_v2_record_replays_raw_timing_and_gpu_evidence(tmp_path):
    record, log, samples, boundaries_path, _ = _record(tmp_path)

    summary = resources.validate_resource_record(
        record, RUN_ID, 2, log, samples, boundaries_path)

    assert record["schema"] == "fedcrag-e0-resources/2"
    assert record["run_id"] == RUN_ID
    assert record["round_ns"] == [150, 400]
    assert record["between_round_ns"] == [150]
    assert record["gpu_memory_samples"] == 3
    assert record["peak_gpu_memory_mib"] == 250
    assert len(record["log_sha256"]) == 64
    assert len(record["samples_sha256"]) == 64
    assert len(record["boundaries_sha256"]) == 64
    assert summary["round_timing_valid"] is True
    assert summary["round_timing_status"] == "measured-monotonic"
    assert summary["round_elapsed_seconds"] == [1.5e-7, 4e-7]


@pytest.mark.parametrize(("field", "replacement"), [
    ("run_id", "e0-another-row"),
    ("pre_ns", 99),
    ("round_ns", [151, 400]),
    ("between_round_ns", [149]),
    ("post_ns", 99),
    ("elapsed_seconds", 123.0),
    ("gpu_memory_samples", 2),
    ("peak_gpu_memory_mib", 999),
    ("gpu_available", False),
    ("log_sha256", "0" * 64),
    ("samples_sha256", "0" * 64),
    ("boundaries_sha256", "0" * 64),
])
def test_schema_v2_refuses_mutated_json_claims(
        tmp_path, field, replacement):
    record, log, samples, boundaries_path, _ = _record(tmp_path)
    record[field] = replacement

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, log, samples, boundaries_path)


@pytest.mark.parametrize("evidence", ["log", "samples"])
def test_schema_v2_refuses_mutated_raw_evidence(tmp_path, evidence):
    record, log, samples, boundaries_path, _ = _record(tmp_path)
    path = log if evidence == "log" else samples
    path.write_bytes(path.read_bytes() + b"mutation\n")

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, log, samples, boundaries_path)


def test_schema_v1_never_publishes_plausible_round_values(tmp_path):
    record = _legacy_record()

    summary = resources.validate_resource_record(
        record, RUN_ID, 2, tmp_path / "absent.log",
        tmp_path / "absent.gpu")

    assert summary["elapsed_seconds"] == 12.0
    assert summary["round_timing_valid"] is False
    assert summary["round_timing_status"] == "legacy-buffered-unavailable"
    assert summary["round_elapsed_seconds"] is None


@pytest.mark.parametrize(("field", "value"), [
    ("elapsed_seconds", 0),
    ("elapsed_seconds", 11.0),
    ("round_elapsed_seconds", [12.0]),
    ("gpu_available", "no"),
    ("peak_gpu_memory_mib", 1),
    ("gpu_memory_samples", -1),
])
def test_schema_v1_still_refuses_malformed_totals_and_gpu_fields(
        tmp_path, field, value):
    record = _legacy_record()
    record[field] = value

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, tmp_path / "absent.log",
            tmp_path / "absent.gpu")


@pytest.mark.parametrize("field", [
    "run_id", "started_utc", "finished_utc", "determinism_probe",
    "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
    "cublas_workspace_config", "python_hash_seed", "torch_version",
    "gpu_available", "peak_gpu_memory_mib", "gpu_memory_samples",
])
def test_schema_v1_still_refuses_historical_structural_omissions(
        tmp_path, field):
    record = _legacy_record()
    del record[field]

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, tmp_path / "absent.log",
            tmp_path / "absent.gpu")


@pytest.mark.parametrize("failure", ["build", "validation", "replace"])
def test_atomic_writer_preserves_existing_destination_on_failure(
        monkeypatch, tmp_path, failure):
    _, log, samples, boundaries_path, boundaries = _record(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    destination = run_dir / "e0_resources.json"
    destination.write_text("old-record\n")

    def fail(*args, **kwargs):
        raise resources.ResourceValidationError(f"forced {failure} failure")

    if failure == "build":
        monkeypatch.setattr(resources, "build_resource_record", fail)
    elif failure == "validation":
        monkeypatch.setattr(resources, "validate_resource_record", fail)
    else:
        monkeypatch.setattr(resources.os, "replace", fail)

    with pytest.raises((resources.ResourceValidationError, OSError)):
        resources.write_resource_record(
            RUN_ID, run_dir, log, samples,
            boundaries_path=boundaries_path,
            num_rounds=boundaries["num_rounds"])

    assert destination.read_text() == "old-record\n"
    assert not list(run_dir.glob(".e0_resources.json.*.tmp"))


@pytest.mark.parametrize("boundary", ["start", "finish"])
def test_schema_v2_refuses_self_consistent_json_boundary_rewrite(
        tmp_path, boundary):
    record, log, samples, boundaries_path, _ = _record(tmp_path)
    if boundary == "start":
        record["started_mono_ns"] = 50
        record["pre_ns"] = 150
    else:
        record["finished_mono_ns"] = 1_050
        record["post_ns"] = 150
    record["elapsed_seconds"] = 9.5e-7

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, log, samples, boundaries_path)


def test_boundary_sidecar_is_the_raw_run_bound_clock_evidence(tmp_path):
    boundaries_path = _write_boundaries(tmp_path / "row.boundaries")

    assert resources.parse_boundary_evidence(boundaries_path, RUN_ID) == {
        "started_wall_ns": 1_700_000_000_000_000_000,
        "started_mono_ns": 100,
        "finished_wall_ns": 1_700_000_001_000_000_000,
        "finished_mono_ns": 1_000,
    }


def test_boundary_writer_flushes_and_fsyncs_each_run_bound_event(
        monkeypatch, tmp_path):
    path = tmp_path / "row.boundaries"
    wall_values = iter([10, 20])
    mono_values = iter([100, 200])
    fsynced = []
    monkeypatch.setattr(resources.time, "time_ns", lambda: next(wall_values))
    monkeypatch.setattr(
        resources.time, "monotonic_ns", lambda: next(mono_values))
    monkeypatch.setattr(resources.os, "fsync", lambda fd: fsynced.append(fd))

    assert resources.record_boundary_evidence(path, RUN_ID, "start") == \
        (10, 100)
    assert resources.record_boundary_evidence(path, RUN_ID, "finish") == \
        (20, 200)

    assert len(fsynced) == 2
    assert resources.parse_boundary_evidence(path, RUN_ID) == {
        "started_wall_ns": 10,
        "started_mono_ns": 100,
        "finished_wall_ns": 20,
        "finished_mono_ns": 200,
    }


@pytest.mark.parametrize("case", [
    "missing", "truncated", "duplicate", "wrong-run", "reversed",
    "malformed",
])
def test_boundary_sidecar_refuses_missing_or_ambiguous_evidence(
        tmp_path, case):
    path = tmp_path / "row.boundaries"
    if case == "missing":
        pass
    elif case == "truncated":
        _write_boundaries(path, events=[("start", 10, 100)])
    elif case == "duplicate":
        _write_boundaries(path, events=[
            ("start", 10, 100), ("start", 11, 110),
            ("finish", 20, 200)])
    elif case == "wrong-run":
        _write_boundaries(path, run_id="e0-other-row")
    elif case == "reversed":
        _write_boundaries(path, events=[
            ("finish", 20, 200), ("start", 10, 100)])
    else:
        path.write_text(
            f"E0_BOUNDARY\tstart\t{RUN_ID}\tnot-an-int\t100\n"
            f"E0_BOUNDARY\tfinish\t{RUN_ID}\t20\t200\n")

    with pytest.raises((resources.ResourceValidationError, OSError)):
        resources.parse_boundary_evidence(path, RUN_ID)


def test_schema_v2_refuses_boundary_sidecar_hash_mutation(tmp_path):
    record, log, samples, boundaries_path, _ = _record(tmp_path)
    boundaries_path.write_bytes(boundaries_path.read_bytes() + b"mutation\n")

    with pytest.raises(resources.ResourceValidationError):
        resources.validate_resource_record(
            record, RUN_ID, 2, log, samples, boundaries_path)


def test_write_cli_reports_extreme_wall_clock_concisely(tmp_path):
    log, samples, boundaries_path, _ = _valid_evidence(tmp_path)
    _write_boundaries(boundaries_path, events=[
        ("start", 10**100, 100),
        ("finish", 10**100 + 1_000_000_000, 1_000),
    ])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    completed = subprocess.run([
        sys.executable, str(resources.__file__), "write",
        "--run-id", RUN_ID, "--run-dir", str(run_dir),
        "--log", str(log), "--samples", str(samples),
        "--boundaries", str(boundaries_path), "--num-rounds", "2",
    ], capture_output=True, text=True)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith("e0_resources: ")
    assert "Traceback" not in completed.stderr
    assert len(completed.stderr.splitlines()) == 1


def test_write_cli_reports_missing_raw_evidence_without_traceback(tmp_path):
    log, samples, _, _ = _valid_evidence(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    completed = subprocess.run([
        sys.executable, str(resources.__file__), "write",
        "--run-id", RUN_ID, "--run-dir", str(run_dir),
        "--log", str(log), "--samples", str(samples),
        "--boundaries", str(tmp_path / "missing.boundaries"),
        "--num-rounds", "2",
    ], capture_output=True, text=True)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr.startswith("e0_resources: ")
    assert "Traceback" not in completed.stderr
    assert len(completed.stderr.splitlines()) == 1


def test_timestamp_filter_flushes_each_line_progressively():
    process = subprocess.Popen(
        [sys.executable, "-u", str(resources.__file__), "timestamp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1)
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write("first\n")
    process.stdin.flush()

    ready, _, _ = select.select([process.stdout], [], [], 2.0)
    assert ready, "timestamp filter buffered its first line"
    first = process.stdout.readline().rstrip("\n").split("\t", 2)
    assert len(first) == 3
    assert all(part.isdigit() for part in first[:2])
    assert first[2] == "first"

    process.stdin.write("second\n")
    process.stdin.close()
    remainder = process.stdout.read()
    assert process.wait(timeout=2) == 0
    assert remainder.endswith("\tsecond\n")


def test_clock_cli_returns_one_wall_and_monotonic_nanosecond_pair():
    completed = subprocess.run(
        [sys.executable, str(resources.__file__), "clock"],
        check=True, capture_output=True, text=True)
    fields = completed.stdout.strip().split("\t")
    assert len(fields) == 2
    assert all(field.isdigit() and int(field) > 0 for field in fields)
