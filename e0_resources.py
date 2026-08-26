"""Build and validate auditable E0 resource telemetry.

Schema v2 derives durations only from monotonic nanoseconds and binds every
published timing/GPU claim to the raw timestamped log and GPU sample stream.
Schema v1 remains readable for total-resource accounting, but its buffered
timestamps can never support a per-round timing claim.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

RESOURCE_SCHEMA_V1 = "fedcrag-e0-resources/1"
RESOURCE_SCHEMA_V2 = "fedcrag-e0-resources/2"
RESOURCE_FILENAME = "e0_resources.json"

_RUN_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_ROUND_MARKER = re.compile(
    r"E0_ROUND_(START|END) ([a-z0-9]+(?:-[a-z0-9]+)*) "
    r"([0-9]+)/([0-9]+)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_CLOCK_VALUE = (1 << 63) - 1


class ResourceValidationError(RuntimeError):
    """Resource evidence is malformed, ambiguous, or internally inconsistent."""


def _require(condition, message):
    if not condition:
        raise ResourceValidationError(message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _parse_ascii_decimal(text, field):
    """Parse one bounded, nonnegative base-10 integer from raw evidence."""
    _require(
        isinstance(text, str) and 1 <= len(text) <= 19
        and re.fullmatch(r"[0-9]+", text) is not None,
        f"{field} is not a bounded ASCII decimal integer")
    value = int(text)
    _require(value <= _MAX_CLOCK_VALUE,
             f"{field} exceeds the supported integer range")
    return value


def validate_run_id(run_id):
    _require(
        isinstance(run_id, str) and _RUN_ID.fullmatch(run_id) is not None
        and len(run_id) <= 128,
        "FEDCRAG_E0_RUN_ID must be 1-128 lowercase letters/digits joined "
        "by single hyphens")
    return run_id


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_from_wall_ns(wall_ns):
    seconds, nanoseconds = divmod(wall_ns, 1_000_000_000)
    try:
        moment = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ResourceValidationError(
            f"wall nanoseconds are outside the supported UTC range: "
            f"{wall_ns}") from exc
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + \
        f".{nanoseconds:09d}Z"


def _read_boundary_rows(boundaries_path):
    rows = []
    with Path(boundaries_path).open(
            "r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            _require(
                len(fields) == 5 and fields[0] == "E0_BOUNDARY",
                f"boundary evidence line {line_number} is malformed")
            _, event, run_id, wall_text, mono_text = fields
            _require(event in ("start", "finish"),
                     f"boundary evidence line {line_number} has invalid "
                     f"event {event!r}")
            _require(_RUN_ID.fullmatch(run_id) is not None
                     and len(run_id) <= 128,
                     f"boundary evidence line {line_number} has malformed "
                     "run identity")
            wall_ns = _parse_ascii_decimal(
                wall_text, f"boundary evidence line {line_number} wall clock")
            mono_ns = _parse_ascii_decimal(
                mono_text,
                f"boundary evidence line {line_number} monotonic clock")
            rows.append((event, run_id, wall_ns, mono_ns))
    return rows


def parse_boundary_evidence(boundaries_path, expected_run_id):
    """Read the exact raw start/finish clock pairs for one launcher row."""
    validate_run_id(expected_run_id)
    rows = _read_boundary_rows(boundaries_path)
    _require(len(rows) == 2,
             f"expected two boundary evidence events, found {len(rows)}")
    for index, expected_event in enumerate(("start", "finish")):
        event, run_id, _, _ = rows[index]
        _require(event == expected_event,
                 f"expected boundary event {expected_event!r} at position "
                 f"{index + 1}, found {event!r}")
        _require(run_id == expected_run_id,
                 f"boundary run id {run_id!r} does not match "
                 f"{expected_run_id!r}")
    _require(rows[1][3] > rows[0][3],
             "finish monotonic boundary must be greater than start")
    return {
        "started_wall_ns": rows[0][2],
        "started_mono_ns": rows[0][3],
        "finished_wall_ns": rows[1][2],
        "finished_mono_ns": rows[1][3],
    }


def boundary_path_for_log(log_path, run_id):
    """Locate independent boundary evidence from the bound row identity."""
    validate_run_id(run_id)
    return Path(log_path).parent / f"{run_id}.boundaries"


def _atomic_replace_text(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def record_boundary_evidence(boundaries_path, run_id, event):
    """Atomically persist and fsync one raw launcher boundary event."""
    validate_run_id(run_id)
    _require(event in ("start", "finish"),
             "boundary event must be 'start' or 'finish'")
    path = Path(boundaries_path)
    if event == "start":
        _require(not path.exists(),
                 f"boundary evidence already exists: {path}")
        previous = ""
    else:
        rows = _read_boundary_rows(path)
        _require(len(rows) == 1 and rows[0][0] == "start"
                 and rows[0][1] == run_id,
                 "finish boundary requires exactly one matching start event")
        previous = (
            f"E0_BOUNDARY\tstart\t{run_id}\t{rows[0][2]}\t{rows[0][3]}\n")
    wall_ns = time.time_ns()
    mono_ns = time.monotonic_ns()
    _atomic_replace_text(
        path, previous
        + f"E0_BOUNDARY\t{event}\t{run_id}\t{wall_ns}\t{mono_ns}\n")
    return wall_ns, mono_ns


def parse_timestamped_rounds(log_path, expected_run_id, num_rounds,
                             started_mono_ns, finished_mono_ns):
    """Replay a strict START/END marker stream into a monotonic partition."""
    validate_run_id(expected_run_id)
    _require(_is_int(num_rounds) and num_rounds > 0,
             "num_rounds must be a positive integer")
    _require(_is_int(started_mono_ns) and started_mono_ns >= 0,
             "started_mono_ns must be a nonnegative integer")
    _require(_is_int(finished_mono_ns)
             and finished_mono_ns > started_mono_ns,
             "finished_mono_ns must be greater than started_mono_ns")

    observed = []
    with Path(log_path).open("r", encoding="utf-8", errors="replace") as log:
        for line_number, line in enumerate(log, 1):
            fields = line.rstrip("\n").split("\t", 2)
            _require(
                len(fields) == 3,
                f"timestamped log line {line_number} lacks wall/monotonic "
                "nanosecond prefixes")
            wall_text, mono_text, text = fields
            _parse_ascii_decimal(
                wall_text, f"timestamped log line {line_number} wall clock")
            monotonic_ns = _parse_ascii_decimal(
                mono_text,
                f"timestamped log line {line_number} monotonic clock")
            match = _ROUND_MARKER.fullmatch(text)
            if match is None:
                _require(
                    "E0_ROUND_START" not in text
                    and "E0_ROUND_END" not in text,
                    f"timestamped log line {line_number} has a malformed "
                    "E0 round marker")
                continue
            kind, run_id, round_text, denominator_text = match.groups()
            observed.append((kind, run_id, _parse_ascii_decimal(
                                 round_text,
                                 f"timestamped log line {line_number} round"),
                             _parse_ascii_decimal(
                                 denominator_text,
                                 f"timestamped log line {line_number} "
                                 "denominator"), monotonic_ns,
                             line_number))

    expected_count = 2 * num_rounds
    _require(
        len(observed) == expected_count,
        f"expected {expected_count} E0 round markers, found {len(observed)}")
    for index, event in enumerate(observed):
        kind, run_id, round_number, denominator, monotonic_ns, line_number = \
            event
        expected_round = index // 2 + 1
        expected_kind = "START" if index % 2 == 0 else "END"
        _require(
            kind == expected_kind and round_number == expected_round,
            f"line {line_number}: expected E0_ROUND_{expected_kind} for "
            f"round {expected_round}, found {kind} {round_number}")
        _require(
            run_id == expected_run_id,
            f"line {line_number}: marker run id {run_id!r} does not match "
            f"{expected_run_id!r}")
        _require(
            denominator == num_rounds,
            f"line {line_number}: marker denominator {denominator} does not "
            f"match {num_rounds}")
        _require(
            started_mono_ns <= monotonic_ns <= finished_mono_ns,
            f"line {line_number}: marker monotonic time lies outside launcher "
            "boundaries")
        if index:
            _require(
                monotonic_ns >= observed[index - 1][4],
                f"line {line_number}: marker monotonic times are out of order")

    starts = [observed[index][4]
              for index in range(0, expected_count, 2)]
    ends = [observed[index][4]
            for index in range(1, expected_count, 2)]
    round_ns = []
    for round_number, (start, end) in enumerate(zip(starts, ends), 1):
        _require(end > start,
                 f"round {round_number} has a nonpositive duration")
        round_ns.append(end - start)
    between_round_ns = [
        starts[index + 1] - ends[index]
        for index in range(num_rounds - 1)
    ]
    _require(all(gap >= 0 for gap in between_round_ns),
             "round marker pairs overlap or cross")

    timing = {
        "pre_ns": starts[0] - started_mono_ns,
        "round_ns": round_ns,
        "between_round_ns": between_round_ns,
        "post_ns": finished_mono_ns - ends[-1],
    }
    partition = (timing["pre_ns"] + sum(timing["round_ns"])
                 + sum(timing["between_round_ns"]) + timing["post_ns"])
    _require(
        partition == finished_mono_ns - started_mono_ns,
        "monotonic timing partition does not equal launcher elapsed time")
    return timing


def _gpu_summary(samples_path):
    values = []
    with Path(samples_path).open("r", encoding="utf-8",
                                 errors="strict") as samples:
        for line_number, line in enumerate(samples, 1):
            value = line.strip()
            values.append(_parse_ascii_decimal(
                value, f"GPU sample line {line_number}"))
    return {
        "gpu_available": bool(values),
        "peak_gpu_memory_mib": max(values) if values else None,
        "gpu_memory_samples": len(values),
    }


def _determinism_metadata():
    # Keep the timestamp and clock subcommands lightweight: importing torch in
    # those streaming/boundary paths can take longer than a short producer and
    # would let its first lines accumulate before the filter starts reading.
    import torch

    return {
        "determinism_probe": "separate interpreter, same environment",
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
    }


def build_resource_record(run_id, log_path, samples_path, *,
                          started_wall_ns, finished_wall_ns,
                          started_mono_ns, finished_mono_ns, num_rounds):
    """Construct schema v2 exclusively from launcher boundaries/raw evidence."""
    validate_run_id(run_id)
    boundaries_path = boundary_path_for_log(log_path, run_id)
    boundaries = parse_boundary_evidence(boundaries_path, run_id)
    supplied = {
        "started_wall_ns": started_wall_ns,
        "finished_wall_ns": finished_wall_ns,
        "started_mono_ns": started_mono_ns,
        "finished_mono_ns": finished_mono_ns,
    }
    for field, value in supplied.items():
        _require(_is_int(value) and value >= 0,
                 f"{field} must be a nonnegative integer")
        _require(value == boundaries[field],
                 f"{field} does not match raw boundary evidence")
    timing = parse_timestamped_rounds(
        log_path, run_id, num_rounds, boundaries["started_mono_ns"],
        boundaries["finished_mono_ns"])
    gpu = _gpu_summary(samples_path)
    elapsed_ns = (boundaries["finished_mono_ns"]
                  - boundaries["started_mono_ns"])
    return {
        "schema": RESOURCE_SCHEMA_V2,
        "run_id": run_id,
        **boundaries,
        "started_utc": _utc_from_wall_ns(boundaries["started_wall_ns"]),
        "finished_utc": _utc_from_wall_ns(boundaries["finished_wall_ns"]),
        "elapsed_seconds": elapsed_ns / 1_000_000_000,
        **timing,
        "round_elapsed_seconds": [
            value / 1_000_000_000 for value in timing["round_ns"]],
        "log_sha256": _sha256(log_path),
        "samples_sha256": _sha256(samples_path),
        "boundaries_sha256": _sha256(boundaries_path),
        **gpu,
        **_determinism_metadata(),
    }


def _validate_common_resource_fields(record, expected_run_id, num_rounds):
    _require(isinstance(record, dict), "resource record must be an object")
    _require(record.get("run_id") == expected_run_id,
             f"resource run id {record.get('run_id')!r} does not match "
             f"{expected_run_id!r}")
    elapsed = record.get("elapsed_seconds")
    _require(_is_finite(elapsed) and float(elapsed) > 0,
             f"resource record has invalid elapsed_seconds {elapsed!r}")
    rounds = record.get("round_elapsed_seconds")
    _require(
        isinstance(rounds, list) and len(rounds) == num_rounds
        and all(_is_finite(value) and float(value) >= 0 for value in rounds),
        f"resource record does not carry one finite elapsed time for each of "
        f"the {num_rounds} rounds")
    _require(isinstance(record.get("deterministic_algorithms"), bool),
             "resource record does not record deterministic algorithms")
    _require(isinstance(record.get("gpu_available"), bool),
             "resource record has invalid GPU availability")
    sample_count = record.get("gpu_memory_samples")
    _require(_is_int(sample_count) and sample_count >= 0,
             "resource record has invalid GPU sample count")
    _require("peak_gpu_memory_mib" in record,
             "resource record omits peak GPU memory")
    peak = record.get("peak_gpu_memory_mib")
    if record["gpu_available"]:
        _require(_is_finite(peak) and float(peak) > 0 and sample_count > 0,
                 "resource record reports a GPU without positive peak/sample "
                 "evidence")
    else:
        _require(peak is None and sample_count == 0,
                 "resource record reports no GPU but carries peak/sample "
                 "evidence")
    return float(elapsed), (float(peak) if record["gpu_available"] else None)


def _validate_determinism_fields(record):
    _require(isinstance(record.get("determinism_probe"), str)
             and bool(record["determinism_probe"]),
             "resource record has invalid determinism_probe")
    for field in ("cudnn_deterministic", "cudnn_benchmark"):
        _require(isinstance(record.get(field), bool),
                 f"resource record has invalid {field}")
    for field in ("cublas_workspace_config", "python_hash_seed"):
        _require(field in record
                 and (record[field] is None or isinstance(record[field], str)),
                 f"resource record has invalid or missing {field}")
    _require(isinstance(record.get("torch_version"), str)
             and bool(record["torch_version"]),
             "resource record has invalid torch_version")


def _legacy_utc(value, field):
    _require(isinstance(value, str),
             f"schema v1 has invalid {field}")
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise ResourceValidationError(
            f"schema v1 has invalid {field}") from exc


def validate_resource_record(record, expected_run_id, num_rounds,
                             log_path, samples_path):
    """Validate a legacy total or fully replay schema-v2 raw evidence."""
    _require(_is_int(num_rounds) and num_rounds > 0,
             "num_rounds must be a positive integer")
    schema = record.get("schema") if isinstance(record, dict) else None
    _require(schema in (RESOURCE_SCHEMA_V1, RESOURCE_SCHEMA_V2),
             f"unsupported resource schema {schema!r}")
    elapsed, peak = _validate_common_resource_fields(
        record, expected_run_id, num_rounds)
    _validate_determinism_fields(record)
    common_summary = {
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": peak,
        "deterministic_algorithms": record["deterministic_algorithms"],
    }
    if schema == RESOURCE_SCHEMA_V1:
        started = _legacy_utc(record.get("started_utc"), "started_utc")
        finished = _legacy_utc(record.get("finished_utc"), "finished_utc")
        _require((finished - started).total_seconds() == elapsed,
                 "schema v1 elapsed_seconds does not reconcile with its UTC "
                 "boundaries")
        return {
            **common_summary,
            "round_timing_valid": False,
            "round_timing_status": "legacy-buffered-unavailable",
            "round_elapsed_seconds": None,
        }

    validate_run_id(expected_run_id)
    boundaries_path = boundary_path_for_log(log_path, expected_run_id)
    raw_boundaries = parse_boundary_evidence(
        boundaries_path, expected_run_id)
    for field, value in raw_boundaries.items():
        _require(record.get(field) == value,
                 f"schema v2 {field} does not match raw boundary evidence")
    integer_fields = (
        "started_wall_ns", "finished_wall_ns",
        "started_mono_ns", "finished_mono_ns", "pre_ns", "post_ns",
    )
    for field in integer_fields:
        _require(_is_int(record.get(field)) and record[field] >= 0,
                 f"schema v2 has invalid {field}")
    _require(record["finished_mono_ns"] > record["started_mono_ns"],
             "schema v2 monotonic boundaries are not increasing")
    for field in ("started_utc", "finished_utc"):
        _require(isinstance(record.get(field), str),
                 f"schema v2 has invalid {field}")
    _require(
        record["started_utc"] == _utc_from_wall_ns(record["started_wall_ns"])
        and record["finished_utc"]
        == _utc_from_wall_ns(record["finished_wall_ns"]),
        "schema v2 UTC provenance does not match its wall nanoseconds")

    replay = parse_timestamped_rounds(
        log_path, expected_run_id, num_rounds,
        raw_boundaries["started_mono_ns"],
        raw_boundaries["finished_mono_ns"])
    for field in ("pre_ns", "round_ns", "between_round_ns", "post_ns"):
        _require(record.get(field) == replay[field],
                 f"schema v2 {field} does not match raw log replay")
    _require(
        isinstance(record.get("round_ns"), list)
        and all(_is_int(value) and value > 0
                for value in record["round_ns"]),
        "schema v2 round_ns must contain positive integers")
    _require(
        isinstance(record.get("between_round_ns"), list)
        and len(record["between_round_ns"]) == num_rounds - 1
        and all(_is_int(value) and value >= 0
                for value in record["between_round_ns"]),
        "schema v2 between_round_ns is malformed")
    elapsed_ns = record["finished_mono_ns"] - record["started_mono_ns"]
    partition = (record["pre_ns"] + sum(record["round_ns"])
                 + sum(record["between_round_ns"]) + record["post_ns"])
    _require(partition == elapsed_ns,
             "schema v2 monotonic partition identity is invalid")
    expected_elapsed = elapsed_ns / 1_000_000_000
    expected_rounds = [value / 1_000_000_000
                       for value in replay["round_ns"]]
    _require(record["elapsed_seconds"] == expected_elapsed,
             "schema v2 elapsed_seconds is not its monotonic boundary delta")
    _require(record["round_elapsed_seconds"] == expected_rounds,
             "schema v2 round seconds do not match monotonic nanoseconds")

    for field, path in (("log_sha256", log_path),
                        ("samples_sha256", samples_path),
                        ("boundaries_sha256", boundaries_path)):
        value = record.get(field)
        _require(isinstance(value, str) and _SHA256.fullmatch(value),
                 f"schema v2 has invalid {field}")
        _require(value == _sha256(path),
                 f"schema v2 {field} does not match raw evidence")
    replay_gpu = _gpu_summary(samples_path)
    for field in ("gpu_available", "peak_gpu_memory_mib",
                  "gpu_memory_samples"):
        _require(record.get(field) == replay_gpu[field],
                 f"schema v2 {field} does not match raw GPU evidence")

    return {
        **common_summary,
        "round_timing_valid": True,
        "round_timing_status": "measured-monotonic",
        "round_elapsed_seconds": expected_rounds,
    }


def write_resource_record(run_id, run_dir, log_path, samples_path, **kwargs):
    """Atomically validate and replace one schema-v2 resource record."""
    run_dir = Path(run_dir)
    _require(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    destination = run_dir / RESOURCE_FILENAME
    temporary = None
    record = build_resource_record(
        run_id, log_path, samples_path, **kwargs)
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=run_dir,
                prefix=f".{RESOURCE_FILENAME}.", suffix=".tmp",
                delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("r", encoding="utf-8") as handle:
            serialized = json.load(handle)
        validate_resource_record(
            serialized, run_id, kwargs["num_rounds"], log_path, samples_path)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return record


def _timestamp_filter():
    for line in sys.stdin:
        sys.stdout.write(f"{time.time_ns()}\t{time.monotonic_ns()}\t{line}")
        sys.stdout.flush()


def _clock(run_id=None, log_path=None, event=None):
    persistence = (run_id, log_path, event)
    _require(
        all(value is None for value in persistence)
        or all(value is not None for value in persistence),
        "clock persistence requires --run-id, --log, and --event together")
    if run_id is None:
        wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
    else:
        wall_ns, mono_ns = record_boundary_evidence(
            boundary_path_for_log(log_path, run_id), run_id, event)
    print(f"{wall_ns}\t{mono_ns}", flush=True)


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("timestamp")
    clock = subparsers.add_parser("clock")
    clock.add_argument("--run-id")
    clock.add_argument("--log")
    clock.add_argument("--event", choices=("start", "finish"))
    write = subparsers.add_parser("write")
    write.add_argument("--run-id", required=True)
    write.add_argument("--run-dir", required=True)
    write.add_argument("--log", required=True)
    write.add_argument("--samples", required=True)
    write.add_argument("--started-wall-ns", required=True)
    write.add_argument("--finished-wall-ns", required=True)
    write.add_argument("--started-mono-ns", required=True)
    write.add_argument("--finished-mono-ns", required=True)
    write.add_argument("--num-rounds", required=True)
    return parser


def main():
    args = _parser().parse_args()
    try:
        if args.command == "timestamp":
            _timestamp_filter()
            return
        if args.command == "clock":
            _clock(args.run_id, args.log, args.event)
            return
        record = write_resource_record(
            args.run_id, args.run_dir, args.log, args.samples,
            started_wall_ns=_parse_ascii_decimal(
                args.started_wall_ns, "--started-wall-ns"),
            finished_wall_ns=_parse_ascii_decimal(
                args.finished_wall_ns, "--finished-wall-ns"),
            started_mono_ns=_parse_ascii_decimal(
                args.started_mono_ns, "--started-mono-ns"),
            finished_mono_ns=_parse_ascii_decimal(
                args.finished_mono_ns, "--finished-mono-ns"),
            num_rounds=_parse_ascii_decimal(
                args.num_rounds, "--num-rounds"))
    except (ResourceValidationError, OSError, UnicodeError,
            json.JSONDecodeError, OverflowError, ValueError) as exc:
        print(f"e0_resources: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"  elapsed={record['elapsed_seconds']}s "
        f"rounds={record['round_elapsed_seconds']} "
        f"peak_gpu_mib={record['peak_gpu_memory_mib']}")


if __name__ == "__main__":
    main()
