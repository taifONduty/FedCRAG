#!/usr/bin/env bash
# Frozen E0 correctness-attribution campaign.
# This script never provisions cloud resources. Run mode assumes an already
# authorized and active GPU machine.
#
# E0 ATTRIBUTES CORRECTNESS; IT DOES NOT PRODUCE PAPER NUMBERS.
# ROUNDS=5 here, while the paper-scale federated matrix in run_w3.sh uses
# ROUNDS=15 at the same 500-step cap. Drift/BWT/forgetting accumulate with the
# round count, so an E0 drift number and a paper cell are different quantities
# and must not be compared or quoted as one another.
#
# The two step regimes below differ on ONE client only. At --batch_size 32 the
# per-round step counts of the four slices are roughly nfcorpus ~3455,
# fiqa 442, scifact 28, arguana 21 (derived from logs/f42.log rates, floor
# divided by the batch size). Only nfcorpus exceeds 500, so "capped-500" vs
# "full" toggles nfcorpus dominance and leaves the other three clients'
# workloads identical.
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-"$SCRIPT_DIR/beir_data"}
E0_OUT=${E0_OUT:-"$(cd "$SCRIPT_DIR/.." && pwd -P)/FedCRAG_E0_RESULTS"}

E0_MANIFEST="$E0_OUT/manifest.json"
E0_STATUS="$E0_OUT/status.tsv"
E0_COMPLETE="$E0_OUT/COMPLETE.json"
GPU_SAMPLE_SECONDS=5

SEED=42
MODEL=contriever
ROUNDS=5
LOCAL_EPOCHS=1
LORA_RANK=16
BATCH_SIZE=32
EVAL_BATCH_SIZE=256
SLICES=(nfcorpus fiqa scifact arguana)

E0_ROWS=(
  "e0-trainable-ab-uniform-capped-500|trainable-ab|uniform|capped-500|500|-"
  "e0-trainable-ab-uniform-full|trainable-ab|uniform|full|0|-"
  "e0-trainable-ab-rawmaxmin-capped-500|trainable-ab|rawmaxmin|capped-500|500|-"
  "e0-trainable-ab-rawmaxmin-full|trainable-ab|rawmaxmin|full|0|-"
  "e0-frozen-a-uniform-capped-500|frozen-a|uniform|capped-500|500|peft-init"
  "e0-frozen-a-uniform-full|frozen-a|uniform|full|0|peft-init"
  "e0-frozen-a-rawmaxmin-capped-500|frozen-a|rawmaxmin|capped-500|500|peft-init"
  "e0-frozen-a-rawmaxmin-full|frozen-a|rawmaxmin|full|0|peft-init"
  "e0-frozen-a-normmaxmin-capped-500|frozen-a|normmaxmin|capped-500|500|peft-init"
  "e0-frozen-a-normmaxmin-full|frozen-a|normmaxmin|full|0|peft-init"
  "e0-frozen-a-unitscale-uniform-full|frozen-a|uniform|full|0|unit"
)

# The declared size of the frozen campaign. Every count printed or checked
# below derives from E0_ROWS itself; this constant exists only so that an
# accidental edit to the grid is refused rather than silently launched.
E0_EXPECTED_ROWS=11

build_command() {
  local run_id=$1
  local coordinate=$2
  local arm=$3
  local max_steps=$4
  local row_scale=${5:--}
  local run_out="$E0_OUT/$run_id"

  ROW_CMD=(
    "$PYTHON" federated_forgetting.py
    --model "$MODEL"
    --slices "${SLICES[@]}"
    --seed "$SEED"
    --num_rounds "$ROUNDS"
    --local_epochs "$LOCAL_EPOCHS"
    --lora_rank "$LORA_RANK"
    --batch_size "$BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --max_steps_per_round "$max_steps"
    --lora_mode "$coordinate"
    --data_root "$DATA_ROOT"
    --save_states
    --no_grad_ckpt
    --out "$run_out"
  )

  # frozen-A rows declare their row scale explicitly. 'peft-init' keeps
  # A A^T = c^2 I at PEFT's own initialization scale (c ~ 0.5774), so the
  # frozen-A rows share the trainable-A+B rows' B->dW step scale and any
  # difference between the two coordinates is attributable to the
  # frozen-orthonormal-A coordinate rather than to a 1.73x rescale. That
  # coordinate bundles the freeze with A-spectrum whitening; the two are not
  # separated by this grid. The single 'unit' row isolates the rescale on its
  # own, in the full-work regime only.
  if [ "$coordinate" = "frozen-a" ]; then
    if [ "$row_scale" = "-" ] || [ -z "$row_scale" ]; then
      printf 'frozen-a row %s must declare a row scale\n' "$run_id" >&2
      exit 2
    fi
    ROW_CMD+=(--frozen_a_row_scale "$row_scale")
  elif [ "$row_scale" != "-" ] && [ -n "$row_scale" ]; then
    printf 'row %s declares a row scale but is not frozen-a\n' "$run_id" >&2
    exit 2
  fi

  case "$arm" in
    uniform)
      ;;
    rawmaxmin)
      ROW_CMD+=(--weighted --weight_by rawmaxmin)
      ;;
    normmaxmin)
      ROW_CMD+=(
        --weighted
        --weight_by normmaxmin
        --fedspan_step_policy median-active
        --fedspan_direction_policy minnorm
        --fedspan_active_abs_tol 1e-12
        --fedspan_active_rel_tol 1e-8
        --fedspan_mixture_norm_tol 1e-6
      )
      ;;
    *)
      printf 'unknown E0 arm: %s\n' "$arm" >&2
      exit 2
      ;;
  esac
}

print_manifest() {
  local row run_id coordinate arm regime max_steps command
  for row in "${E0_ROWS[@]}"; do
    IFS="|" read -r run_id coordinate arm regime max_steps row_scale <<< "$row"
    build_command "$run_id" "$coordinate" "$arm" "$max_steps" "$row_scale"
    printf -v command '%q ' "${ROW_CMD[@]}"
    command=${command% }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$coordinate" "$arm" "$regime" "$max_steps" "$command"
  done
}

# NUL-delimited row stream so argv tokens survive with no quoting ambiguity.
manifest_stream() {
  local row run_id coordinate arm regime max_steps token
  for row in "${E0_ROWS[@]}"; do
    IFS="|" read -r run_id coordinate arm regime max_steps row_scale <<< "$row"
    build_command "$run_id" "$coordinate" "$arm" "$max_steps" "$row_scale"
    printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0' \
      "ROW" "$run_id" "$coordinate" "$arm" "$regime" "$max_steps" \
      "${#ROW_CMD[@]}"
    for token in "${ROW_CMD[@]}"; do
      printf '%s\0' "$token"
    done
  done
}

build_manifest_file() {
  local destination=$1
  local commit stream
  commit=$(git rev-parse --short=12 HEAD)
  require_expected_row_count
  stream=$(mktemp)
  manifest_stream > "$stream"
  "$PYTHON" - "$commit" "$destination" "$stream" "$E0_EXPECTED_ROWS" <<'PY'
import json
import sys
import time

commit, destination, stream = sys.argv[1], sys.argv[2], sys.argv[3]
expected = int(sys.argv[4])
with open(stream, "rb") as handle:
    fields = [chunk.decode() for chunk in handle.read().split(b"\0")]
rows = []
index = 0
while index < len(fields):
    if fields[index] != "ROW":
        index += 1
        continue
    run_id, coordinate, arm, regime, max_steps = fields[index + 1:index + 6]
    count = int(fields[index + 6])
    rows.append({
        "run_id": run_id,
        "coordinate": coordinate,
        "arm": arm,
        "regime": regime,
        "max_steps": int(max_steps),
        "argv": fields[index + 7:index + 7 + count],
    })
    index += 7 + count
if len(rows) != expected:
    sys.exit(f"expected {expected} manifest rows, built {len(rows)}")
with open(destination, "w") as handle:
    json.dump({
        "schema": "fedcrag-e0-manifest/1",
        "commit": commit,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
    }, handle, indent=2, sort_keys=True)
PY
  rm -f "$stream"
}

require_manifest_unchanged() {
  local scratch
  scratch=$(mktemp)
  build_manifest_file "$scratch"
  "$PYTHON" - "$E0_MANIFEST" "$scratch" <<'PY'
import json
import sys

frozen = json.load(open(sys.argv[1]))
current = json.load(open(sys.argv[2]))
for field in ("schema", "commit", "rows"):
    if frozen.get(field) != current.get(field):
        sys.exit(f"frozen manifest {field} differs from the current tree; "
                 "refusing to resume a campaign that would mix contracts")
PY
  rm -f "$scratch"
}

write_resource_record() {
  local run_id=$1 run_out=$2 log=$3 samples=$4
  local started_wall_ns=$5 finished_wall_ns=$6
  local started_mono_ns=$7 finished_mono_ns=$8
  "$PYTHON" e0_resources.py write \
    --run-id "$run_id" --run-dir "$run_out" --log "$log" \
    --samples "$samples" \
    --started-wall-ns "$started_wall_ns" \
    --finished-wall-ns "$finished_wall_ns" \
    --started-mono-ns "$started_mono_ns" \
    --finished-mono-ns "$finished_mono_ns" --num-rounds "$ROUNDS"
}

timestamp_filter() {
  if [[ ${E0_SELFTEST_FAIL_STAGE:-} == "filter" ]]; then
    while IFS= read -r _; do :; done
    return 92
  fi
  "$PYTHON" -u e0_resources.py timestamp
}

timestamp_tee() {
  local log=$1
  if [[ ${E0_SELFTEST_FAIL_STAGE:-} == "tee" ]]; then
    while IFS= read -r _; do :; done
    return 93
  fi
  tee "$log"
}

# The production and self-test paths share this exact three-stage pipeline.
# PIPESTATUS is copied by the very next command, before any status can be lost.
run_timestamped_pipeline() {
  local log=$1
  shift
  "$@" 2>&1 | timestamp_filter | timestamp_tee "$log"
  E0_LAST_PIPELINE_STATUSES=("${PIPESTATUS[@]}")
  local index stage_names=(producer filter tee)
  for index in 0 1 2; do
    if [[ ${E0_LAST_PIPELINE_STATUSES[$index]} -ne 0 ]]; then
      printf 'E0 telemetry pipeline %s stage failed (statuses: %s %s %s)\n' \
        "${stage_names[$index]}" "${E0_LAST_PIPELINE_STATUSES[0]}" \
        "${E0_LAST_PIPELINE_STATUSES[1]}" \
        "${E0_LAST_PIPELINE_STATUSES[2]}" >&2
      return "${E0_LAST_PIPELINE_STATUSES[$index]}"
    fi
  done
}

append_status() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$E0_STATUS"
}

row_is_validated() {
  [[ -f "$E0_STATUS" ]] || return 1
  awk -F'\t' -v id="$1" '$1 == id && $2 == "VALIDATED" { found = 1 }
       END { exit found ? 0 : 1 }' "$E0_STATUS"
}

require_row_artifacts_absent() {
  local run_id=$1 artifact
  local blockers=()
  for artifact in \
      "$E0_OUT/$run_id" \
      "$E0_OUT/logs/$run_id.log" \
      "$E0_OUT/logs/$run_id.gpu" \
      "$E0_OUT/logs/$run_id.boundaries"; do
    if [[ -e "$artifact" ]]; then
      blockers+=("$artifact")
    fi
  done
  if [[ ${#blockers[@]} -eq 0 ]]; then
    return 0
  fi
  printf 'refusing to overwrite preserved artifacts for unvalidated row %s; '\
'remove every listed artifact before resume:\n' "$run_id" >&2
  for artifact in "${blockers[@]}"; do
    printf '  %s\n' "$artifact" >&2
  done
  return 2
}

validate_row() {
  "$PYTHON" validate_e0.py "$E0_OUT/$1" \
    --manifest "$E0_MANIFEST" --run_id "$1"
}

execute_row() {
  local run_id=$1 coordinate=$2 arm=$3 max_steps=$4
  local run_out="$E0_OUT/$run_id"
  local log="$E0_OUT/logs/$run_id.log"
  local samples="$E0_OUT/logs/$run_id.gpu"
  local started_wall_ns finished_wall_ns started_mono_ns finished_mono_ns
  local elapsed_seconds sampler_pid="" status=0

  mkdir -p "$run_out"
  build_command "$run_id" "$coordinate" "$arm" "$max_steps" "$row_scale"
  printf 'START %s\n' "$run_id"
  read -r started_wall_ns started_mono_ns \
    < <("$PYTHON" e0_resources.py clock --run-id "$run_id" \
      --log "$log" --event start)
  : > "$samples"
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          >> "$samples" 2>/dev/null || true
        sleep "$GPU_SAMPLE_SECONDS"
      done
    ) &
    sampler_pid=$!
  fi

  export FEDCRAG_E0_RUN_ID="$run_id"
  set +e
  run_timestamped_pipeline "$log" "${ROW_CMD[@]}"
  status=$?
  set -e

  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
  read -r finished_wall_ns finished_mono_ns \
    < <("$PYTHON" e0_resources.py clock --run-id "$run_id" \
      --log "$log" --event finish)
  elapsed_seconds=$((
    (finished_mono_ns - started_mono_ns) / 1000000000
  ))

  if [[ $status -ne 0 ]]; then
    append_status "$run_id" "FAILED" "$elapsed_seconds"
    printf 'FAILED %s (exit %s)\n' "$run_id" "$status" >&2
    exit "$status"
  fi

  if ! write_resource_record \
      "$run_id" "$run_out" "$log" "$samples" \
      "$started_wall_ns" "$finished_wall_ns" \
      "$started_mono_ns" "$finished_mono_ns"; then
    append_status "$run_id" "UNAUDITABLE" "$elapsed_seconds"
    printf 'UNAUDITABLE %s: no resource record was written\n' "$run_id" >&2
    exit 1
  fi
  if ! validate_row "$run_id"; then
    append_status "$run_id" "INVALID" "$elapsed_seconds"
    printf 'INVALID %s: contract validation failed; inspect and remove %s '\
'before resuming\n' "$run_id" "$run_out" >&2
    exit 1
  fi
  append_status "$run_id" "VALIDATED" "$elapsed_seconds"
  printf 'VALIDATED %s\n' "$run_id"
}

timing_selftest_producer() {
  local run_id=$FEDCRAG_E0_RUN_ID
  printf 'E0_ROUND_START %s 1/2\n' "$run_id"
  # Long enough for the hermetic Popen test to distinguish live streaming
  # from a filter/tee that emits only after the producer exits.
  sleep 1
  printf 'E0_ROUND_END %s 1/2\n' "$run_id"
  sleep 0.03
  printf 'E0_ROUND_START %s 2/2\n' "$run_id"
  sleep 0.03
  printf 'E0_ROUND_END %s 2/2\n' "$run_id"
  if [[ ${E0_SELFTEST_FAIL_STAGE:-} == "producer" ]]; then
    return 91
  fi
}

timing_selftest() {
  local scratch run_id run_out log samples
  local started_wall_ns finished_wall_ns started_mono_ns finished_mono_ns
  local status stage expected_index index
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/fedcrag-e0-timing.XXXXXX")
  run_id="e0-timing-selftest"
  run_out="$scratch/$run_id"
  log="$scratch/$run_id.log"
  samples="$scratch/$run_id.gpu"
  mkdir -p "$run_out"
  : > "$samples"
  export FEDCRAG_E0_RUN_ID="$run_id"

  read -r started_wall_ns started_mono_ns \
    < <("$PYTHON" e0_resources.py clock --run-id "$run_id" \
      --log "$log" --event start)
  set +e
  run_timestamped_pipeline "$log" timing_selftest_producer
  status=$?
  set -e
  read -r finished_wall_ns finished_mono_ns \
    < <("$PYTHON" e0_resources.py clock --run-id "$run_id" \
      --log "$log" --event finish)
  if [[ $status -ne 0 ]]; then
    printf 'timing self-test success pipeline failed\n' >&2
    rm -rf "$scratch"
    return 1
  fi
  if ! "$PYTHON" e0_resources.py write \
      --run-id "$run_id" --run-dir "$run_out" --log "$log" \
      --samples "$samples" \
      --started-wall-ns "$started_wall_ns" \
      --finished-wall-ns "$finished_wall_ns" \
      --started-mono-ns "$started_mono_ns" \
      --finished-mono-ns "$finished_mono_ns" \
      --num-rounds 2 >/dev/null; then
    rm -rf "$scratch"
    return 1
  fi
  if ! "$PYTHON" - "$run_out/e0_resources.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
assert all(value > 0 for value in record["round_ns"])
assert all(value > 0 for value in record["between_round_ns"])
assert record["pre_ns"] >= 0 and record["post_ns"] >= 0
assert (record["pre_ns"] + sum(record["round_ns"])
        + sum(record["between_round_ns"]) + record["post_ns"]
        == record["finished_mono_ns"] - record["started_mono_ns"])
PY
  then
    rm -rf "$scratch"
    return 1
  fi

  for stage in producer filter tee; do
    E0_SELFTEST_FAIL_STAGE=$stage
    export E0_SELFTEST_FAIL_STAGE
    : > "$scratch/fail-$stage.log"
    set +e
    run_timestamped_pipeline \
      "$scratch/fail-$stage.log" timing_selftest_producer >/dev/null 2>&1
    status=$?
    set -e
    unset E0_SELFTEST_FAIL_STAGE
    if [[ $status -eq 0 ]]; then
      printf 'timing self-test accepted %s failure\n' "$stage" >&2
      rm -rf "$scratch"
      return 1
    fi
    case "$stage" in
      producer) expected_index=0 ;;
      filter) expected_index=1 ;;
      tee) expected_index=2 ;;
    esac
    for index in 0 1 2; do
      if [[ $index -eq $expected_index ]]; then
        if [[ ${E0_LAST_PIPELINE_STATUSES[$index]} -eq 0 ]]; then
          printf 'timing self-test did not observe %s failure\n' "$stage" >&2
          rm -rf "$scratch"
          return 1
        fi
      elif [[ ${E0_LAST_PIPELINE_STATUSES[$index]} -ne 0 ]]; then
        printf 'timing self-test %s failure contaminated another stage\n' \
          "$stage" >&2
        rm -rf "$scratch"
        return 1
      fi
    done
    printf 'timing self-test refused %s failure\n' "$stage"
  done
  rm -rf "$scratch"
  printf 'timing self-test passed\n'
}

write_completion_marker() {
  "$PYTHON" - "$E0_COMPLETE" "$E0_MANIFEST" "$E0_STATUS" <<'PY'
import json
import sys
import time

complete_path, manifest_path, status_path = sys.argv[1:4]
manifest = json.load(open(manifest_path))
expected = [row["run_id"] for row in manifest["rows"]]
validated = set()
with open(status_path) as handle:
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[1] == "VALIDATED":
            validated.add(parts[0])
missing = [run_id for run_id in expected if run_id not in validated]
if missing:
    sys.exit(f"refusing to mark E0 complete; unvalidated rows: {missing}")
with open(complete_path, "w") as handle:
    json.dump({
        "schema": "fedcrag-e0-complete/1",
        "commit": manifest["commit"],
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validated_rows": expected,
    }, handle, indent=2, sort_keys=True)
PY
}

require_external_output_root() {
  local resolved
  resolved=$("$PYTHON" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$E0_OUT")
  case "$resolved" in
    "$SCRIPT_DIR"|"$SCRIPT_DIR"/*)
      printf 'E0_OUT must be outside the Git worktree: %s\n' "$resolved" >&2
      exit 2
      ;;
  esac
}

require_clean_provenance() {
  local dirty
  dirty=$(git status --porcelain --untracked-files=all)
  if [[ -n "$dirty" ]]; then
    printf 'E0 requires a clean Git tree. Dirty paths:\n%s\n' "$dirty" >&2
    exit 2
  fi
}

verify_environment() {
  if [[ ! -x "$PYTHON" ]]; then
    printf 'Python executable not found: %s\n' "$PYTHON" >&2
    exit 2
  fi
  "$PYTHON" -c 'import numpy, scipy, torch, peft, sentence_transformers'
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
    tests -q -p no:cacheprovider
}

require_expected_row_count() {
  if [[ ${#E0_ROWS[@]} -ne $E0_EXPECTED_ROWS ]]; then
    printf 'E0 grid drift: E0_ROWS holds %d rows but the frozen campaign '\
'declares %d. Restore the grid, or change E0_EXPECTED_ROWS deliberately.\n' \
      "${#E0_ROWS[@]}" "$E0_EXPECTED_ROWS" >&2
    exit 2
  fi
}

verify_all() {
  require_clean_provenance
  require_external_output_root
  require_expected_row_count
  print_manifest
  verify_environment
  printf 'E0 verification complete: %d frozen commands; no run launched.\n' \
    "${#E0_ROWS[@]}"
}

campaign() {
  local resuming=$1
  verify_all
  mkdir -p "$E0_OUT/logs"
  if [[ -e "$E0_COMPLETE" ]]; then
    printf 'E0 is already marked complete: %s\n' "$E0_COMPLETE" >&2
    exit 2
  fi
  if [[ $resuming -eq 1 ]]; then
    if [[ ! -f "$E0_MANIFEST" ]]; then
      printf 'no frozen manifest to resume from: %s\n' "$E0_MANIFEST" >&2
      exit 2
    fi
    require_manifest_unchanged
  else
    if [[ -e "$E0_MANIFEST" ]]; then
      printf 'E0 already started under %s; use "resume"\n' "$E0_MANIFEST" >&2
      exit 2
    fi
    build_manifest_file "$E0_MANIFEST"
    : > "$E0_STATUS"
  fi
  printf 'E0 manifest frozen at %s\n' "$E0_MANIFEST"

  local row run_id coordinate arm regime max_steps run_out
  for row in "${E0_ROWS[@]}"; do
    IFS="|" read -r run_id coordinate arm regime max_steps row_scale <<< "$row"
    run_out="$E0_OUT/$run_id"
    if row_is_validated "$run_id"; then
      # Re-check rather than trusting the marker: a resumed campaign must
      # still be able to state that every row satisfies the contract now.
      validate_row "$run_id" >/dev/null
      printf 'SKIP %s (already validated)\n' "$run_id"
      continue
    fi
    require_row_artifacts_absent "$run_id"
    execute_row "$run_id" "$coordinate" "$arm" "$max_steps"
  done
  write_completion_marker
  printf 'E0 COMPLETE: all %d rows executed and contract-validated (%s).\n' \
    "${#E0_ROWS[@]}" "$E0_COMPLETE"
}

main() {
  case "${1:-}" in
    manifest)
      print_manifest
      ;;
    verify)
      verify_all
      ;;
    run)
      campaign 0
      ;;
    resume)
      campaign 1
      ;;
    timing-selftest)
      timing_selftest
      ;;
    *)
      printf 'usage: %s {manifest|verify|run|resume|timing-selftest}\n' \
        "$0" >&2
      exit 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
