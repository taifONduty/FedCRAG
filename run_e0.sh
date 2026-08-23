#!/usr/bin/env bash
# Frozen E0 correctness-attribution campaign.
# This script never provisions cloud resources. Run mode assumes an already
# authorized and active GPU machine.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$SCRIPT_DIR"

PYTHON=${PYTHON:-.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-"$SCRIPT_DIR/beir_data"}
E0_OUT=${E0_OUT:-"$(cd "$SCRIPT_DIR/.." && pwd -P)/FedCRAG_E0_RESULTS"}

SEED=42
MODEL=contriever
ROUNDS=5
LOCAL_EPOCHS=1
LORA_RANK=16
BATCH_SIZE=32
EVAL_BATCH_SIZE=256
SLICES=(nfcorpus fiqa scifact arguana)

E0_ROWS=(
  "e0-trainable-ab-uniform-capped-500|trainable-ab|uniform|capped-500|500"
  "e0-trainable-ab-uniform-full|trainable-ab|uniform|full|0"
  "e0-trainable-ab-rawmaxmin-capped-500|trainable-ab|rawmaxmin|capped-500|500"
  "e0-trainable-ab-rawmaxmin-full|trainable-ab|rawmaxmin|full|0"
  "e0-frozen-a-uniform-capped-500|frozen-a|uniform|capped-500|500"
  "e0-frozen-a-uniform-full|frozen-a|uniform|full|0"
  "e0-frozen-a-rawmaxmin-capped-500|frozen-a|rawmaxmin|capped-500|500"
  "e0-frozen-a-rawmaxmin-full|frozen-a|rawmaxmin|full|0"
  "e0-frozen-a-normmaxmin-capped-500|frozen-a|normmaxmin|capped-500|500"
  "e0-frozen-a-normmaxmin-full|frozen-a|normmaxmin|full|0"
)

build_command() {
  local run_id=$1
  local coordinate=$2
  local arm=$3
  local max_steps=$4
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
    IFS='|' read -r run_id coordinate arm regime max_steps <<< "$row"
    build_command "$run_id" "$coordinate" "$arm" "$max_steps"
    printf -v command '%q ' "${ROW_CMD[@]}"
    command=${command% }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$coordinate" "$arm" "$regime" "$max_steps" "$command"
  done
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

verify_all() {
  require_clean_provenance
  require_external_output_root
  [[ ${#E0_ROWS[@]} -eq 10 ]]
  print_manifest
  verify_environment
  printf 'E0 verification complete: 10 frozen commands; no run launched.\n'
}

run_all() {
  verify_all
  mkdir -p "$E0_OUT/logs"
  local row run_id coordinate arm regime max_steps run_out
  for row in "${E0_ROWS[@]}"; do
    IFS='|' read -r run_id coordinate arm regime max_steps <<< "$row"
    run_out="$E0_OUT/$run_id"
    if [[ -e "$run_out" ]]; then
      printf 'refusing to overwrite existing E0 run directory: %s\n' \
        "$run_out" >&2
      exit 2
    fi
    mkdir -p "$run_out"
    build_command "$run_id" "$coordinate" "$arm" "$max_steps"
    printf 'START %s\n' "$run_id"
    "${ROW_CMD[@]}" 2>&1 | tee "$E0_OUT/logs/$run_id.log"
    "$PYTHON" validate_e0.py "$run_out"
    printf 'VALIDATED %s\n' "$run_id"
  done
  printf 'E0 COMPLETE: all ten rows executed and contract-validated.\n'
}

case "${1:-}" in
  manifest)
    print_manifest
    ;;
  verify)
    verify_all
    ;;
  run)
    run_all
    ;;
  *)
    printf 'usage: %s {manifest|verify|run}\n' "$0" >&2
    exit 2
    ;;
esac

