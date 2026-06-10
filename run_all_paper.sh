#!/bin/bash
set -o pipefail
cd "$(dirname "$(realpath "$0")")"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results

PYTHON=.venv/bin/python
SLICES="nfcorpus fiqa scifact arguana"
PASS=0; FAIL=0

run() {
    local id=$1; shift
    echo ""; echo "======== START $id  $(date) ========"
    if "$@" 2>&1 | tee "logs/${id}.log"; then
        echo "======== DONE  $id  $(date) ========"
        PASS=$((PASS+1))
    else
        echo "======== FAIL  $id  exit=$?  $(date) ========"
        FAIL=$((FAIL+1))
    fi
}

# Phase 1: Pilot — centralized sequential forgetting
run p42  $PYTHON pilot_forgetting.py     --model bge-m3 --slices $SLICES --seed 42 --batch_size 8

# Phase 2: Controls — frozen / independent / joint ceilings
run c42  $PYTHON controls.py             --model bge-m3 --slices $SLICES --seed 42 --batch_size 8

# Phase 3: Federated — unweighted FedAvg
run f42  $PYTHON federated_forgetting.py --model bge-m3 --slices $SLICES --seed 42 --num_rounds 5 --batch_size 8 --save_states

# Phase 4: Federated — weighted FedAvg (corpus-size weighting, as in the original F4 run)
run fw42 $PYTHON federated_forgetting.py --model bge-m3 --slices $SLICES --seed 42 --num_rounds 5 --batch_size 8 --weighted --weight_by corpus --save_states

echo ""
echo "================================================"
echo "ALL DONE $(date)  passed=$PASS  failed=$FAIL"
echo "================================================"
