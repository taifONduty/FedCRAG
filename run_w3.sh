#!/bin/bash
# W3' campaign (planned 2026-08-18, supervisor review + decisions files):
#   smoke    -> 1-round scifact sanity run (minutes; run FIRST on any new machine)
#   controls -> headroom gate: contriever + contriever-msmarco (then: python check_headroom.py)
#   rt       -> R-T sequential forgetting, PRIMARY backbone, seeds 42/123/2024
#   rs       -> R-S federated matrix: 3 seeds x {unweighted, n_k, corpus-size}, R=15, states saved
#   all      -> controls, then rt, then rs (run smoke + gate manually first)
# Usage:  bash run_w3.sh <phase>          (PRIMARY=contriever by default;
#         override after the gate, e.g. PRIMARY=contriever-msmarco bash run_w3.sh rt)
# Disk note: rs saves adapter states every round: 9 runs x 15 rounds x ~5 files — budget ~10-20 GB.
set -o pipefail
cd "$(dirname "$(realpath "$0")")"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs results
PYTHON=${PYTHON:-.venv/bin/python}
SLICES="nfcorpus fiqa scifact arguana"
SEEDS=${SEEDS:-"42 123 2024"}
ROUNDS=${ROUNDS:-15}
BS=${BS:-32}
PRIMARY=${PRIMARY:-contriever}
PASS=0; FAIL=0

bwt_of() {  # $1 = result json -> primary-metric BWT or "-"
    "$PYTHON" - "$1" 2>/dev/null <<'EOF' || echo "-"
import json, sys
d = json.load(open(sys.argv[1]))
b = d.get("BWT") or {}
v = b.get("ndcg@10")
print(f"{v:+.4f}" if isinstance(v, (int, float)) else "-")
EOF
}

forg_of() {  # $1 = result json -> forgetting or "-" (pilot runs only)
    "$PYTHON" - "$1" 2>/dev/null <<'EOF' || echo "-"
import json, sys
d = json.load(open(sys.argv[1]))
b = d.get("forgetting") or {}
v = b.get("ndcg@10")
print(f"{v:+.4f}" if isinstance(v, (int, float)) else "-")
EOF
}

note_run() {  # id phase script model seed variant status json log
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$1" "$2" "$3" "$4" "$SLICES" "$5" "$6" "$7" \
        "$(bwt_of "$8")" "$(forg_of "$8")" "$8" "$9" >> runs.tsv
}

run() {  # id phase script model seed variant json cmd...
    local id=$1 phase=$2 script=$3 model=$4 seed=$5 variant=$6 json=$7; shift 7
    echo ""; echo "======== START $id  $(date) ========"
    if "$@" 2>&1 | tee "logs/${id}.log"; then
        echo "======== DONE  $id  $(date) ========"
        note_run "$id" "$phase" "$script" "$model" "$seed" "$variant" "done" "$json" "logs/${id}.log"
        PASS=$((PASS+1))
    else
        echo "======== FAIL  $id  exit=$?  $(date) ========"
        note_run "$id" "$phase" "$script" "$model" "$seed" "$variant" "FAILED" "$json" "logs/${id}.log"
        FAIL=$((FAIL+1))
    fi
}

phase=${1:-help}
case "$phase" in
smoke)
    run smoke-ct 0-smoke federated_forgetting.py contriever 42 smoke-1r \
        results_smoke/federated_contriever_seed42_unweighted_r1.json \
        $PYTHON federated_forgetting.py --slices scifact --model contriever \
        --num_rounds 1 --batch_size 8 --seed 42 --out results_smoke
    ;;
controls)
    for M in contriever contriever-msmarco; do
        run "ctl-${M}" 2-controls controls.py "$M" 42 frozen+indep+joint \
            "results/controls_${M}_seed42.json" \
            $PYTHON controls.py --model "$M" --slices $SLICES --seed 42 --batch_size $BS
    done
    echo ""; echo "== HEADROOM GATE =="
    for M in contriever contriever-msmarco; do
        $PYTHON check_headroom.py "results/controls_${M}_seed42.json" || true
    done
    ;;
rt)
    for s in $SEEDS; do
        run "rt-${PRIMARY}-${s}" 1-pilot pilot_forgetting.py "$PRIMARY" "$s" sequential \
            "results/pilot_${PRIMARY}_seed${s}.json" \
            $PYTHON pilot_forgetting.py --model "$PRIMARY" --slices $SLICES \
            --seed "$s" --batch_size $BS --eval_batch_size 256 --no_grad_ckpt
    done
    ;;
rs)
    for s in $SEEDS; do
        run "rs-${PRIMARY}-${s}-uniform" 3-federated federated_forgetting.py "$PRIMARY" "$s" "uniform-${ROUNDS}r" \
            "results/federated_${PRIMARY}_seed${s}_unweighted_r${ROUNDS}.json" \
            $PYTHON federated_forgetting.py --model "$PRIMARY" --slices $SLICES \
            --seed "$s" --num_rounds $ROUNDS --batch_size $BS --save_states \
            --eval_batch_size 256 --no_grad_ckpt --max_steps_per_round ${STEP_CAP:-500}
        run "rs-${PRIMARY}-${s}-nk" 3-federated federated_forgetting.py "$PRIMARY" "$s" "nk-${ROUNDS}r" \
            "results/federated_${PRIMARY}_seed${s}_weighted-examples_r${ROUNDS}.json" \
            $PYTHON federated_forgetting.py --model "$PRIMARY" --slices $SLICES \
            --seed "$s" --num_rounds $ROUNDS --batch_size $BS --save_states \
            --eval_batch_size 256 --no_grad_ckpt --max_steps_per_round ${STEP_CAP:-500} \
            --weighted --weight_by examples
        run "rs-${PRIMARY}-${s}-corpus" 3-federated federated_forgetting.py "$PRIMARY" "$s" "corpus-${ROUNDS}r" \
            "results/federated_${PRIMARY}_seed${s}_weighted-corpus_r${ROUNDS}.json" \
            $PYTHON federated_forgetting.py --model "$PRIMARY" --slices $SLICES \
            --seed "$s" --num_rounds $ROUNDS --batch_size $BS --save_states \
            --eval_batch_size 256 --no_grad_ckpt --max_steps_per_round ${STEP_CAP:-500} \
            --weighted --weight_by corpus
    done
    ;;
all)
    "$0" controls && "$0" rt && "$0" rs
    exit $?
    ;;
*)
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac

echo ""
echo "================================================"
echo "PHASE '$phase' DONE $(date)  passed=$PASS  failed=$FAIL"
echo "================================================"
