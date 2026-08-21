#!/bin/bash
# Repair arms (2026-08-21): conflict-aware max-min weighting, seed 42.
#   Arm 1: capped regime (500 steps/round, R=15) — head-to-head vs the n_k arm's
#          minority erosion (scifact E +0.0246 @ peak r3).
#   Arm 2: THE RESCUE TEST — full-epoch rounds, R=8, the catastrophic cell
#          (n_k baseline: BWT -0.053, scifact E +0.150). If maxmin turns this
#          positive, it is the paper's closing figure.
# SELF-STOPS THE VM on completion (sudo shutdown -h => instance TERMINATED,
# disk persists, billing stops).
set -o pipefail
cd "$(dirname "$(realpath "$0")")"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
mkdir -p logs

# GATE: 4-minute micro-test of the maxmin path. Abort (and stop the VM) on
# any failure — never burn arm-hours on unvalidated code.
$P federated_forgetting.py --model contriever --slices scifact arguana \
  --seed 7 --num_rounds 2 --batch_size 8 --max_steps_per_round 10 \
  --no_grad_ckpt --eval_batch_size 256 --weighted --weight_by maxmin \
  --out results_smoke 2>&1 | tee logs/repair-microtest.log
$P - <<'EOF' || { echo "MICROTEST FAILED - aborting repair arms" | tee logs/repair-ABORTED.marker; sudo shutdown -h now; exit 1; }
import json
d = json.load(open("results_smoke/federated_contriever_seed7_weighted-maxmin_r2.json"))
rw = d["round_weights"]
assert len(rw) == 2 and all(abs(sum(v) - 1) < 1e-3 for v in rw.values()), rw
assert d["BWT"] is not None
print("REPAIR MICROTEST PASS | round_weights:", rw)
EOF

$P federated_forgetting.py --model contriever --slices nfcorpus fiqa scifact arguana \
  --seed 42 --num_rounds 15 --batch_size 32 --eval_batch_size 256 --no_grad_ckpt \
  --max_steps_per_round 500 --save_states --weighted --weight_by maxmin --out results \
  2>&1 | tee logs/repair-capped-maxmin.log

$P federated_forgetting.py --model contriever --slices nfcorpus fiqa scifact arguana \
  --seed 42 --num_rounds 8 --batch_size 32 --eval_batch_size 256 --no_grad_ckpt \
  --max_steps_per_round 0 --save_states --weighted --weight_by maxmin --out results \
  2>&1 | tee logs/repair-fullepoch-maxmin.log

echo "REPAIR ARMS COMPLETE $(date)" | tee logs/repair-done.marker
sync
sudo shutdown -h now
