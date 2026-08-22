#!/bin/bash
# E1 — headline seed completion (plan 2026-08-22 §2, BLOCKER 2).
# Seeds 123 & 2024 for the four headline cells:
#   maxmin capped (R=15, cap 500)      — repair table
#   n_k full-epoch (R=8, cap 0)        — catastrophic cell (seed42: BWT -0.0527)
#   maxmin full-epoch (R=8, cap 0)     — rescue (seed42: +0.0391)
#   uniform full-epoch (R=8, cap 0)    — saturation control (seed42: +0.0512)
# Order is decision-critical: one full replicate of every cell lands for seed 123
# before seed 2024 starts, so the Aug-30 replication verdict can fire early.
# SELF-STOPS THE VM on completion. Abort-with-stop on micro-gate failure.
set -o pipefail
cd "$(dirname "$(realpath "$0")")"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
mkdir -p logs

# GATE: 4-minute micro-test — env just came back from a snapshot restore;
# never burn arm-hours on an unvalidated environment.
$P federated_forgetting.py --model contriever --slices scifact arguana \
  --seed 7 --num_rounds 2 --batch_size 8 --max_steps_per_round 10 \
  --no_grad_ckpt --eval_batch_size 256 --weighted --weight_by maxmin \
  --out results_smoke 2>&1 | tee logs/e1-microtest.log
$P - <<'EOF' || { echo "MICROTEST FAILED - aborting E1" | tee logs/e1-ABORTED.marker; sudo shutdown -h now; exit 1; }
import json
d = json.load(open("results_smoke/federated_contriever_seed7_weighted-maxmin_r2.json"))
rw = d["round_weights"]
assert len(rw) == 2 and all(abs(sum(v) - 1) < 1e-3 for v in rw.values()), rw
assert d["BWT"] is not None
print("E1 MICROTEST PASS | round_weights:", rw)
EOF

run_arm () {
  tag="$1"; shift
  $P federated_forgetting.py --model contriever --slices nfcorpus fiqa scifact arguana \
    --batch_size 32 --eval_batch_size 256 --no_grad_ckpt --save_states --out results \
    "$@" 2>&1 | tee "logs/e1-${tag}.log"
  echo "E1 ARM DONE ${tag} $(date)" >> logs/e1-progress.marker
}

for SEED in 123 2024; do
  run_arm "maxmin-capped-s${SEED}"  --seed "$SEED" --num_rounds 15 --max_steps_per_round 500 --weighted --weight_by maxmin
  run_arm "nk-full-s${SEED}"        --seed "$SEED" --num_rounds 8  --max_steps_per_round 0   --weighted --weight_by examples
  run_arm "maxmin-full-s${SEED}"    --seed "$SEED" --num_rounds 8  --max_steps_per_round 0   --weighted --weight_by maxmin
  run_arm "uniform-full-s${SEED}"   --seed "$SEED" --num_rounds 8  --max_steps_per_round 0
done

# Bookkeeping: one row per new JSON into runs.tsv
for SEED in 123 2024; do
  for f in "federated_contriever_seed${SEED}_weighted-maxmin_r15" \
           "federated_contriever_seed${SEED}_weighted-examples_r8" \
           "federated_contriever_seed${SEED}_weighted-maxmin_r8" \
           "federated_contriever_seed${SEED}_unweighted_r8"; do
    BWT=$($P -c "import json;print(json.load(open('results/${f}.json'))['BWT'])" 2>/dev/null || echo NA)
    printf 'e1\t%s\t%s\tBWT=%s\n' "$(date +%F)" "$f" "$BWT" >> runs.tsv
  done
done

# Final safety: mirror everything new to the GCS archive before stopping
gsutil -m -q rsync -r results gs://fedcrag-archive-rokkh-503122/results_vm_final || true
gsutil -m -q rsync -r logs gs://fedcrag-archive-rokkh-503122/logs_vm_final || true
gsutil -q cp runs.tsv gs://fedcrag-archive-rokkh-503122/code_snapshot/runs.tsv || true

echo "E1 COMPLETE $(date)" | tee logs/e1-done.marker
sync
sudo shutdown -h now
