#!/usr/bin/env bash
# Block T1 on the GPU machine (registration/E3_PREREGISTRATION.md, section 14).
#   bash run_continual.sh bootstrap    # deps, data, tests
#   bash run_continual.sh manifests    # schedules A and B and the calibration stream
#   bash run_continual.sh profile      # one short run, timed
#   bash run_continual.sh calibrate    # the recipe grid on the calibration stream
#   bash run_continual.sh stage1       # all of the above, then power off if POWEROFF=1
#   bash run_continual.sh pilot        # the registered pilot (after 14.1), then power off
# Every run is validated before the chain continues; markers record how it ended.
set -Euo pipefail
cd "$(dirname "$0")"
PY="${CONTINUAL_PYTHON:-$PWD/.venv/bin/python}"
DATA="${CONTINUAL_DATA_ROOT:-$PWD/beir_data}"
OUT="${CONTINUAL_OUT:-$HOME/T1_20260921}"
MANIFESTS="$OUT/manifests"
MSSHIFT_COMMIT=07e873ce1da65ad5a42a092d4c87217e31f80518
MODE="${1:-stage1}"
mkdir -p "$OUT" "$MANIFESTS"
LOG="$OUT/chain.log"
say() { echo "[t1] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
finish() {
  say "$1"; touch "$OUT/$1.marker"; sync
  [ "${POWEROFF:-0}" = "1" ] && sudo poweroff
  [ "$1" = "DONE" ] && exit 0
  exit 1
}
trap 'finish FAILED' ERR

SCHEDULE_A='{"0":[0,1,2,3],"1":[2,0,3,1],"2":[1,3,0,2],"3":[3,2,1,0],"4":[0,3,1,2]}'
SCHEDULE_B='{"0":[3,1,0,2],"1":[1,3,2,0],"2":[2,0,1,3],"3":[0,2,3,1],"4":[1,0,2,3]}'
SCHEDULE_CAL='{"50":[0,1,2,3],"51":[2,0,3,1]}'
COUNTS_PRIMARY='{"train":1354,"guard":135,"test":350}'
COUNTS_CAL='{"train":2000,"guard":200,"test":500}'

bootstrap() {
  say "bootstrap"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [ -d .venv ] || uv venv --python 3.13 .venv
  uv pip install --python "$PY" -r requirements.txt
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('CUDA', torch.cuda.get_device_name(0))"
  mkdir -p "$DATA/msmarco-passage"
  for f in collection.tar.gz queries.tar.gz qrels.train.tsv qrels.dev.small.tsv; do
    [ -f "$DATA/msmarco-passage/$f" ] || [ -f "$DATA/msmarco-passage/${f%.tar.gz}.tsv" ] \
      || curl -sSL -o "$DATA/msmarco-passage/$f" "https://msmarco.z22.web.core.windows.net/msmarcoranking/$f"
  done
  ( cd "$DATA/msmarco-passage" && for t in collection queries; do
      [ -f "$t.tar.gz" ] && tar -xzf "$t.tar.gz" && rm "$t.tar.gz"; done; sha256sum *.tsv > SHA256SUMS )
  if [ ! -d "$DATA/ms-marco-shift" ]; then
    git clone -q https://github.com/naver/ms-marco-shift "$DATA/ms-marco-shift-src"
    git -C "$DATA/ms-marco-shift-src" checkout -q "$MSSHIFT_COMMIT"
    mkdir -p "$DATA/ms-marco-shift"
    cp -R "$DATA/ms-marco-shift-src/TRAIN" "$DATA/ms-marco-shift-src/EVAL" "$DATA/ms-marco-shift-src/license.txt" "$DATA/ms-marco-shift/"
    echo "$MSSHIFT_COMMIT" > "$DATA/ms-marco-shift/SOURCE_COMMIT"
  fi
  "$PY" -m pytest -q -p no:cacheprovider tests
}

manifests() {
  say "manifests"
  [ -f "$MANIFESTS/primary_B.json" ] || "$PY" experiences.py build --data_root "$DATA" --seed 1 \
    --clients 0 1 2 3 4 --experiences 4 --schedules "{\"A\":$SCHEDULE_A,\"B\":$SCHEDULE_B}" \
    --counts "$COUNTS_PRIMARY" --corpus_size 60000 --hard_k 5 --name primary --out "$MANIFESTS"
  [ -f "$MANIFESTS/calibration_cal.json" ] || "$PY" experiences.py build --data_root "$DATA" --seed 1 \
    --clients 5 --pseudo_clients 2 --experiences 4 --schedules "{\"cal\":$SCHEDULE_CAL}" \
    --counts "$COUNTS_CAL" --corpus_size 60000 --hard_k 5 --name calibration --out "$MANIFESTS"
  for m in "$MANIFESTS"/*.json; do "$PY" experiences.py verify --seed 1 --out "$m"; done
  ( cd "$MANIFESTS" && sha256sum *.json > SHA256SUMS && cat SHA256SUMS | tee -a "$LOG" )
}

run_one() {  # name manifest arm seed rounds [extra args]
  local name=$1 manifest=$2 arm=$3 seed=$4 rounds=$5; shift 5
  local dir="$OUT/$name"
  if [ -f "$dir/.validated" ]; then say "SKIP $name"; return 0; fi
  mkdir -p "$dir"
  say "START $name"
  local t0=$(date +%s)
  "$PY" continual_driver.py --manifest "$manifest" --data_root "$DATA" --arm "$arm" \
    --seed "$seed" --rounds "$rounds" --out "$dir" "$@" > "$dir/run.log" 2>&1
  echo "$(( $(date +%s) - t0 ))" > "$dir/wall_seconds"
  "$PY" validate_continual.py "$dir" > "$dir/validation.json"
  echo ok > "$dir/.validated"
  say "DONE $name ($(cat "$dir/wall_seconds") s)"
}

profile() {
  say "profile"
  run_one profile-cal-r1 "$MANIFESTS/calibration_cal.json" fedavg-replay 123 1
  "$PY" - "$OUT/profile-cal-r1" <<'PYEOF'
import glob, json, sys
d = sys.argv[1]; r = json.load(open(glob.glob(d + "/continual_*.json")[0]))
wall = int(open(d + "/wall_seconds").read())
print(json.dumps({"wall_seconds": wall, "clients": len(r["clients"]), "rounds": len(r["rounds"]),
                  "steps": [rec["steps"] for rec in r["rounds"]]}, indent=1))
PYEOF
}

calibrate() {
  say "calibrate"
  for rounds in 2 4 8; do for lr in 2e-5 5e-5; do
    run_one "cal-D-r${rounds}-lr${lr}" "$MANIFESTS/calibration_cal.json" fedavg-replay 123 "$rounds" --lr "$lr"
  done; done
  choice=$("$PY" - "$OUT" <<'PYEOF'
import glob, json, sys
out = sys.argv[1]; best = None
for d in sorted(glob.glob(out + "/cal-D-r*-lr*")):
    r = json.load(open(glob.glob(d + "/continual_*.json")[0]))
    a = r["summary"]["A"]; rounds = r["rounds_per_experience"]; lr = r["args"]["lr"]
    if best is None or a > best[0]: best = (a, rounds, lr)
print(json.dumps({"rounds": best[1], "lr": best[2], "A": best[0]}))
PYEOF
)
  echo "$choice" > "$OUT/calibration_recipe.json"; say "recipe $choice"
  rounds=$(echo "$choice" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["rounds"])')
  lr=$(echo "$choice" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["lr"])')
  for lam in 0.5 1.0 2.0; do
    run_one "cal-E-lam${lam}" "$MANIFESTS/calibration_cal.json" fedavg-replay-distill 123 "$rounds" --lr "$lr" --lambda_distill "$lam"
  done
  "$PY" - "$OUT" <<'PYEOF' | tee "$OUT/calibration_lambda.json"
import glob, json, sys
out = sys.argv[1]; runs = []
for d in sorted(glob.glob(out + "/cal-E-lam*")):
    r = json.load(open(glob.glob(d + "/continual_*.json")[0]))
    runs.append((r["args"]["lambda_distill"], r["summary"]["A"], r["summary"].get("G")))
best_a = max(a for _, a, _ in runs)
eligible = [x for x in runs if x[1] >= best_a - 0.005]
lam = min(eligible, key=lambda x: x[2])[0]
print(json.dumps({"lambda": lam, "runs": runs}))
PYEOF
}

pilot() {  # the registered pilot with the frozen recipe (14.1 must exist before this runs)
  say "pilot"
  rounds=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rounds"])' "$OUT/calibration_recipe.json")
  lr=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["lr"])' "$OUT/calibration_recipe.json")
  lam=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["lambda"])' "$OUT/calibration_lambda.json")
  for sched in A B; do
    run_one "pilot-frozen-$sched-s123" "$MANIFESTS/primary_$sched.json" frozen 123 "$rounds"
    for seed in 123 2024 3407; do
      for arm in local fedavg fedavg-replay; do
        run_one "pilot-$arm-$sched-s$seed" "$MANIFESTS/primary_$sched.json" "$arm" "$seed" "$rounds" --lr "$lr"
      done
      run_one "pilot-fedavg-replay-distill-$sched-s$seed" "$MANIFESTS/primary_$sched.json" \
        fedavg-replay-distill "$seed" "$rounds" --lr "$lr" --lambda_distill "$lam"
    done
  done
  "$PY" t1_gate.py "$OUT" | tee "$OUT/gate.json"
}

case "$MODE" in
  bootstrap) bootstrap ;;
  manifests) manifests ;;
  profile) profile ;;
  calibrate) calibrate ;;
  stage1) bootstrap; manifests; profile; calibrate; finish DONE ;;
  pilot) pilot; finish DONE ;;
  *) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
