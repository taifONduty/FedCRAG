#!/usr/bin/env bash
# Block T1 on the GPU machine (registration/E3_PREREGISTRATION.md, section 14).
#   bash run_continual.sh bootstrap    # deps, data, tests
#   bash run_continual.sh manifests    # schedules A and B and the calibration stream
#   bash run_continual.sh profile      # one short run, timed
#   bash run_continual.sh calibrate    # the recipe grid on the calibration stream
#   bash run_continual.sh stage1       # all of the above, then power off if POWEROFF=1
#   EXPECT_COMMIT=<sha> bash run_continual.sh pilot   # the registered pilot (after 14.1)
#   CONTINUAL_OUT=<dir> CONTINUAL_MANIFESTS=<T1 manifests> EXPECT_COMMIT=<sha> \
#     bash run_continual.sh dev                        # the development study (section 15)
#   CONTINUAL_OUT=<dir> EXPECT_COMMIT=<sha> bash run_continual.sh lotte-profile   # one short LoTTE run, timed
#   CONTINUAL_OUT=<dir> EXPECT_COMMIT=<sha> bash run_continual.sh lotte  # the LoTTE block (after 16.1)
#   CONTINUAL_OUT=<dir> CONTINUAL_MANIFESTS=<T1 manifests> EXPECT_COMMIT=<sha> \
#     bash run_continual.sh rar-dev                    # rank-anchored replay development (section 17)
# The pilot refuses to start unless the repository is at EXPECT_COMMIT with a clean tree and
# the manifest digests match; every run records its exit status, and an interrupted run is
# never silently resumed.
# Every run is validated before the chain continues; markers record how it ended.
set -Euo pipefail
cd "$(dirname "$0")"
PY="${CONTINUAL_PYTHON:-$PWD/.venv/bin/python}"
DATA="${CONTINUAL_DATA_ROOT:-$PWD/beir_data}"
OUT="${CONTINUAL_OUT:-$HOME/T1_20260921}"
MANIFESTS="${CONTINUAL_MANIFESTS:-$OUT/manifests}"
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
COUNTS_CAL="$COUNTS_PRIMARY"   # calibrate on a stream shaped like the pilot

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
  if [ -f "$dir/.running" ]; then
    say "$name was interrupted on $(cat "$dir/.running"); move it aside before resuming"
    finish REFUSED
  fi
  mkdir -p "$dir"
  say "START $name"
  date -u +%FT%TZ > "$dir/.running"
  local t0=$(date +%s)
  local rc=0
  "$PY" continual_driver.py --manifest "$manifest" --data_root "$DATA" --arm "$arm" \
    --seed "$seed" --rounds "$rounds" --out "$dir" "$@" > "$dir/run.log" 2>&1 || rc=$?
  echo "$(( $(date +%s) - t0 ))" > "$dir/wall_seconds"
  echo "$rc" > "$dir/exit_status"
  [ "$rc" = "0" ] || { say "FAILED $name (driver exit $rc)"; finish FAILED; }
  "$PY" validate_continual.py "$dir" > "$dir/validation.json" || {
    say "FAILED $name (validation)"; finish FAILED; }
  rm -f "$dir/.running"
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

require_approved_commit() {
  [ -n "${EXPECT_COMMIT:-}" ] || {
    say "EXPECT_COMMIT is not set: the pilot runs only from the approved commit"; finish REFUSED; }
  local have; have=$(git rev-parse HEAD 2>/dev/null || echo unknown)
  [ "$have" = "$EXPECT_COMMIT" ] || {
    say "repository is at $have, not the approved $EXPECT_COMMIT"; finish REFUSED; }
  git diff --quiet HEAD 2>/dev/null || { say "working tree is dirty"; finish REFUSED; }
  say "running at approved commit $have"
}

require_manifests() {
  [ -f "$MANIFESTS/SHA256SUMS" ] || { say "no manifest digests to check"; finish REFUSED; }
  ( cd "$MANIFESTS" && sha256sum -c SHA256SUMS ) >> "$LOG" 2>&1 || {
    say "manifest digests do not match SHA256SUMS"; finish REFUSED; }
  say "manifest digests verified"
}

pilot() {  # the registered pilot with the frozen recipe (14.1 must exist before this runs)
  say "pilot"
  require_approved_commit
  require_manifests
  for f in calibration_recipe.json calibration_lambda.json; do
    [ -f "$OUT/$f" ] || { say "missing $f: the recipe is not frozen"; finish REFUSED; }
  done
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

dev() {  # the development study (section 15): schedule A, seed 123
  say "dev"
  require_approved_commit
  require_manifests
  local m="$MANIFESTS/primary_A.json" hits="$OUT/guard_hits_primary_A.json"
  [ -f "$hits" ] || "$PY" experiences.py guard-hits --data_root "$DATA" --seed 1 \
    --manifest "$m" --hard_k 5 --out "$hits"
  sha256sum "$hits" | tee -a "$LOG"
  for rounds in 4 2; do
    run_one "dev-fedavg-replay-r$rounds-A-s123" "$m" fedavg-replay 123 "$rounds" --lr 5e-5
  done
  for lam in 0.25 0.1; do
    run_one "dev-fedavg-replay-distill-lam$lam-A-s123" "$m" fedavg-replay-distill 123 8 \
      --lr 5e-5 --lambda_distill "$lam"
  done
  run_one dev-fedavg-replay-accept-A-s123 "$m" fedavg-replay-accept 123 8 --lr 5e-5 \
    --guard_hits "$hits"
}

lotte_profile() {
  say "lotte-profile"
  require_approved_commit
  require_manifests
  run_one l1-profile-r1 "$MANIFESTS/lotte_A.json" fedavg-replay 123 1 --lr 5e-5
  "$PY" - "$OUT/l1-profile-r1" <<'PYEOF'
import glob, json, sys
d = sys.argv[1]; r = json.load(open(glob.glob(d + "/continual_*.json")[0]))
print(json.dumps({"wall_seconds": int(open(d + "/wall_seconds").read()),
                  "steps": [rec["steps"] for rec in r["rounds"]]}, indent=1))
PYEOF
}

lotte() {  # the LoTTE confirmation block (section 16; 16.1 must exist before this runs)
  say "lotte"
  require_approved_commit
  require_manifests
  local settings="$OUT/l1_settings.json"
  [ -f "$settings" ] || { say "missing $settings: the settings of 15 are not fixed"; finish REFUSED; }
  local lam accept
  lam=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["lambda"])' "$settings")
  accept=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["accept"]))' "$settings")
  for sched in A B; do
    local m="$MANIFESTS/lotte_$sched.json" hits="$OUT/guard_hits_lotte_$sched.json"
    if [ "$accept" = 1 ] && [ ! -f "$hits" ]; then
      "$PY" lotte.py guard-hits --manifest "$m" --out "$hits"
    fi
    for seed in 123 2024 3407; do
      for arm in local local-replay fedavg fedavg-replay; do
        run_one "l1-$arm-$sched-s$seed" "$m" "$arm" "$seed" 8 --lr 5e-5
      done
      run_one "l1-fedavg-replay-distill-$sched-s$seed" "$m" fedavg-replay-distill "$seed" 8 \
        --lr 5e-5 --lambda_distill "$lam"
      if [ "$accept" = 1 ]; then
        run_one "l1-fedavg-replay-accept-$sched-s$seed" "$m" fedavg-replay-accept "$seed" 8 \
          --lr 5e-5 --guard_hits "$hits"
      fi
      run_one "l1-iid-fedavg-replay-$sched-s$seed" "$MANIFESTS/lotte_iid_$sched.json" \
        fedavg-replay "$seed" 8 --lr 5e-5
    done
  done
}

rar_dev() {  # rank-anchored replay development (section 17): schedule A, seed 123
  say "rar-dev"
  require_approved_commit
  require_manifests
  local m="$MANIFESTS/primary_A.json"
  run_one rar-fedavg-replay-A-s123 "$m" fedavg-replay 123 8 --lr 5e-5
  for retention in random fragile; do
    for lam in 0.5 2.0; do
      run_one "rar-anchor-lam$lam-$retention-A-s123" "$m" fedavg-replay-anchor 123 8 \
        --lr 5e-5 --lambda_anchor "$lam" --anchor_k 10 --retention "$retention"
    done
  done
  "$PY" rar_settings.py "$OUT" | tee "$OUT/rar_settings.json"
}

case "$MODE" in
  bootstrap) bootstrap ;;
  manifests) manifests ;;
  profile) profile ;;
  calibrate) calibrate ;;
  stage1) bootstrap; manifests; profile; calibrate; finish DONE ;;
  pilot) pilot; finish DONE ;;
  dev) dev; finish DONE ;;
  lotte-profile) lotte_profile; finish DONE ;;
  lotte) lotte; finish DONE ;;
  rar-dev) rar_dev; finish DONE ;;
  *) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
