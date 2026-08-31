#!/usr/bin/env bash
# E3 launch — executes registration/E3_PREREGISTRATION.md Part A exactly.
#
#   bash run_e3.sh verify   # print all 33 commands, run nothing
#   bash run_e3.sh probe    # run the 1-round P0 probe and evaluate the gate
#   bash run_e3.sh run      # probe+gate if needed, then the 33 runs, poweroff
#
# The P0 gate is registered: mean clone-block cosine >= 0.15 after round 1 of
# the first seed, clone-singleton mean strictly below it. A FAIL stops
# everything before the grid spends a dollar, and the measured cosines are
# the report. Every registered run is validated by the canonical validator
# the moment it finishes; any failure aborts the chain. On ANY exit the VM
# powers off (marker files record which way it ended).
set -Euo pipefail
cd "$(dirname "$0")"
PY="${E3_PYTHON:-$HOME/FedCRAG/.venv/bin/python}"
DATA="${E3_DATA_ROOT:-$HOME/FedCRAG/beir_data}"
OUT="${E3_OUT:-$HOME/E3_RESULTS_20260831}"
MODE="${1:-verify}"
mkdir -p "$OUT"
LOG="$OUT/chain.log"

say() { echo "[e3] $*" | tee -a "$LOG"; }

finish() {
  say "$1 $(date -u +%FT%TZ)"; touch "$OUT/$1.marker"; sync
  [ "$MODE" = "run" ] && sudo poweroff
  exit 0
}
trap 'finish FAILED' ERR

manifest() {
  "$PY" - "$1" <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
import e3_manifest
which = sys.argv[1]
runs = ([e3_manifest.p0_probe()] if which == "probe"
        else e3_manifest.registered_runs())
print(json.dumps(runs))
PYEOF
}

run_one() {
  local name=$1; shift
  local outdir="$OUT/$name"
  if [ -f "$outdir/.validated" ]; then say "SKIP $name (validated)"; return 0; fi
  mkdir -p "$outdir"
  say "START $name"
  "$PY" federated_forgetting.py "$@" --data_root "$DATA" --out "$outdir" \
      > "$outdir/run.log" 2>&1
  "$PY" - "$outdir" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from validate_e0 import validate_run_directory
report = validate_run_directory(sys.argv[1])
print(f"validated {report['rounds_validated']} rounds")
open(sys.argv[1] + "/.validated", "w").write("ok\n")
PYEOF
  say "DONE  $name"
}

gate() {
  "$PY" - "$OUT/p0-probe" <<'PYEOF'
import glob, json, sys
sys.path.insert(0, ".")
from e3_manifest import p0_gate
paths = glob.glob(sys.argv[1] + "/federated_*.json")
assert len(paths) == 1, f"expected one probe JSON, found {len(paths)}"
passed, report = p0_gate(json.load(open(paths[0])))
print(report)
sys.exit(0 if passed else 1)
PYEOF
}

case "$MODE" in
  verify)
    manifest registered | "$PY" -c '
import json, sys
runs = json.load(sys.stdin)
for run in runs:
    print(run["name"] + ":\n  " + " ".join(run["args"]))
print(f"\n{len(runs)} registered runs")'
    ;;
  probe|run)
    if [ ! -f "$OUT/P0_PASSED.marker" ]; then
      probe_json=$(manifest probe)
      name=$(echo "$probe_json" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)[0]["name"])')
      readarray -t args < <(echo "$probe_json" | "$PY" -c 'import json,sys;[print(a) for a in json.load(sys.stdin)[0]["args"]]')
      if [ ! -f "$OUT/$name/.validated" ]; then run_one "$name" "${args[@]}"; fi
      if gate | tee -a "$LOG"; then
        say "P0 GATE PASSED"; touch "$OUT/P0_PASSED.marker"
      else
        say "P0 GATE FAILED — shards are not a clone federation; STOPPING per registration"
        finish P0_FAILED
      fi
    fi
    [ "$MODE" = "probe" ] && { say "probe mode: stopping after gate"; exit 0; }
    runs_json=$(manifest registered)
    count=$(echo "$runs_json" | "$PY" -c 'import json,sys;print(len(json.load(sys.stdin)))')
    for idx in $(seq 0 $((count - 1))); do
      name=$(echo "$runs_json" | "$PY" -c "import json,sys;print(json.load(sys.stdin)[$idx]['name'])")
      readarray -t args < <(echo "$runs_json" | "$PY" -c "import json,sys;[print(a) for a in json.load(sys.stdin)[$idx]['args']]")
      run_one "$name" "${args[@]}"
    done
    finish DONE
    ;;
  *) echo "usage: run_e3.sh {verify|probe|run}"; exit 2;;
esac
