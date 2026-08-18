# GPU machine setup — W3′ campaign (Aug 2026)

Target: any Linux box with an NVIDIA GPU (16 GB is enough for Contriever-110M;
24 GB gives slack for the bge-m3 appendix runs later). Total campaign compute:
roughly 25–35 GPU-hours — calibrate with the smoke run and the first R-S run.

## 1. Bootstrap

```bash
git clone <repo-url> FedCRAG && cd FedCRAG        # or rsync the folder
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x True <your GPU>   — CUDA MUST be True (amp + speed depend on it)
```

## 2. Verify before burning GPU-hours (non-negotiable)

```bash
.venv/bin/python -m pytest tests/ -q      # metric semantics: must be all green
bash run_w3.sh smoke                      # 1-round scifact run, ~minutes
```

The smoke run must end with `saved results_smoke/federated_contriever_seed42_unweighted_r1.json`
and that JSON must contain a real `commit` hash and `"use_amp": true`.

## 3. The campaign (order matters)

```bash
bash run_w3.sh controls     # headroom gate: contriever vs contriever-msmarco
.venv/bin/python check_headroom.py results/controls_contriever_seed42.json
.venv/bin/python check_headroom.py results/controls_contriever-msmarco_seed42.json
```

**GATE:** the primary backbone is whichever checkpoint PASSes (independent >
frozen on all four slices). If both pass, prefer `contriever` (weaker zero-shot
= more adaptation headroom = cleaner forgetting signal). If ArguAna alone fails
on both → ping the supervisor session: slice swap decision (D2) activates.

```bash
PRIMARY=contriever bash run_w3.sh rt     # sequential forgetting, 3 seeds
PRIMARY=contriever bash run_w3.sh rs     # federated matrix: 3 seeds x 3 weightings, R=15, states saved
```

Every finished run auto-appends a row to `runs.tsv` (status, BWT, file paths).
Disk: `rs` saves adapter states every round — budget ~10 GB free.

## 4. Ship results back

Copy (or commit on a results branch, as in May) — but this time include code:

```
results/*.json  logs/*.log  runs.tsv  results/states_*  + `git rev-parse HEAD`
```

Rule from the audit: results are only paper-grade if the exact commit that
produced them is recorded (now automatic in each JSON's `commit` field) and
`pip freeze > requirements.lock` from the run machine is committed alongside.

## Notes

- `--batch_size 32` is the campaign default (May pilot used 8 for bge-m3
  memory; Contriever-110M has no such constraint). Keep it uniform across arms.
- AMP now auto-disables off-CUDA (`use_amp` recorded in every JSON), so runs
  are technically possible on CPU/MPS for debugging — never for paper numbers.
- If a run dies mid-way: federated JSONs are dumped atomically after every
  round, so partial trajectories are valid; re-run the arm from scratch anyway
  (optimizer/order state is not checkpointed).
