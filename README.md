# FedCRAG — Federated Continual Retrieval-Augmented Generation

A research codebase for studying **catastrophic forgetting in dense retrievers under
temporally evolving / federated client knowledge**, and for building a continual
adapter-based federated retriever that prevents it without sharing raw documents.

> ## Status: implemented; E0 correctness campaign strengthened-validated
>
> **The external eleven-row E0 correctness campaign completed and all eleven rows passed
> strengthened post-hoc validation.** The audit binds the frozen launch manifest, dataset
> fingerprints, runtime/source provenance, state continuity, aggregation replay, and
> FedSpan direction certificates. This remains correctness/attribution evidence only: it
> supports no paper-scale efficacy claim, and no E1–E5 run exists. See the
> [strengthened E0 closeout](docs/2026-08-27-e0-strengthened-closeout.md).
>
> - Everything in `results/` is a **historical `bge-m3` trainable-A+B run**. `bge-m3`
>   fails this repo's own headroom gate (below), so those files are not paper evidence.
> - **Historical max-min results must not be read as validating frozen-A FedSpan.**
>   They were produced by a different coordinate (`trainable-ab`) and a different
>   aggregation rule (`rawmaxmin`, which applies simplex weights to raw states).
> - The E0 grid in `run_e0.sh` is a **correctness-attribution** grid, not a results
>   grid, and its 5 rounds do not match the paper-scale regime (see
>   [E0 correctness grid](#e0-correctness-grid)).

---

## The research question

Existing federated RAG systems assume **static** client corpora: the dense retriever is
trained once and frozen. In practice corpora evolve — new filings, guidelines, reports
keep arriving. When a retriever is updated sequentially on new corpus batches, gradient
updates for new documents overwrite the embedding geometry learned for older ones, and
retrieval on earlier corpus slices degrades. This repo:

1. **Proves the phenomenon** exists and is measurable, with confounds controlled.
2. **Separates two axes** — temporal/sequential forgetting (already shown centrally by
   FlowRAG, WWW'26) vs. the **federated-aggregation interference** that arises when
   adapters from clients with disagreeing corpora are averaged. The federated axis is the
   intended novelty.
3. **Implements a candidate fix** — FedSpan (`--weight_by normmaxmin`): **one** global
   LoRA adapter, one **shared row-orthonormal frozen** LoRA A, and conflict-aware
   norm-consistent aggregation applied to the **B deltas only**. Whether it fixes
   anything is untested (see Status).

   What FedSpan is *not*, so the description above cannot be misread: there are **no
   per-client or personalized adapters** (every client receives and returns the same
   global adapter), **no EWC and no Fisher information**, and **no replay buffer**.
   None of those appear anywhere in this codebase — `grep -riE 'ewc|fisher|personaliz'`
   over the sources returns nothing. The design constraints are C1 (only adapter
   parameters leave a silo), C2 (no replay), C3 (one global model).

### How forgetting is measured

Train on a sequence of corpus slices `T1 … TN`. After finishing slice `i`, evaluate
retrieval on **every** slice against its **own fixed index** (Framing A — the document set
never changes between measurements, so any score drop is pure embedding drift, not a
larger haystack). This produces an R-matrix `R[i][j]` = score on slice `j` using the
checkpoint after training through stage `i`. From it:

- **Forgetting** (FlowRAG Eq. 11):  `mean_i ( max_{t>=i} R[t][i] − R[N][i] )` — peak minus final.
- **BWT** (backward transfer):  `mean_i ( R[N][i] − R[i][i] )` — negative means forgetting.
- **Dissociation check**: new slice rises while old slices fall in the *same* checkpoint —
  this is what rules out optimizer instability and makes "forgetting" the only explanation.

---

## Repository layout

| File | Role |
|------|------|
| `fedcrag_common.py` | Shared module: model registries (local + API), BEIR slice loading, metric computation (nDCG/Recall/MRR via `pytrec_eval`), encoding + caching, `APIEmbedder`. |
| `benchmark_retrievers.py` | **Zero-shot** comparison of retrievers across slices (no training). Produces the "retriever comparison" table. Supports local models and API models. |
| `pilot_forgetting.py` | **Centralized sequential** training → R-matrix → Forgetting + BWT. The FlowRAG-comparable existence proof. |
| `controls.py` | Reference ceilings: **frozen** (floor), **independent** (per-slice ceiling), **joint oracle** (multi-task ceiling). The gap to sequential = headroom a method must recover. |
| `federated_forgetting.py` | **Federated simulation**: historical trainable-A+B controls, shared frozen-A comparators, and the corrected `normmaxmin` FedSpan path. |
| `aggregation_schemes.py` | Pure server-side aggregation, including PEFT-scale-aware Grams, frozen-A validation, exact FedSpan coefficients/application, fail-closed edge cases, and state hashing. |
| `README.md` / `task.tsv` | This file / the task tracker. |

`run_all_paper.sh` orchestrates the four paper runs (pilot, controls, federated
unweighted, federated weighted) with per-run logs in `logs/`.

**Planned (see `task.tsv`):** the evolving-corpus / federated-temporal environment, a
FlowRAG baseline, and the experiment campaign that would tell us whether the implemented
FedSpan aggregation helps.

---

## Installation

```bash
conda create -n fedcrag python=3.11 -y && conda activate fedcrag
pip install -r requirements.txt
```

For paper-grade reproducibility, snapshot the exact environment of the machine that
produced the results: `pip freeze > requirements.txt` (sentence-transformers/peft minor
versions change `model.fit` and adapter-naming behavior).

Notes:
- Use `pytrec_eval-terrier` (prebuilt wheel); plain `pytrec_eval` needs compilation.
- On Colab, first `pip uninstall -y torchao` (an old preinstalled version breaks PEFT's
  LoRA dispatcher), then restart the session.
- Set `WANDB_MODE=disabled` to stop `sentence-transformers` prompting for a W&B login.

### Local verification

Run the CPU-only suite before any GPU experiment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

The FedSpan coverage includes real PEFT state keys, row-orthonormal/frozen A,
unequal module scales and client norms, exact dense/effective true-step reconstruction,
finite active sets, malformed/nonfinite/zero/singleton/cancellation/solver/cap cases,
randomized scale and permutation properties, atomic persistence, collision-safe run IDs,
and a mocked end-to-end `normmaxmin` driver round.

Quick GPU + adapter sanity check before any long run (catches backbone/library issues in
seconds rather than after an overnight job):

```bash
python -c "
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict
m = SentenceTransformer('facebook/contriever')
inner = m[0].auto_model
names = {n.split('.')[-1] for n,_ in inner.named_modules()}
print('targets:', [x for x in ['query','key','value','dense'] if x in names])
m.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                         lora_dropout=0.1, target_modules=['query','key','value','dense']))
inner = m[0].auto_model
s = get_peft_model_state_dict(inner); set_peft_model_state_dict(inner, s)
print('adapter round-trip OK,', len(s), 'tensors')
"
```

---

## Usage

The three training scripts (`pilot_forgetting`, `controls`, `federated_forgetting`) share
the flags `--model`, `--slices`, `--metrics`, `--seed`, `--batch_size`, `--lora_rank`,
`--data_root`, `--out` (federated adds `--num_rounds`, `--local_epochs`, `--weighted`;
the other two use `--epochs`). `benchmark_retrievers` is separate — it takes
`--local_models`/`--api_models` instead of `--model` and does no training. Model names
resolve through the registry in `fedcrag_common.py` (e.g. `contriever`, `bge-m3`,
`bge-small`, `e5-large`) or fall through as a raw HF path.

**Backbone: use `contriever`.** `bge-m3` was the pilot backbone and it **fails the
headroom gate** — reproduce with the archived controls file in this repo:

```bash
python check_headroom.py results/controls_bge-m3_seed42.json   # exits 1
```

which reports `independent − frozen` of **−0.0124** on nfcorpus and **−0.0139** on fiqa
(ndcg@10, seed 42): fine-tuning makes those slices *worse*, so a later drop would partly
measure recipe-induced degradation rather than forgetting. `contriever` is the campaign
backbone (`run_e0.sh`, `run_w3.sh`, `GPU_SETUP.md`); `bge-m3` survives only as a
historical pilot and a possible appendix run.

### 1. Zero-shot benchmark — *which retriever is strongest?*
```bash
python benchmark_retrievers.py --local_models compare \
    --slices nfcorpus fiqa scifact arguana scidocs \
    --metrics ndcg@10 recall@10 recall@100
```
`--local_models` accepts a model-set shortcut (`train_tier`, `compare`, `sota_local`,
`leaderboard_top`) or an explicit list. **Expectation:** larger SOTA models (qwen3, e5-mistral)
clearly top `bge-m3`; pick one strong + one cheap model to carry into training experiments.

### 2. Pilot — *does forgetting happen?* (centralized sequential)
```bash
python pilot_forgetting.py --model contriever --slices nfcorpus fiqa scifact arguana --seed 42
```
**Expectation:** negative BWT / positive Forgetting on early slices, robust across seeds
42/123/2024 (each reshuffles the training order). This is the existence proof.

### 3. Controls — *how much headroom is there?*
```bash
python controls.py --model contriever --slices nfcorpus fiqa scifact arguana --seed 42
```
**Expectation:** sequential-final sits below `independent` and `joint`. The gap is what a
method can recover. No gap ⇒ no phenomenon worth fixing.

### 4. Federated — *does aggregation make it worse?* (the novel claim)
```bash
python federated_forgetting.py --model contriever --slices nfcorpus fiqa scifact arguana \
    --num_rounds 5 --seed 42            # add --weighted for weighted FedAvg
```
Flags specific to this script:
- `--weighted` enables weighted FedAvg; `--weight_by examples` (default, canonical
  FedAvg `n_k` = local training-pair count) or `--weight_by corpus` (corpus size — the
  weighting the original F4 run used; nonstandard, becomes wrong under query sharding
  where several clients share one corpus). State which one was used in the paper.
- `--lora_mode trainable-ab` preserves the historical ordinary-LoRA coordinate.
  `--lora_mode frozen-a` row-orthonormalizes one shared A while LoRA B is still zero,
  freezes A, and applies every server update only to B deltas so A remains bitwise fixed.
- `--weight_by maxmin` is retained as a filename-compatible alias for historical
  `rawmaxmin`: it solves the cosine game but applies simplex weights to raw states/deltas.
  It is an attribution control, not corrected FedSpan.
- Corrected FedSpan requires all of:

  ```bash
  python federated_forgetting.py \
      --model contriever --slices nfcorpus fiqa scifact arguana \
      --num_rounds 5 --seed 42 --weighted \
      --lora_rank 16 --lora_mode frozen-a \
      --frozen_a_row_scale peft-init --weight_by normmaxmin \
      --fedspan_step_policy median-active \
      --fedspan_direction_policy minnorm \
      --fedspan_active_abs_tol 1e-12 \
      --fedspan_active_rel_tol 1e-8 \
      --fedspan_mixture_norm_tol 1e-6 \
      --save_states
  ```

  Both policies are always explicit; there is no implicit default, and omitting either
  is a CLI error before any data work.

  `--fedspan_step_policy` fixes the *magnitude*. `median-active` sets each round's true
  effective-B norm to the median finite active-client update norm after activity gating.
  The alternative `fixed` policy requires a positive finite `--fedspan_step_norm`;
  `median-active` rejects that constant.

  `--fedspan_direction_policy` fixes the *direction*, and the two choices do not optimize
  the same thing:

  - `minnorm` (the campaign default) maximizes the worst-case cosine of the direction
    **actually applied** — the *normalized* mixture. It is solved as the min-norm point
    of the convex hull of the unit client directions.
    **Disclosure: this primitive is not novel.** It is equivalent to **FedMGDA+**
    ([arXiv:2006.11489](https://arxiv.org/abs/2006.11489), Hu, Shaloudegi, Zhang & Yu,
    *Federated Learning Meets Multi-objective Optimization*) at `epsilon = 1`, here
    applied to the cosine Gram of normalized client updates. Any novelty claim must rest
    on the surrounding construction (shared frozen row-orthonormal A, raw-B delta
    coefficients, true-effective-step normalization), never on this solver.
  - `maxmin-lp` is the historical LP. It maximizes `min_i (Cw)_i` over the simplex, but
    the applied direction is normalized, so the LP's objective is **not** the applied
    quantity. It is retained only as a recorded ablation.

  Whichever is selected, **both** values are computed and logged every round, so the gap
  is measured rather than assumed: `achieved_min_direction_cosine`, `min_norm_value`,
  `direction_solver_shortfall`, and `min_norm_solver{gap,iterations,converged,tol}` land
  in `fedspan_diagnostics`, alongside `direction_policy` and `direction_policy_specified`.

  `--frozen_a_row_scale` (frozen-A only) sets the row constant `c` in `A Aᵀ = c² I`.
  `unit` fixes `c = 1`; `peft-init` rescales the orthonormal rows to the module's own
  measured pre-orthogonalization row RMS, matching PEFT's initialization scale. These
  are **separate explicit choices**, not interchangeable settings, and there is no safe
  implicit row-scale default. Every frozen-A invocation must declare one because
  switching rescales every client's effective step. The resolved value, its
  explicitness, and the per-module record are written to `method_contract`, folded into
  the configuration hash, and tagged into the output filename so the two arms cannot
  overwrite each other.

  `--save_states` is mandatory. The command refuses
  unknown or dirty Git provenance unless `--allow_dirty_provenance` is supplied for a
  development-only run. That override
  is recorded and must not count as paper-grade E0--E5 evidence.
- `normmaxmin` forms the game from concatenated `alpha/r`-scaled raw-B deltas, excludes
  nonfinite and zero/tiny clients, handles empty and singleton active sets, and fails
  closed to a zero update on solver failure, near cancellation, reconstruction failure,
  or a declared coefficient-cap violation. It records the full-precision simplex and
  raw-delta coefficients, active set, Gram, solver residuals, actual direction/norm, and
  broadcast/client/applied-state hashes in `fedspan_diagnostics`.
- `--save_states` saves the exact broadcast, per-client, and applied-global adapter
  states plus their hashes each round
  (`states_<model>_seed<seed>_<arm>_r<rounds>_round<N>.pt`, a few MB each
  at LoRA r=16) — required
  for any mechanism diagnostics (principal angles between client updates,
  `‖avg(B)avg(A) − avg(BA)‖`). Turn it on for every scaled run you may analyze later.
- Output goes to `federated_<model>_seed<seed>_<arm>_r<rounds>.json`; frozen-A and
  `normmaxmin` filenames include the coordinate, step policy (and the declared constant
  for fixed mode), and a 12-character
  hash of every result-affecting frozen-A option so incompatible arms cannot overwrite
  each other. The complete hash is recorded in the result JSON.
  (the round count in the name keeps runs with different `--num_rounds` from
  overwriting each other), is rewritten
  atomically **after every round** (a crash loses at most the current round), and the log
  prints **all** metrics per round, so trajectories are always recoverable.
**Expectation (decides the paper's framing):**
- federated-final **below** centralized-final ⇒ aggregation compounds forgetting → thesis holds.
- **about equal** ⇒ reframe around "forgetting persists even under federated averaging".
- **above** (positive transfer) ⇒ reframe around "when federation helps vs. hurts".
Run this before committing the narrative.

<a id="e0-correctness-grid"></a>
### E0 correctness grid

The frozen eleven-row E0 grid is exposed separately:

```bash
bash run_e0.sh manifest  # print the exact eleven commands; no tests or training
bash run_e0.sh verify    # require clean provenance and run CPU gates; no training
bash run_e0.sh run       # execute only after cloud spend is explicitly authorized
bash run_e0.sh resume    # continue an interrupted campaign against its frozen manifest
```

`run_e0.sh` never provisions cloud resources. Its output root defaults outside the Git
worktree so completed rows cannot dirty provenance for later rows. `run` freezes a
`manifest.json` and refuses to start on top of an existing one; `resume` refuses to
continue if the manifest's commit or rows no longer match the tree.

For the completed legacy campaign, legacy E0 per-round timings are unavailable because
buffered output did not preserve trustworthy round boundaries. The total row runtime
remains usable from the row-level launcher timestamps. Future launcher artifacts use the
strengthened timing evidence contract; this does not retroactively reconstruct legacy
round timings.

The strengthened closeout validated all eleven rows and published the canonical 3.3 GiB
package locally under `post_e0_audit/2026-08-25/`. GCP stockouts prevented the original
`g2-standard-8` and the first six Singapore/Taiwan zone attempts, so preservation used a
fresh snapshot and a suitable `g2-standard-4` VM with the same single 24 GiB NVIDIA L4 in
`asia-northeast1-c`. That host-shape change affects only post-hoc validation and artifact
transfer; it did not rerun or alter E0. Both the original and clone were independently
observed `TERMINATED` after publication. Full identities, digest, and limitations are in
the [closeout record](docs/2026-08-27-e0-strengthened-closeout.md).

**E0 numbers are not comparable to the paper's cells. Do not quote them as results.**

- **E0 runs 5 rounds** (`ROUNDS=5` in `run_e0.sh`). The repo's paper-scale federated
  matrix runs **15** (`ROUNDS=${ROUNDS:-15}` in `run_w3.sh`, with the same 500-step cap).
  Drift, BWT, and forgetting all accumulate with the round count, so an E0 drift number
  and a paper drift number are different quantities, not a small and a large sample of
  one quantity. E0 exists to attribute *correctness* — does each arm do what its contract
  says — not to measure retrieval outcomes.
- **The `capped-500` step cap binds on exactly one client.** At the campaign
  `--batch_size 32`, the per-round step counts of the four E0 slices are approximately:

  | slice | train pairs | steps/round @ bs 32 | capped at 500? |
  |---|---|---|---|
  | nfcorpus | ~110,600 | ~3,455 | **yes** (~3,455 → 500) |
  | fiqa | ~14,160 | 442 | no |
  | scifact | ~912 | 28 | no |
  | arguana | ~696 | 21 | no |

  So `capped-500` vs `full` is a **single-client** intervention: it changes only how much
  nfcorpus trains, and leaves the other three clients bit-identical in workload. Read the
  two regimes as "nfcorpus dominance on/off", not as a global compute knob.

  Provenance of those figures: derived from the archived `logs/f42.log`
  (`train_samples_per_second × train_runtime` per client, floor-divided by 32, matching
  `NoDuplicatesDataLoader.__len__`). The nfcorpus figure is the least precise of the four
  because the log rates are rounded — but it is an order of magnitude above the cap, so
  its exact value does not affect which client binds. The external E0 correctness
  campaign has completed, but these figures remain estimates from the cited `f42` log,
  not measurements read from E0.

### API retrievers (OpenRouter / OpenAI / Cohere / Voyage)
Benchmark-only — no weight access, so API models **cannot** be LoRA-trained and never enter
the pilot/controls/federated scripts.
```bash
export OPENROUTER_API_KEY=sk-or-...
python benchmark_retrievers.py --api_models gemini-001 \
    --slices nfcorpus scifact --metrics ndcg@10 recall@10
```
Embeddings are cached to `--out/emb_cache/` after the first run; embedding a large corpus
(e.g. FiQA's 57k docs) through an API is a real charge. `openai-3-large`/`cohere-v4`/
`voyage-3.5` route to their own endpoints and need their own keys, not OpenRouter.

---

## Models & datasets

**Local registry** (`LOCAL_MODELS` in `fedcrag_common.py`) — 25 retrievers spanning
small/base (≤~600M: bge-small/base/large/m3, e5-base/large, me5-large, gte-large,
gte-mbase, mxbai-large, nomic-v1.5, snowflake-l, embeddinggemma, jina-v3, qwen3-0.6b),
mid (~1.5–4B: stella-1.5b, gte-qwen2-1.5b, qwen3-4b), and large (7–8B: e5-mistral,
gte-qwen2-7b, qwen3-8b, nv-embed-v2, sfr-2r, linq-mistral, gritlm-7b). The `MODEL_SETS`
shortcuts `train_tier`, `compare`, `sota_local`, `leaderboard_top` group common subsets.
Each model carries its **instruction prefix** — mandatory; a wrong/missing prefix silently
halves a model's scores.

**Slices** are BEIR subsets used as proxy clients / time-slices (nfcorpus, fiqa, scifact,
arguana, scidocs, trec-covid, …). For the existence proof use disjoint domains (clean, large
effect); for realism, the planned temporal setup uses within-domain snapshots (e.g. EDGAR
10-Ks partitioned by filing year, and by SIC code for federated clients).

---

## Result so far (Colab pilot, `bge-small`, 2 slices, 3 seeds)

| Seed | Order | BWT | Reading |
|------|-------|-----|---------|
| 42   | scifact→nfcorpus | **−0.0424** | SciFact forgotten after NFCorpus |
| 123  | scifact→nfcorpus | **−0.0373** | SciFact forgotten after NFCorpus |
| 2024 | nfcorpus→scifact | +0.0015 | near-zero (SciFact too small to overwrite NFCorpus) |

Forgetting is **asymmetric** and scales with the *subsequent* slice's training volume
(at `--batch_size 32` NFCorpus drives ~3,455 steps vs. SciFact's 28). This is the
motivation for **conflict-aware aggregation**: in federation, large-corpus clients
dominate the averaged adapter and can overwrite small-corpus clients' representations.
It is a motivation, not evidence that FedSpan addresses it.

This is a **pilot on `bge-small` over 2 slices** — it is not a paper cell, and no run in
this repository is. The paper numbers are to be produced by the **`contriever`** campaign
(`run_w3.sh`, 4 slices, 3 seeds, R=15), which **has not been run**. `bge-m3` is not the
paper backbone: it fails the headroom gate on nfcorpus and fiqa (see
[Usage](#usage) — reproduce with `check_headroom.py`).

---

## Known limitations / caveats

- **LoRA targets are BERT/XLM-R only.** Training scripts use
  `target_modules=["query","key","value","dense"]` (note: `"dense"` also matches the FFN
  and pooler layers, so the LoRA is attention+FFN — state this in the paper), which fits
  bge-*, e5, gte-large, etc. **Qwen-family** retrievers (qwen3, gte-qwen2, e5-mistral,
  nv-embed) use `q_proj/k_proj/v_proj/o_proj` — they work for zero-shot benchmarking, and
  the training scripts now **fail fast with a clear error** (`check_lora_targets`) instead
  of silently attaching LoRA to nothing. Auto-detection tracked in `task.tsv`.
- **Slices without a BEIR train split (e.g. ArguAna) fall back** to a deterministic halving
  of their test queries (sorted qids) into train/eval. The fallback now logs a warning and
  is recorded as `"split_fallback"` in every result JSON — mention it in the paper's setup,
  since ArguAna ships no train split.
- **Federated BWT is anchored at round 1** (`final − round_1`), because in federation every
  client trains every round so there is no single "moment slice i was learned". Compare
  *final scores across regimes*, not raw BWT deltas, when claiming aggregation interference.
- **Optimizer state resets every round.** `model.fit` re-creates the optimizer per call, so
  Adam moments reset at each federated round (standard in FedAvg) and at each sequential
  stage in the pilot. Say so in the paper.
- **Determinism is partial.** Seeds pin python/numpy/torch (incl. CUDA RNG), but AMP and
  cudnn nondeterminism are not pinned — acceptable, but state it in the repro statement.
- **Verify prefixes & API model strings** against each model's HF card / the OpenRouter
  catalog before trusting absolute numbers — they drift between versions.

Fixed (previously listed here): `mrr@k` now respects its cutoff (the run is truncated to
`k` before `recip_rank`, so a gold doc beyond rank `k` contributes 0), and a
`requirements.txt` + `run_all_paper.sh` orchestration script exist.

---

## Positioning vs. prior work

**FlowRAG (WWW'26)** established centralized sequential forgetting in dense retrievers, so
this repo *cites and builds on it* rather than claiming the phenomenon first. Its metric
(`Forget = max_t − final`) is matched exactly in `pilot_forgetting.py` for direct
comparability. FlowRAG is **entirely centralized** (no clients, no aggregation, no privacy
constraint), and its generator-guided loss assumes centralized LLM access over the raw
documents a federated setting forbids — which is precisely the gap this work targets.

## License

See [`LICENSE`](./LICENSE).
