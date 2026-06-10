# FedCRAG — Federated Continual Retrieval-Augmented Generation

A research codebase for studying **catastrophic forgetting in dense retrievers under
temporally evolving / federated client knowledge**, and for building a continual
adapter-based federated retriever that prevents it without sharing raw documents.

> Status: **phenomenon-validation stage.** The pilot result (below) confirms forgetting
> exists; the cluster experiments that turn it into paper-grade evidence, the temporal /
> federated-temporal environment, and the FedCRAG method itself are not built yet. See
> [`task.tsv`](./task.tsv) for the full execution plan and current status of every step.

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
3. (Planned) **Fixes it** with the FedCRAG method: per-client LoRA adapters + EWC
   consolidation + forgetting-severity-aware aggregation.

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
| `federated_forgetting.py` | **Federated simulation**: N clients, local LoRA training, FedAvg aggregation over rounds. `--weighted` toggles corpus-size-weighted averaging. The novel-axis experiment. |
| `README.md` / `task.tsv` | This file / the task tracker. |

`run_all_paper.sh` orchestrates the four paper runs (pilot, controls, federated
unweighted, federated weighted) with per-run logs in `logs/`.

**Planned (see `task.tsv`):** the evolving-corpus / federated-temporal environment, and
the FedCRAG method plus a FlowRAG baseline.

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

Quick GPU + adapter sanity check before any long run (catches backbone/library issues in
seconds rather than after an overnight job):

```bash
python -c "
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict
m = SentenceTransformer('BAAI/bge-m3')
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
resolve through the registry in `fedcrag_common.py` (e.g. `bge-m3`, `bge-small`,
`e5-large`) or fall through as a raw HF path.

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
python pilot_forgetting.py --model bge-m3 --slices nfcorpus fiqa scifact arguana --seed 42
```
**Expectation:** negative BWT / positive Forgetting on early slices, robust across seeds
42/123/2024 (each reshuffles the training order). This is the existence proof.

### 3. Controls — *how much headroom is there?*
```bash
python controls.py --model bge-m3 --slices nfcorpus fiqa scifact arguana --seed 42
```
**Expectation:** sequential-final sits below `independent` and `joint`. The gap is what a
method can recover. No gap ⇒ no phenomenon worth fixing.

### 4. Federated — *does aggregation make it worse?* (the novel claim)
```bash
python federated_forgetting.py --model bge-m3 --slices nfcorpus fiqa scifact arguana \
    --num_rounds 5 --seed 42            # add --weighted for weighted FedAvg
```
Flags specific to this script:
- `--weighted` enables weighted FedAvg; `--weight_by examples` (default, canonical
  FedAvg `n_k` = local training-pair count) or `--weight_by corpus` (corpus size — the
  weighting the original F4 run used; nonstandard, becomes wrong under query sharding
  where several clients share one corpus). State which one was used in the paper.
- `--save_states` saves per-client + global adapter states each round
  (`states_<model>_seed<seed>_<tag>_round<N>.pt`, a few MB each at LoRA r=16) — required
  for any mechanism diagnostics (principal angles between client updates,
  `‖avg(B)avg(A) − avg(BA)‖`). Turn it on for every scaled run you may analyze later.
- Output goes to `federated_<model>_seed<seed>_<weighted|unweighted>.json`, is rewritten
  atomically **after every round** (a crash loses at most the current round), and the log
  prints **all** metrics per round, so trajectories are always recoverable.
**Expectation (decides the paper's framing):**
- federated-final **below** centralized-final ⇒ aggregation compounds forgetting → thesis holds.
- **about equal** ⇒ reframe around "forgetting persists even under federated averaging".
- **above** (positive transfer) ⇒ reframe around "when federation helps vs. hurts".
Run this before committing the narrative.

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
(NFCorpus drove ~3,455 steps vs. SciFact's 28). This directly motivates
**forgetting-severity-aware aggregation**: in federation, large-corpus clients dominate the
averaged adapter and overwrite small-corpus clients' representations. The cluster runs with
`bge-m3` + 4 balanced slices produce the actual paper numbers.

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
