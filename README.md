# FedCRAG

Code and experiment records for my undergraduate thesis on federated
fine-tuning of dense retrievers. Four organisations ("silos") with very
different amounts of private data jointly train one retrieval model with LoRA
adapters, and the project studies what the server's aggregation rule does to
each of them.

## What we found

Three results, each with the seeds and caveats spelled out in the paper draft
under `paper_draft/`:

1. **The default weighting hurts everyone.** FedAvg's standard rule weights
   each client by its data size. In our four-silo BEIR testbed (size ratio
   157:20:1.3:1), that rule leaves the small clients *below the untrained
   backbone* after training (SciFact -0.074 +/- 0.004 nDCG@10 across three
   seeds), and a budget-matched control shows every measured silo ends worse
   than if it had trained alone. Plain uniform weighting beats size weighting
   for every client, including the largest one.
2. **The damage needs two ingredients.** A pre-registered experiment reran the
   same skewed weighting with a shared frozen LoRA A-factor, which makes
   aggregation exact. The harm disappeared (two of three seeds complete, both
   clean; the third can no longer change the outcome). So weight skew alone
   redistributes, and weight skew combined with standard two-factor LoRA
   training is what destroys. Removing either ingredient removes the absolute
   harm. This overturned our own earlier explanation, and the repo keeps the
   record of both.
3. **A geometry-aware aggregation rule (FedSpan) redistributes.** The server
   solves a small min-norm problem on the cosine Gram of client updates and
   protects the worst-aligned client. Within its own coordinate it reliably
   lifts the smallest silos at little cost to the large ones. It is not a
   mean-score improvement, and the paper says so.

## Layout

- `federated_forgetting.py` - the training and evaluation driver
- `aggregation_schemes.py` - weighting rules: uniform, n_k, q-FedAvg, FedMGDA+,
  FedSpan (exact min-norm solver with a per-round optimality certificate)
- `e3_shard.py`, `e3_manifest.py`, `run_e3.sh` - the clone-federation
  experiment, generated from a manifest and gated on a round-1 geometry check
- `validate_e0.py` - independent validator that recomputes every round's
  aggregate from the persisted client states and refuses mismatches
- `tests/` - 408 tests, including mutation and tamper tests
- `registration/` - the signed pre-registration for the experiment program,
  with predictions and decision rules committed before the data existed
- `paper_draft/` - LaTeX source of the paper in progress

## Running

```
pip install -r requirements.txt
python -m pytest tests/          # no GPU needed
bash run_e3.sh verify            # prints the registered runs without executing
```

Training runs need a GPU and the BEIR corpora; `GCP_RUNBOOK.md` documents the
exact setup we used.

## A note on the history

The branch history here is deliberately kept intact, including mistakes and
their corrections, because the pre-registration in `registration/` is only
worth something if the commits it names still exist. Development branches
(`fedspan-e0-*`, `w3-campaign`, `results/*`) hold the full audit trail;
`main` carries the current state.

## License

MIT, see `LICENSE`.
