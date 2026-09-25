# FedCRAG

Code and experiment records for my undergraduate thesis on federated
fine-tuning of dense retrievers. Four organisations ("silos") with very
different amounts of private data jointly train one retrieval model with LoRA
adapters, and the project studies what the server's aggregation rule does to
each of them.

## Current findings

Documentation snapshot: 16 September 2026. The
[research status](docs/research-status.md) records the completed comparisons,
result sources and limitations. These findings concern one four-client BEIR
proxy federation with a Contriever backbone, not all federated retrievers.

1. **Average improvement can hide client-level degradation.** In two clean
   paired eight-round runs, size-proportional aggregation with full local
   epochs and two trainable LoRA factors leaves the two smallest clients
   below their pretrained scores. Uniform weighting improves all four clients
   over that configuration and keeps them above the pretrained model. The
   registered local-only comparison supports a participation disadvantage
   for three clients, not all four.
2. **The configuration matters.** Equal weighting, a shared frozen factor and
   equal-total-work controls each remove the below-backbone losses in their
   measured settings. These interventions do not establish three universally
   necessary or sufficient causes: freezing a factor also changes the
   trainable space, and equalising work changes data exposure and warmup.
   Saved-state measurements show that unequal update magnitudes leave the
   largest client influential even under equal weights.
3. **Uniform remains a strong baseline.** FedNova and q-FFL remove the losses
   at both seeds; AFL does not. None improves on uniform for every client.
   FedSpan helps the smallest clients relative to frozen-factor uniform, but
   loses to ordinary two-factor uniform on worst-client score and
   cross-client variance.
4. **Measured-response aggregation produces a tradeoff.** Both full runs
   improve the three smaller clients over uniform on matched training splits,
   but NFCorpus finishes slightly lower and cross-client variance increases.
   All clients remain above their pretrained test scores in every measured
   round. This does not meet the complete registered success rule. Numerical
   validation also does not erase the recorded candidate-count protocol
   discrepancy.

Continual retrieval has one pilot (registration section 14.2): on a
constructed MS-Shift stream of five clients and four experiences, shared
training acquires each new experience while per-query regression on earlier
ones remains above the registered threshold under bounded replay, and the
tested distillation arm reduces that regression at a larger acquisition cost
than its criterion allowed. The pilot now informs method design, so it is
development evidence rather than confirmation. The current evidence concerns
retrieval quality, not generated-answer quality or formal privacy protection.

## Presentation notes

- [Six-step method, equations, notation and an example](docs/presentation-method.md)
- [Related work and aggregation baselines](docs/presentation-literature.md)
- [Two-seed results and claim boundaries](docs/research-status.md)

These notes document the current solution. They do not amend experiment
registrations or replace the paper draft.

## Layout

- `federated_forgetting.py` - the training and evaluation driver
- `aggregation_schemes.py` - weighting rules: uniform, n_k, q-FedAvg, FedMGDA+,
  FedSpan (exact min-norm solver with a per-round optimality certificate)
- `response_arm.py`, `response_aggregation.py` - measured-response candidate
  construction, prediction, selection and exact held-out verification
- `e3_shard.py`, `e3_manifest.py`, `run_e3.sh` - the clone-federation
  experiment, generated from a manifest and gated on a round-1 geometry check
- `validate_e0.py` - independent validator that recomputes every round's
  aggregate from the persisted client states and refuses mismatches
- `experiences.py`, `memory.py`, `regression.py`, `continual_driver.py`,
  `validate_continual.py`, `run_continual.sh` - the continual pipeline: a
  constructed semantic-shift stream over MS MARCO and MS-Shift, the retained-query
  budget, the retention measures, the experience loop with its baseline arms, its
  validator and the GPU-machine chain (design in `docs/continual-pipeline-design.md`)
- `lotte.py` - LoTTE as a continual stream for the confirmation block: five topics as
  clients, each topic's dev and test forums as its two experiences
- `acceptance.py` - arm F of the development study (registration section 15): a
  reference-based acceptance check on the server's update
- `l1_settings.py` - the rules of registration section 15 that fix the LoTTE block's
  distillation weight and whether the acceptance arm enters
- `t1_gate.py`, `t1_report.py` - the pilot's registered gate, and its report
  recomputed from the per-query records without the production metric code
- `l1_report.py` - the LoTTE block's hypotheses H1 to H6, from the same recomputation
- `anchors.py`, `rar_settings.py` - rank-anchored replay (registration section 17): each
  retained query keeps its acquisition reference's top-10 order, and the rule that picks
  its development configuration
- `seed_churn.py` - per-query differences between two seeds at the same stage, on the
  scale of the regression it is compared with
- `tests/` - unit, integration, mutation and tamper tests
- `registration/` - the signed pre-registration for the experiment program,
  with predictions and decision rules committed before the data existed
- `paper_draft/` - historical LaTeX draft (August 2026), superseded by
  `docs/research-status.md`; kept for the record
- `docs/superpowers/` - historical planning notes for the FedSpan step policy
  (August 2026)

## Running

```
pip install -r requirements.txt
python -m pytest tests/          # no GPU needed
bash run_e3.sh verify            # prints the registered runs without executing
```

Training runs need a GPU and the BEIR corpora; `GCP_RUNBOOK.md` documents the
exact setup we used.

`--weighted --weight_by response-maxmin` selects aggregation by measured
response. Each client measures embedding responses to the current model and
each single-client update. In the completed v2 runs, the server ranks compact
contenders, including response-selected mixtures, by the worst client's
predicted mean retrieval gain. Up to two leaders are evaluated directly, and
the best verified floor-feasible candidate is applied. The registered rule
allows two step halvings and otherwise retains the current model.
`validate_e0.py` recomputes the decision from the record. See the
[method notes](docs/presentation-method.md) and
[registration](registration/E3_PREREGISTRATION.md) for the full configuration;
the two flags alone do not specify it.

## A note on the history

The branch history here is deliberately kept intact, including mistakes and
their corrections, because the pre-registration in `registration/` is only
worth something if the commits it names still exist. Development branches
(`fedspan-e0-*`, `w3-campaign`, `results/*`) hold the full audit trail;
`main` carries the current state.

## License

MIT, see `LICENSE`.
