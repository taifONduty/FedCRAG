# Research Status

Documentation snapshot: 16 September 2026. This is a retrospective summary,
not a new registration or permission to run experiments. It supersedes the
older README claims that every client loses against local-only training and
that two ingredients universally cause degradation. Paper files and original
registrations are unchanged.

## Completed evidence

- Size-proportional aggregation with full local epochs and two trainable LoRA
  factors leaves SciFact and ArguAna below the pretrained model at both clean
  paired seeds. Uniform improves every client over this configuration.
- Equal weighting, frozen-factor training and equal-total-work controls remove
  these losses in their measured settings. The controls change more than one
  property of training, so they do not isolate a universal cause.
- FedNova and q-FFL remove below-backbone degradation at both seeds. AFL does
  not. None consistently improves on ordinary uniform for every client;
  q-FFL can slightly exceed it on NFCorpus while losing on smaller clients.
- FedSpan improves the two smallest clients relative to frozen-factor
  uniform, but ordinary two-factor uniform has a better worst-client score
  and lower cross-client variance at both paired seeds.
- The protected shared-head pilot produced no accepted non-identity head:
  its local solver failures prevented an accepted candidate. This is not a
  proof that the entire head class is infeasible.
- The measured-response arm and its matched-split uniform references have
  completed eight rounds at seeds 123 and 2024. Their result files carry
  recorded validation markers. Those validators were not rerun for this
  documentation update.

## Measured response versus matched uniform

Final test nDCG@10, rounded to four decimals. Variance is the population
variance across the four final client scores, in squared nDCG units.

| Seed | Method | NFCorpus | FiQA | SciFact | ArguAna | Cross-client variance |
|---|---|---:|---:|---:|---:|---:|
| 123 | Matched uniform | 0.3407 | 0.2674 | 0.6617 | 0.5442 | 0.02473 |
| 123 | Measured response | 0.3362 | 0.2801 | 0.6820 | 0.5627 | 0.02685 |
| 2024 | Matched uniform | 0.3387 | 0.2619 | 0.6634 | 0.5418 | 0.02543 |
| 2024 | Measured response | 0.3373 | 0.2855 | 0.6816 | 0.5619 | 0.02620 |

The [score extract](results/a5-final-scores.csv) retains unrounded values and
each run's pretrained scores. Both arms and both references train on the
matched 90% query split at their seed. The original E1 uniform runs used the
full training split and are a distinct comparator.

The method improves the three smaller clients at both seeds, with a small
NFCorpus cost. All four clients remain above their pretrained test scores in
every measured round of both response runs. Neither halving nor a zero step
occurs. These observations do not establish monotonic improvement between
rounds, an advantage for every client over uniform, or statistical significance
from two seeds.

### Registered verdict and limitations

- The pretrained-floor criterion holds at both seeds. The worst-client
  clause holds, but the cross-client variance clause fails at both seeds,
  against both original E1 uniform and matched uniform. The arm therefore
  does not meet the complete registered requirement for the main repair.
- The magnitude-family prediction fails at both seeds, triggering its
  registered falsifier. The method is supported as measured candidate
  search, not as a validated magnitude-based explanation of its gains.
- The runs record 19 to 20 contenders per round, above the amendment's cap of
  15. Numerical validation does not establish full protocol conformance.
- Maximum absolute response-prediction error among verified candidates is
  0.0138 at seed 123 and 0.0079 at seed 2024. The former exceeds the 0.010
  fidelity threshold; acceptance uses direct measurement, not prediction.
- Both response runs and the seed-123 uniform reference used an L4. The
  seed-2024 uniform reference used a T4 under the execution note. Its frozen
  ArguAna score differs by about 0.0007. That comparison is matched on the
  query split, not on hardware.
- Saved-state single-round searches are exploratory. Their query split was
  not cleanly held out from the training that produced those states. Their
  apparent all-client advantage must not replace the full-run verdict.

The floor enforced by the method is a mean score on the measured held-out
queries. It is not a guarantee for each query, unseen queries, the test set,
previous-round scores or the best score reached earlier in training.
Repeated selection on small held-out sets also risks overfitting.

## What remains open

The next scientific question is whether one shared retriever can acquire new
retrieval ability while retaining earlier ability across sequential
experiences. Small private query memories are a proposed extension, not a
completed result. The current pretrained floor does not retain every gain
made during earlier experiences.

Any temporal evaluation needs explicit acquisition and retention references,
separate training/protection/test queries, fixed evaluation corpora for the
initial study, and comparisons with uniform training, matched replay and a
frozen model. It must report every client and earlier experience, together
with computational cost and rounds without accepted progress. Changes to
the method or experiment rules require prospective registration.

## Evidence provenance

The interpretation was checked against the completed pre-defense report's
Chapter 4 (full runs and limitations) and Chapter 5. The table was checked
directly against the four archived result JSONs in campaign
`PROGRAM7_A5_20260911`, not copied from a screenshot or an interim run.
Recorded source commits are in the CSV. Source JSON hashes are in
[the checksum list](results/a5-source-json.sha256).

The archive's raw JSONs, model states and validation evidence are not included
in this documentation extract. Hashes identify those archived files; hashes
and this score table alone are not an independently reproducible experiment
package. See the unchanged [registration](../registration/E3_PREREGISTRATION.md)
and [implementation](../response_arm.py) for the method record.
