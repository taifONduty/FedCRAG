# Related Work for the Presentation

Updated 16 September 2026. The final column identifies questions for this
thesis's setting, not failures established in the cited papers.

## What Existing Research Addresses

| Research work | Main contribution | Question for our setting |
|---|---|---|
| **[FedE4RAG](https://arxiv.org/abs/2504.19101)** | Collaboratively trains retrievers while keeping raw data local. Uses knowledge distillation to share knowledge and homomorphic encryption to protect exchanged parameters. | Does collaboration benefit every participant, including clients with little training data? |
| **[FedEx-LoRA](https://arxiv.org/abs/2410.09432)** | Corrects the aggregation error caused by averaging LoRA factors separately. Adds a residual correction to the backbone to recover the intended average weight update. | Does exact aggregation also prevent client-level retrieval degradation? |
| **[FedMGDA+](https://arxiv.org/abs/2006.11489)** | Treats client losses as separate objectives. Combines normalized updates to seek a shared improvement direction, with convergence guarantees under stated assumptions. | Does improving training objectives also improve each client's document rankings? |
| **[FedFomo](https://arxiv.org/abs/2012.08565)** | Evaluates other clients' models on local validation data. Uses their measured usefulness to form a personalized model combination for each client. | Can validation guide one common model instead of separate personalized models? |
| **[FlowRAG](https://doi.org/10.1145/3774904.3792361)** | Adapts retrievers over time using layer-wise prompts and cross-layer fusion. Generator-guided training connects retrieval learning to answer quality. | How can one shared federated retriever retain earlier retrieval ability? |

## Aggregation Baselines

**[FedNova (Wang et al., 2020)](https://arxiv.org/abs/2007.07481).**
Normalizes client updates to correct the bias caused by unequal amounts of
local training. It addresses optimization imbalance, but does not directly
protect each client's retrieval quality.

**[Agnostic Federated Learning, AFL (Mohri et al., 2019)](https://proceedings.mlr.press/v97/mohri19a.html).**
Optimizes a shared model against the worst-case mixture of client
distributions, increasing attention to high-loss clients. Its objective
concerns training loss rather than document-ranking quality.

**[q-FFL (Li et al., 2020)](https://arxiv.org/abs/1905.10497).**
Uses a tunable parameter to give higher-loss clients greater influence,
aiming for more balanced performance. Its q-FedAvg algorithm implements this
approach, but does not enforce a pretrained retrieval-score floor for every
client.

## Details to Keep Straight When Speaking

- FedNova normalizes by effective local solver work, not necessarily by the
  Euclidean norm of an update. Dividing by the number of steps is the simple
  local-SGD case. It need not assign equal client weights.
- AFL's high-loss client is not necessarily the smallest client. Different
  client loss scales can affect which client receives most attention.
- q-FFL is the objective; q-FedAvg is an algorithm for optimizing it. The
  algorithm includes step-size normalization, so it is not just averaging
  updates with normalized loss-to-the-power-q weights. At q = 0 the objective
  reduces to the usual weighted average loss.
- FedMGDA+'s descent and convergence statements have assumptions. They are
  not unconditional guarantees that every retrieval score improves.
- FedEx-LoRA's exact residual correction changes the backbone at aggregation;
  this is different from leaving it entirely unchanged. Approximate
  compression variants should not be called algebraically exact.
- FedFomo produces client-specific model combinations. Our current experiment
  deploys one common model and checks every participating client.
- FlowRAG here means *FlowRAG: Continual Learning for Dynamic Retriever in
  Retrieval-Augmented Generation*, not a similarly named knowledge-graph
  retrieval method.

These works address related but different objectives. None of the three
aggregation baselines above imposes our explicit mean held-out retrieval
floor. That distinction alone is not a claim that the whole method is novel
or that it outperforms those methods. Keep their published contributions
separate from [our measured comparisons](research-status.md).
