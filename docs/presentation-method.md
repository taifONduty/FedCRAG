# Aggregation by Measured Response

Presentation notes, updated 16 September 2026. This describes the implemented
mean-selection configuration used in the completed two-seed runs, checked
against [response_arm.py](../response_arm.py) and
[response_aggregation.py](../response_aggregation.py) at commit `c456d03`.
It does not describe a best-so-far floor or a temporal extension.

## The Idea

Clients first train locally from the same shared retriever. Before combining
their updates, each client measures what those updates do to its own query
and document embeddings. These responses help rank candidate combinations.
The leading combinations are then evaluated directly. The server accepts a
candidate only if every client's measured held-out score meets its fixed
pretrained baseline.

The response model saves repeated encoder work during candidate search. It
does not eliminate the cost of corpus encoding, ranking or exact checks.

## Notation

| Symbol | Meaning |
|---|---|
| $t$ | Current communication round |
| $K$ | Number of participating clients; four in the current experiments |
| $j$ | Client that proposes an update |
| $k$ | Client that evaluates a model on its own data |
| $\theta^t$ | Shared LoRA factor parameters at the start of round $t$ |
| $\theta^0$ | Initial adapter state representing the pretrained retriever |
| $\Delta_j^t$ | Client $j$'s change to the adapter parameters |
| $v_j$ | Coefficient applied to client $j$'s update |
| $f(x;\theta)$ | Retriever embedding of a query or document $x$ |
| $N(z)$ | Unit-length normalization, $z/\lVert z\rVert_2$ for nonzero $z$ |
| $Q_k$ | Client $k$'s held-out selection queries, excluded from local training |
| $\mathcal C_k$ | Client $k$'s full evaluation corpus |
| $M_k(\theta)$ | Mean held-out nDCG@10 under the actual model |
| $\widehat M_k^t(v)$ | Mean held-out score predicted from embedding responses |
| $F_k$ | Fixed floor, $M_k(\theta^0)$ |

Here $\theta$ denotes the trainable LoRA **factors**, not their effective
matrix product. The pretrained backbone is frozen. Coefficients are
nonnegative but need not sum to one: their total can also control step size.
Test queries are not used to select candidates.

## Slide 1: Local Training and Response Estimation

Arrange steps 1 to 3 as three rows. Put a short label on the left and its
equation on the right. Give step 3 more room because it has two equations.

### 1. Broadcast the Shared Model

Every client starts from the same adapter state:

```latex
\theta_j^{t,\mathrm{start}} = \theta^t,
\qquad j = 1,\ldots,K.
```

Spoken explanation: "We send the same model to every client. The pretrained
backbone stays fixed; clients train only the small LoRA adapters."

### 2. Train Locally and Return Updates

```latex
\Delta_j^t = \theta_j^{t,\mathrm{local}} - \theta^t.
```

The local model results from the configured local training procedure, not
necessarily one gradient step. Clients return their parameter updates; raw
training queries and documents stay local.

### 3. Estimate Retrieval Responses

Every client evaluates the current model and each of the $K$ solo updates
on its own queries and corpus. For a local query or document $x$:

```latex
u_{kj}^t(x) = f(x;\theta^t + \Delta_j^t) - f(x;\theta^t).
```

The predicted embedding for a combination is:

```latex
\widehat f_k^t(x;v) = N\!\left(
f(x;\theta^t) + \sum_{j=1}^{K} v_j u_{kj}^t(x)
\right).
```

Spoken explanation: "Each client checks every proposed update, not just its
own. We measure how its embeddings move, then use those movements to predict
what combinations might do."

This is an embedding-response approximation, not a linear average of nDCG
scores and not an exact gradient model. Query and document embeddings are
predicted first; their similarities produce rankings and predicted scores.
It agrees with the measured broadcast and solo responses in exact arithmetic,
but mixtures can have prediction error. The implementation guards
normalization with a small positive denominator floor.

## Slide 2: Candidate Selection and Verification

Start with a brief bridge: "Step 3 supplies predicted client retrieval
scores." Then show steps 4 to 6. Keep the full update rule in backup notes if
the slide becomes crowded.

### 4. Rank Candidate Combinations

A candidate model is constructed in adapter-parameter space:

```latex
\theta^t(v) = \theta^t + \sum_{j=1}^{K} v_j \Delta_j^t.
```

The completed runs rank candidates by their worst client's predicted mean
gain relative to the **current** model:

```latex
R_t(v) = \min_k\left[
\widehat M_k^t(v) - M_k(\theta^t)
\right].
```

Only candidates predicted to meet every client's floor are shortlisted.
Select up to two with the largest $R_t(v)$; ties use mean gain, then candidate
name. The pool includes familiar aggregations, solo updates, magnitude-based
families and response-selected combinations. This is a finite search, not a
global optimization over all possible models. The implemented "compact"
mode still evaluates a larger predicted pool to obtain some contenders;
see the [recorded candidate-count limitation](research-status.md).

### 5. Evaluate the Leading Candidates Directly

Clients encode their queries and full corpus using the actual candidate
adapter, then compute:

```latex
M_k\!\left(\theta^t(v)\right) =
\frac{1}{|Q_k|}\sum_{q\in Q_k}
\operatorname{nDCG@10}\!\left(q;\theta^t(v),\mathcal C_k\right).
```

Spoken explanation: "The prediction gives us a shortlist. We then run the
actual candidate models and measure their rankings before accepting one."

"Exact" means direct evaluation of these models on the specified finite
query sets and corpus. It does not mean exact optimization or a guarantee
about unseen queries.

### 6. Accept a Verified Update

Every client must meet its fixed **pretrained** floor:

```latex
M_k\!\left(\theta^t(v)\right) \ge M_k(\theta^0),
\qquad \forall k.
```

Among the verified candidates that pass, choose the one with the largest
measured worst-client gain, using the same tie rules as selection.
If neither passes, halve the highest-ranked predicted candidate's
coefficients and check again, at most twice ($v/2$, then $v/4$).
Stop at the first passing halved candidate. If no candidate is shortlisted
or no checked candidate passes, keep the current model.

```latex
\theta^{t+1} =
\begin{cases}
\theta^t + \displaystyle\sum_{j=1}^{K} v_j^*\Delta_j^t,
& \text{if a verified candidate passes},\\
\theta^t, & \text{otherwise}.
\end{cases}
```

Here $v^*$ includes any accepted halving. The completed runs used the mean
selection rule, a frozen floor with zero tolerance, two verified candidates
and at most two halvings. Pessimistic and Pareto options exist in the code;
they are not the selection rule used in these reported full runs.

## A Small Example

**Illustrative numbers only, not experimental results.** Suppose the four
clients' current held-out scores are $(0.32,0.27,0.62,0.50)$, and their
pretrained floors are $(0.25,0.20,0.50,0.40)$.

| Candidate | Predicted gains for clients 1 to 4 | Worst predicted gain |
|---|---|---:|
| Uniform | $(0.005,0.010,0.020,0.010)$ | 0.005 |
| A three-client subset | $(0.010,0.020,0.015,0.020)$ | 0.010 |
| Client 2 alone | $(-0.010,0.050,0.040,0.040)$ | -0.010 |

All three are predicted to meet the floor. The subset and uniform are
shortlisted. Direct evaluation may reverse their order:

| Candidate | Measured scores for clients 1 to 4 | Worst measured gain |
|---|---|---:|
| Uniform | $(0.326,0.281,0.638,0.513)$ | 0.006 |
| Three-client subset | $(0.323,0.292,0.636,0.522)$ | 0.003 |

Both pass the floor, but uniform now has the larger worst-client gain. The
method therefore accepts uniform. Candidate search does not assume that a
new mixture must beat ordinary aggregation.

## Questions to Be Ready For

**Does this keep each client's best score?** No. If the pretrained score is
0.30 and the current score is 0.36, a candidate scoring 0.35 still meets the
floor. Selection aims for good current-round gains; the acceptance constraint
protects the pretrained baseline. A running-best floor is a different design.

**Why can NFCorpus finish below uniform?** The floor compares against the
pretrained model, not a parallel uniform trajectory. Neither an advantage
over uniform nor reduced cross-client variance is enforced. The completed
runs demonstrate that distinction; they do not establish that one simple
change would remove the tradeoff.

**Is it expensive?** Yes. With four clients, response construction needs five
model configurations per client: the broadcast and four solo updates.
Up to two exact checks add more corpus and query encoding, and halving can
add two further checks. Predicted candidate search reuses embeddings but
still requires similarity and ranking computations. Do not describe it as
cost-free or more efficient overall without a measured comparison.

**Does this prove continual retention or privacy?** No. Temporal memories and
sequential-experience evaluation remain proposed. Raw data staying local is
not a formal privacy guarantee. The present results assess retrieval, not
generated-answer quality.

Use the [current results](research-status.md) for the empirical verdict and
the [related-work notes](presentation-literature.md) for the comparison to
existing methods. The [registration](../registration/E3_PREREGISTRATION.md)
governs the experiment rules; these teaching notes do not replace it.
