"""Arm F (registration section 15): a reference-based acceptance check on the server's
update. Guard queries are scored within a small fixed pool per client, and the server keeps
the largest tried step whose guard regression stays within the threshold."""
import numpy as np

GUARD_SLOTS = 64
POOL_RANDOM = 1000
STEPS = (1.0, 0.5, 0.25)
THRESHOLD = 0.010


def check_pool(corpus_ids, guard_qrels, guard_hits, seed):
    """The guard queries' relevant passages and BM25 hits, plus POOL_RANDOM other passages
    of the client's corpus drawn with ``seed``."""
    corpus = set(corpus_ids)
    fixed = {p for rels in guard_qrels.values() for p, rel in rels.items() if rel > 0}
    fixed |= {p for q in guard_qrels for p in guard_hits[q]}
    if not fixed <= corpus:
        raise ValueError(f"{len(fixed - corpus)} guard passages are outside the corpus")
    rest = sorted(corpus - fixed)
    drawn = np.random.default_rng(seed).choice(len(rest), size=min(POOL_RANDOM, len(rest)),
                                               replace=False)
    return sorted(fixed | {rest[i] for i in drawn})


def step_state(broadcast, average, step):
    """broadcast + step * (average - broadcast), exact at steps 0 and 1."""
    if step == 1.0:
        return {k: v.clone() for k, v in average.items()}
    if step == 0.0:
        return {k: v.clone() for k, v in broadcast.items()}
    return {k: broadcast[k] + step * (average[k] - broadcast[k]) for k in broadcast}


def choose_step(regression_at):
    """Try STEPS in order and keep the first whose guard regression is at most THRESHOLD;
    keep the broadcast model (step 0) when none qualifies."""
    tried = []
    for step in STEPS:
        tried.append([step, float(regression_at(step))])
        if tried[-1][1] <= THRESHOLD:
            return step, tried
    return 0.0, tried


def rule_holds(step, tried):
    """Whether a recorded choice follows choose_step."""
    steps = [s for s, _ in tried]
    if steps != list(STEPS[:len(steps)]) or not steps:
        return False
    if any(value <= THRESHOLD for _, value in tried[:-1]):
        return False
    if tried[-1][1] <= THRESHOLD:
        return step == tried[-1][0]
    return step == 0.0 and len(tried) == len(STEPS)
