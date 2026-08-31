"""The 33 registered E3 runs, as data (registration/E3_PREREGISTRATION.md).

Signing commit 1b7d397. Every run's exact CLI arguments derive from here so
the launch script contains no inline experiment definitions; tests pin the
counts, the pairing invariants, and the arities against the signed document.

Design invariants the tests enforce:

- The MAIN grid (6 arms x 3 seeds, m=3) uses ONE shard partition
  (shard_seed 42) for every arm and every training seed, so cross-arm
  comparisons are paired on identical shards. Only the 4
  partition-robustness runs vary shard_seed.
- "K=2 clone control" is read literally as the two-client no-clone anchor
  federation {nfcorpus (unsharded), arguana}: FedSpan's weights should stay
  near uniform when there is no redundancy to discount. This interpretation
  is recorded in the pre-registration's deviations log (2026-08-31).
- The m-sweep cells (nfcorpus:2 and :4, seed 42, FedSpan + norm-equalised
  uniform) are SECONDARY: nothing in prereg SS5 reads them.
- Every run through the FedSpan geometry pipeline carries the registered
  shadow-sketch instrumentation (m in {1024, 4096}); plain-uniform, n_k and
  q-FedAvg runs do not go through that pipeline and cannot carry it.
"""

CAPPED_STEPS = 500
ROUNDS = 15
SEEDS = (42, 123, 2024)
MAIN_SHARD_SEED = 42
ROBUSTNESS_SHARD_SEEDS = (7, 99, 1234, 5678)
SHADOW_SIZES = (1024, 4096)

# Client order after sharding is --slices order with shards expanded in
# place: [nfcorpus-s0 .. nfcorpus-s{m-1}, arguana, scifact].
MAIN_SLICES = ("nfcorpus", "arguana", "scifact")


def _shard_spec(m):
    return (f"nfcorpus:{m}", "arguana:1", "scifact:1")


def _fixed_weights(kind, m):
    clients = m + 2
    if kind == "norm-eq-uniform":
        return (1.0 / clients,) * clients
    if kind == "uniform-over-distributions":
        return (1.0 / 3.0 / m,) * m + (1.0 / 3.0, 1.0 / 3.0)
    raise ValueError(kind)


def _base(name, seed, *, slices=MAIN_SLICES, shard_spec=None,
          shard_seed=None, rounds=ROUNDS):
    run = {
        "name": name,
        "args": ["--model", "contriever",
                 "--slices", *slices,
                 "--metrics", "ndcg@10", "recall@10", "recall@100",
                 "--seed", str(seed),
                 "--num_rounds", str(rounds),
                 "--max_steps_per_round", str(CAPPED_STEPS),
                 "--batch_size", "32", "--eval_batch_size", "256",
                 "--lr", "2e-05", "--lora_rank", "16",
                 "--save_states"],
    }
    if shard_spec is not None:
        run["args"] += ["--shard_spec", *shard_spec,
                        "--shard_seed", str(shard_seed),
                        "--conserve_shard_steps"]
    return run


def _arm(run, arm, m=None):
    """Append the arm's weighting flags; m = clone count for arity."""
    a = run["args"]
    if arm == "fedspan-exact":
        a += ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "peft-init",
              "--weighted", "--weight_by", "normmaxmin",
              "--fedspan_step_policy", "median-active",
              "--fedspan_direction_policy", "exact",
              "--fedspan_shadow_sketch", *map(str, SHADOW_SIZES)]
    elif arm in ("norm-eq-uniform", "uniform-over-distributions"):
        weights = (_fixed_weights(arm, m) if m is not None
                   else (0.5, 0.5))          # K=2 control arity
        a += ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "peft-init",
              "--weighted", "--weight_by", "normmaxmin",
              "--fedspan_step_policy", "median-active",
              "--fedspan_direction_policy", "fixed",
              "--fedspan_fixed_weights", *[f"{w:.12g}" for w in weights],
              "--fedspan_shadow_sketch", *map(str, SHADOW_SIZES)]
    elif arm == "plain-uniform":
        a += ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "peft-init"]
    elif arm == "nk":
        a += ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "peft-init",
              "--weighted", "--weight_by", "examples"]
    elif arm == "qffl":
        # Registered fallback: E2 has not run, so q is pre-registered at 1.0.
        a += ["--lora_mode", "frozen-a", "--frozen_a_row_scale", "peft-init",
              "--weighted", "--weight_by", "qffl", "--qffl_q", "1.0"]
    else:
        raise ValueError(arm)
    run["arm"] = arm
    return run


MAIN_ARMS = ("fedspan-exact", "norm-eq-uniform", "uniform-over-distributions",
             "plain-uniform", "nk", "qffl")


def p0_probe():
    """1-round FedSpan probe whose round-1 diagnostics feed the P0 gate.

    Operational, NOT one of the 33 registered runs; its result is never
    analysed as an outcome.
    """
    run = _base("p0-probe", 42, shard_spec=_shard_spec(3),
                shard_seed=MAIN_SHARD_SEED, rounds=1)
    return _arm(run, "fedspan-exact", m=3)


def registered_runs():
    """All 33 registered runs, launch order = registration order."""
    runs = []
    # 6 arms x 3 seeds, m=3, one shared partition
    for seed in SEEDS:
        for arm in MAIN_ARMS:
            run = _base(f"e3-{arm}-s{seed}", seed, shard_spec=_shard_spec(3),
                        shard_seed=MAIN_SHARD_SEED)
            runs.append(_arm(run, arm, m=3))
    # K=2 no-clone control: {nfcorpus, arguana}, 2 arms x 3 seeds
    for seed in SEEDS:
        for arm in ("fedspan-exact", "norm-eq-uniform"):
            run = _base(f"e3-k2-{arm}-s{seed}", seed,
                        slices=("nfcorpus", "arguana"))
            runs.append(_arm(run, arm, m=None))
    # partition robustness: FedSpan, seed 42, alternate shard seeds
    for shard_seed in ROBUSTNESS_SHARD_SEEDS:
        run = _base(f"e3-partition-ss{shard_seed}", 42,
                    shard_spec=_shard_spec(3), shard_seed=shard_seed)
        runs.append(_arm(run, "fedspan-exact", m=3))
    # m-sweep light (secondary): m in {2, 4}, 2 arms, seed 42
    for m in (2, 4):
        for arm in ("fedspan-exact", "norm-eq-uniform"):
            run = _base(f"e3-msweep-m{m}-{arm}-s42", 42,
                        shard_spec=_shard_spec(m),
                        shard_seed=MAIN_SHARD_SEED)
            runs.append(_arm(run, arm, m=m))
    # nondeterminism floor: exact repeat of the first main FedSpan cell
    run = _base("e3-noise-floor-fedspan-exact-s42", 42,
                shard_spec=_shard_spec(3), shard_seed=MAIN_SHARD_SEED)
    runs.append(_arm(run, "fedspan-exact", m=3))
    return runs


def p0_gate(result, min_clone_cosine=0.15):
    """(passed, report) from a probe result dict's round-1 diagnostics.

    Requires: mean clone-block cosine >= min_clone_cosine AND the mean
    clone-singleton cosine strictly below the clone-block mean — otherwise
    the shards are not a clone federation and E3 tests nothing.
    """
    diag = result["fedspan_diagnostics"]["round_1"]
    cosine = diag["cosine_gram_active"]
    n_clients = len(cosine)
    if n_clients != 5:
        return False, f"expected 5 active clients, found {n_clients}"
    block = [cosine[i][j] for i in range(3) for j in range(i + 1, 3)]
    cross = [cosine[i][j] for i in range(3) for j in (3, 4)]
    block_mean = sum(block) / len(block)
    cross_mean = sum(cross) / len(cross)
    passed = block_mean >= min_clone_cosine and cross_mean < block_mean
    report = (f"clone-block mean cosine {block_mean:.4f} "
              f"(gate >= {min_clone_cosine}), clone-singleton mean "
              f"{cross_mean:.4f} (must be < block mean): "
              + ("PASS" if passed else "FAIL"))
    return passed, report
