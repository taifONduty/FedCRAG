import os
import json
import argparse
import hashlib
import importlib.metadata
import platform
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from peft import (LoraConfig, TaskType,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from fedcrag_common import (load_slice_with_train, doc_text, resolve_local,
                            evaluate_metrics, check_lora_targets, LORA_TARGETS,
                            amp_enabled, get_git_commit)


def get_adapter_state(model):
    inner = model[0].auto_model
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(inner).items()}


def set_adapter_state(model, state):
    inner = model[0].auto_model
    set_peft_model_state_dict(inner, state)


# Gram/LP/QP/delta-space weighting machinery lives in aggregation_schemes.py
# (pure math, unit-tested in tests/test_aggregation.py). The trace identity
# used there is pinned by those tests against dense reconstruction; the
# campaign-era inline copy it replaced is not present in this repository, so
# no comparison against it can be checked from this tree.
from aggregation_schemes import (SchemeResult, afl_update, apply_delta_weights,
                                 apply_fedspan_update,
                                 apply_frozen_b_delta_weights,
                                 client_update_norms,
                                 configure_frozen_lora_a,
                                 fednova_delta_weights,
                                 fedspan_delta_weights, maxmin_weights,
                                 frozen_a_state_diagnostics,
                                 mgda_weights, qffl_delta_weights,
                                 state_dict_sha256, update_gram,
                                 validate_frozen_a_states)

LORA_A_SUFFIX = ".lora_A.weight"


def _assert_finite_state(state, context):
    """Reject a nonfinite aggregated state, naming the offending tensor.

    A single nonfinite entry in one client's factor propagates through
    weighted averaging into every subsequent round, so the poisoned key has
    to be named at the round that produced it rather than inferred later.
    """
    for key in sorted(state):
        value = state[key]
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            bad = int(value.numel() - int(finite.sum()))
            raise ValueError(
                f"{context} produced a nonfinite aggregated tensor: "
                f"'{key}' has {bad} of {value.numel()} nonfinite entries "
                "(at least one client update was nonfinite)")


def fedavg(states, weights=None):
    if weights is None:
        weights = [1.0] * len(states)
    total = sum(weights)
    if total <= 0:
        print("  WARNING: all FedAvg weights are zero (no client trained?); "
              "falling back to uniform averaging")
        weights = [1.0] * len(states)
        total = float(len(states))
    weights = [w / total for w in weights]
    avg = {}
    for key in states[0]:
        avg[key] = sum(w * s[key].float() for w, s in zip(weights, states))
    _assert_finite_state(avg, "FedAvg state averaging")
    return avg


def lora_a_norms(state):
    """Per-module ||A||_F of one adapter state, keyed by module name."""
    return {
        key[:-len(LORA_A_SUFFIX)]: float(
            torch.linalg.vector_norm(value.detach().cpu().double()).item())
        for key, value in sorted(state.items())
        if key.endswith(LORA_A_SUFFIX)
    }


def make_examples(data, q_prefix, d_prefix):
    ex = []
    for qid, rels in data["train_qrels"].items():
        if qid not in data["train_q"]:
            continue
        for did, rel in rels.items():
            if rel > 0 and did in data["corpus"]:
                ex.append(InputExample(texts=[q_prefix + data["train_q"][qid],
                                              d_prefix + doc_text(data["corpus"][did])]))
    return ex


def new_model(model_name, model_path, lora_rank, fp16,
              lora_mode="trainable-ab", grad_ckpt=True, row_scale="unit"):
    model = SentenceTransformer(model_path, trust_remote_code=True)
    if fp16:
        model.half()
    check_lora_targets(model, model_name)
    cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                     r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.1,
                     target_modules=LORA_TARGETS)
    model.add_adapter(cfg)
    if lora_mode == "frozen-a":
        module_scales = configure_frozen_lora_a(
            model[0].auto_model, adapter_name="default", row_scale=row_scale)
    elif lora_mode == "trainable-ab":
        # All modules use this script's single LoraConfig, so alpha/r is
        # exactly common. Keep it explicit for PEFT-scale-aware Grams.
        module_scales = float(cfg.lora_alpha) / float(cfg.r)
    else:
        raise ValueError(f"unknown lora_mode: {lora_mode}")
    if grad_ckpt:
        model[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, module_scales


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_fingerprints(data):
    """Canonical content identifiers for exactly the loaded experiment data."""
    fingerprints = {}
    for slice_name, payload in sorted(data.items()):
        digest = hashlib.sha256()
        for section in ("corpus", "train_q", "train_qrels",
                        "eval_q", "eval_qrels"):
            digest.update(section.encode("utf-8") + b"\0")
            values = payload.get(section, {})
            for key in sorted(values, key=str):
                digest.update(str(key).encode("utf-8") + b"\0")
                encoded = json.dumps(
                    values[key], sort_keys=True, ensure_ascii=False,
                    separators=(",", ":")).encode("utf-8")
                digest.update(encoded + b"\0")
        fingerprints[slice_name] = digest.hexdigest()
    return fingerprints


def _runtime_provenance(commit, requested_model, model_path, model,
                        module_scales, data_root, data_sha256):
    root = os.path.dirname(os.path.abspath(__file__))
    source_files = ("federated_forgetting.py", "aggregation_schemes.py",
                    "fedcrag_common.py", "requirements.txt")
    versions = {}
    for package in ("torch", "sentence-transformers", "peft", "scipy",
                    "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    installed = {
        (distribution.metadata.get("Name") or "unknown").lower():
            distribution.version
        for distribution in importlib.metadata.distributions()
    }
    installed = dict(sorted(installed.items()))
    environment_json = json.dumps(
        installed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config = model[0].auto_model.config
    config_json = json.dumps(
        config.to_dict(), sort_keys=True, default=str,
        separators=(",", ":")).encode("utf-8")
    return {
        "git_commit": commit,
        "source_sha256": {
            name: _sha256_file(os.path.join(root, name))
            for name in source_files
        },
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "installed_packages": installed,
        "installed_packages_sha256": hashlib.sha256(
            environment_json).hexdigest(),
        "model": {
            "requested_name": requested_model,
            "resolved_path": model_path,
            "config_name_or_path": getattr(config, "_name_or_path", None),
            "upstream_commit_hash": getattr(config, "_commit_hash", None),
            "config_sha256": hashlib.sha256(config_json).hexdigest(),
        },
        "data_root": os.path.realpath(data_root),
        "data_sha256": data_sha256,
        "module_scales": module_scales,
    }


def _frozen_run_configuration_sha256(args, data_sha256=None, row_scale=None):
    """Hash every result-affecting option for collision-safe frozen-A runs.

    ``row_scale`` is passed resolved rather than read off ``args`` so that an
    explicitly requested default hashes identically to an omitted one.
    """
    fields = {
        "frozen_a_row_scale": row_scale,
        "fedspan_direction_policy": getattr(
            args, "fedspan_direction_policy", None),
        "slices": args.slices,
        "metrics": args.metrics,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": getattr(args, "eval_batch_size", None),
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_mode": args.lora_mode,
        "weighted": args.weighted,
        "weight_by": args.weight_by,
        "qffl_q": args.qffl_q,
        "afl_eta": args.afl_eta,
        "loss_sample": args.loss_sample,
        "max_steps_per_round": args.max_steps_per_round,
        "no_grad_ckpt": getattr(args, "no_grad_ckpt", False),
        "fedspan_step_policy": args.fedspan_step_policy,
        "fedspan_step_norm": args.fedspan_step_norm,
        "fedspan_active_abs_tol": args.fedspan_active_abs_tol,
        "fedspan_active_rel_tol": args.fedspan_active_rel_tol,
        "fedspan_mixture_norm_tol": args.fedspan_mixture_norm_tol,
        "fedspan_max_abs_delta_weight": args.fedspan_max_abs_delta_weight,
        "dirty_provenance_override": getattr(
            args, "allow_dirty_provenance", False),
        "data_sha256": data_sha256,
    }
    encoded = json.dumps(
        fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def client_train(model, global_state, data, q_prefix, d_prefix,
                 epochs, batch_size, lr, name, max_steps=0):
    # max_steps > 0 caps local optimization steps per round (standard FL
    # practice: fixed local work per round rather than full epochs). Without
    # a cap, the largest client (nfcorpus, ~110k pairs) trains a full 3.4k-step
    # epoch EVERY round, which dominates wall-clock and makes R=15 x 3 seeds
    # x 3 weightings infeasible on one GPU. Aggregation weights (n_k) still
    # use example counts, so client size asymmetry is preserved where the
    # weighting arms need it. Convention introduced 2026-08-18; the May pilot
    # (bge-m3, R=5) trained full epochs per round.
    set_adapter_state(model, global_state)
    examples = make_examples(data, q_prefix, d_prefix)
    num_steps = 0
    if not examples:
        print(f"  WARNING: client '{name}' has 0 training pairs -> "
              f"NO local training this round (state = incoming global state)")
    else:
        if len(examples) < batch_size:
            bs = len(examples)
            print(f"  WARNING: client '{name}' has only {len(examples)} train "
                  f"pairs (< batch_size {batch_size}); shrinking batch_size to {bs}")
            loader = DataLoader(examples, shuffle=True, batch_size=bs,
                                drop_last=False)
        else:
            # avoids in-batch false negatives when a query has several positives
            loader = NoDuplicatesDataLoader(examples, batch_size=batch_size)
        if max_steps > 0 and epochs != 1:
            raise ValueError(
                "max_steps_per_round requires local_epochs=1: sentence-"
                "transformers silently drops steps_per_epoch when epochs > 1 "
                "(fit_mixin), which would un-cap the round unnoticed.")
        steps_per_epoch = (min(len(loader), max_steps) if max_steps > 0
                           else len(loader))
        loss_fn = losses.MultipleNegativesRankingLoss(model)
        model.fit(train_objectives=[(loader, loss_fn)], epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  optimizer_params={"lr": lr},
                  warmup_steps=max(1, int(0.1 * steps_per_epoch)),
                  show_progress_bar=False, use_amp=amp_enabled())
        num_steps = steps_per_epoch * epochs
    return get_adapter_state(model), len(examples), num_steps


def estimate_client_losses(model, global_state, data_by_slice, slices,
                           q_prefix, d_prefix, sample, batch_size, rng_seed):
    """Per-client training loss at the BROADCAST point w^t (the evaluation
    point q-FedAvg and AFL prescribe: F_k(w^t), before local training).

    MNRL-consistent estimator: for a deterministic sample of up to ``sample``
    training pairs per client, loss = in-batch-negatives cross-entropy over
    cosine scores at the MNRL default scale of 20. No gradients; batched with
    the eval batch size. The sample is fixed per (seed, round) so all schemes
    see identical losses.
    """
    set_adapter_state(model, global_state)
    out = []
    for s in slices:
        examples = make_examples(data_by_slice[s], q_prefix, d_prefix)
        if not examples:
            out.append(0.0)
            continue
        rng = np.random.RandomState(rng_seed)
        idx = rng.permutation(len(examples))[:sample]
        losses = []
        with torch.no_grad():
            for lo in range(0, len(idx), batch_size):
                chunk = [examples[i] for i in idx[lo:lo + batch_size]]
                q = model.encode([e.texts[0] for e in chunk],
                                 batch_size=batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
                d = model.encode([e.texts[1] for e in chunk],
                                 batch_size=batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
                scores = 20.0 * (q @ d.T)
                target = torch.arange(len(chunk), device=scores.device)
                ce = torch.nn.functional.cross_entropy(scores, target,
                                                       reduction="sum")
                losses.append(float(ce))
        out.append(sum(losses) / max(len(idx), 1))
    return out


def eval_global(model, global_state, data, slices, q_prefix, d_prefix,
                metrics, batch_size):
    set_adapter_state(model, global_state)
    scores = {}
    for s in slices:
        corpus = data[s]["corpus"]
        cids = list(corpus.keys())
        qids = [q for q in data[s]["eval_q"]
                if q in data[s]["eval_qrels"] and data[s]["eval_qrels"][q]]
        if not qids:
            scores[s] = {m: float("nan") for m in metrics}
            continue
        c_emb = model.encode([d_prefix + doc_text(corpus[c]) for c in cids],
                             batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([q_prefix + data[s]["eval_q"][q] for q in qids],
                             batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        scores[s] = evaluate_metrics(cids, c_emb, qids, q_emb,
                                     data[s]["eval_qrels"], metrics)
    return scores


def print_scores(label, scores, slices, metrics):
    for s in slices:
        print(f"  [{label}] {s}: "
              + " ".join(f"{m}:{scores[s][m]:.4f}" for m in metrics))


def dump_json(out, jpath):
    tmp = jpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, jpath)


def dump_torch(out, path):
    tmp = path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=["nfcorpus", "fiqa", "scifact", "arguana"])
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--metrics", nargs="+", default=["ndcg@10", "recall@10", "recall@100"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_rounds", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=128,
                    help="encode batch for evaluation only (no gradients; "
                         "affects speed, not results)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_mode", choices=["trainable-ab", "frozen-a"],
                    default="trainable-ab",
                    help="'trainable-ab' preserves the historical ordinary-"
                         "LoRA coordinate; 'frozen-a' row-orthonormalizes one "
                         "shared A at initialization and trains B only")
    ap.add_argument("--frozen_a_row_scale", choices=["unit", "peft-init"],
                    default=None,
                    help="row constant c in A A^T = c^2 I for --lora_mode "
                         "frozen-a (default 'unit'). 'peft-init' rescales the "
                         "orthonormal rows to the module's own measured "
                         "pre-orthogonalization row RMS, so client effective "
                         "step magnitudes are comparable with trainable-ab. "
                         "The two modes are distinct arms, not interchangeable "
                         "settings of one arm")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--weight_by",
                    choices=["examples", "corpus", "maxmin", "rawmaxmin",
                             "normmaxmin",
                             "qffl", "afl", "mgda", "fednova"],
                    default="examples",
                    help="FedAvg weighting basis when --weighted: 'examples' = "
                         "local training-pair count (canonical FedAvg n_k), "
                         "'corpus' = client corpus size (the original F4 run), "
                         "'maxmin'/'rawmaxmin' = historical-style cosine-game "
                         "simplex weights applied by state averaging; "
                         "'normmaxmin' = corrected FedSpan, legal only with "
                         "--lora_mode frozen-a and an explicit "
                         "--fedspan_step_policy; "
                         "E2 external baselines: 'qffl' = q-FedAvg "
                         "(arXiv:1905.10497, loss^q with h_k normalization), "
                         "'afl' = agnostic-FL multiplicative-weights ascent "
                         "(arXiv:1902.00146), 'mgda' = min-norm weights on the "
                         "raw update Gram (arXiv:1810.04650), 'fednova' = "
                         "tau-normalized averaging (arXiv:2007.07481)")
    ap.add_argument("--qffl_q", type=float, default=1.0,
                    help="q-FedAvg fairness exponent q (only with "
                         "--weight_by qffl)")
    ap.add_argument("--afl_eta", type=float, default=0.1,
                    help="AFL mixture-weight ascent step size (only with "
                         "--weight_by afl)")
    ap.add_argument("--loss_sample", type=int, default=2048,
                    help="max training pairs per client for the broadcast-"
                         "point loss estimate (qffl/afl only; deterministic "
                         "per round)")
    ap.add_argument("--fedspan_step_policy",
                    choices=["fixed", "median-active"], default=None,
                    help="explicit true effective-B step policy required for "
                         "--weight_by normmaxmin")
    ap.add_argument("--fedspan_direction_policy",
                    choices=["minnorm", "maxmin-lp"], default=None,
                    help="explicit direction solver required for --weight_by "
                         "normmaxmin. 'minnorm' maximizes the worst-case "
                         "cosine of the direction actually applied (the "
                         "normalized mixture) and equals FedMGDA+ "
                         "(arXiv:2006.11489) at epsilon=1 on the cosine Gram; "
                         "'maxmin-lp' is the historical LP, whose objective is "
                         "not the applied quantity, retained as a recorded "
                         "ablation. Both values are measured every round")
    ap.add_argument("--fedspan_step_norm", type=float, default=None,
                    help="positive finite true effective-B step norm s, "
                         "required only with --fedspan_step_policy fixed")
    ap.add_argument("--fedspan_active_abs_tol", type=float, default=1e-12,
                    help="absolute effective-update norm threshold")
    ap.add_argument("--fedspan_active_rel_tol", type=float, default=1e-8,
                    help="relative threshold times the largest finite client "
                         "effective-update norm")
    ap.add_argument("--fedspan_mixture_norm_tol", type=float, default=1e-6,
                    help="dimensionless near-cancellation threshold for "
                         "sqrt(w^T H w); violations fail closed to no update")
    ap.add_argument("--fedspan_max_abs_delta_weight", type=float, default=None,
                    help="optional declared cap on |c_k|; a violation fails "
                         "closed rather than silently clipping")
    ap.add_argument("--allow_dirty_provenance", action="store_true",
                    help="allow normmaxmin on an unknown/dirty Git tree for "
                         "development only; recorded in output and invalid for "
                         "paper-grade E0--E5 evidence")
    ap.add_argument("--save_states", action="store_true",
                    help="save broadcast + per-client + global adapter states "
                         "and hashes each round "
                         "(needed for mechanism diagnostics, e.g. principal "
                         "angles between client updates)")
    ap.add_argument("--max_steps_per_round", type=int, default=0,
                    help="cap on local optimization steps per client per round "
                         "(0 = full epoch, the May-pilot convention)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable gradient checkpointing (faster when VRAM "
                         "allows; no effect on results)")
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    canonical_weight_by = ("rawmaxmin" if args.weight_by == "maxmin"
                           else args.weight_by)
    if args.weight_by == "maxmin":
        print("  WARNING: --weight_by maxmin is the historical rawmaxmin "
              "alias, not norm-consistent FedSpan")
    if canonical_weight_by == "normmaxmin":
        if not args.weighted:
            ap.error("--weight_by normmaxmin requires --weighted")
        if args.lora_mode != "frozen-a":
            ap.error("--weight_by normmaxmin requires --lora_mode frozen-a")
        if not args.save_states:
            ap.error("--weight_by normmaxmin requires --save_states so exact "
                     "broadcast/client/applied states are preserved")
        if args.fedspan_step_policy is None:
            ap.error("--weight_by normmaxmin requires "
                     "--fedspan_step_policy")
        if args.fedspan_direction_policy is None:
            ap.error("--weight_by normmaxmin requires "
                     "--fedspan_direction_policy")
        if args.fedspan_step_policy == "fixed":
            if (args.fedspan_step_norm is None
                    or not np.isfinite(args.fedspan_step_norm)
                    or args.fedspan_step_norm <= 0):
                ap.error("fixed policy requires a positive finite "
                         "--fedspan_step_norm")
        elif args.fedspan_step_norm is not None:
            ap.error("median-active policy rejects --fedspan_step_norm")
        for option, value in (
                ("--fedspan_active_abs_tol", args.fedspan_active_abs_tol),
                ("--fedspan_active_rel_tol", args.fedspan_active_rel_tol),
                ("--fedspan_mixture_norm_tol",
                 args.fedspan_mixture_norm_tol)):
            if not np.isfinite(value) or value < 0:
                ap.error(f"{option} must be finite and nonnegative")
        if (args.fedspan_max_abs_delta_weight is not None
                and (not np.isfinite(args.fedspan_max_abs_delta_weight)
                     or args.fedspan_max_abs_delta_weight <= 0)):
            ap.error("--fedspan_max_abs_delta_weight must be positive and "
                     "finite when supplied")
    elif (args.fedspan_step_policy is not None
          or args.fedspan_step_norm is not None):
        ap.error("FedSpan step options are legal only with "
                 "--weight_by normmaxmin")
    elif args.fedspan_direction_policy is not None:
        ap.error("--fedspan_direction_policy is legal only with "
                 "--weight_by normmaxmin")
    if args.frozen_a_row_scale is not None and args.lora_mode != "frozen-a":
        ap.error("--frozen_a_row_scale is legal only with "
                 "--lora_mode frozen-a")
    row_scale = args.frozen_a_row_scale or "unit"
    commit = get_git_commit()
    if (canonical_weight_by == "normmaxmin"
            and (commit == "unknown" or commit.endswith("-dirty"))
            and not args.allow_dirty_provenance):
        ap.error("normmaxmin refuses unknown/dirty source provenance; commit "
                 "the exact implementation or use --allow_dirty_provenance "
                 "for development-only runs")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    data = {s: load_slice_with_train(s, args.data_root) for s in args.slices}
    data_sha256 = _data_fingerprints(data)
    model_path, q_prefix, d_prefix, fp16 = resolve_local(args.model)

    model, module_scales = new_model(
        args.model, model_path, args.lora_rank, fp16,
        lora_mode=args.lora_mode, grad_ckpt=not args.no_grad_ckpt,
        row_scale=row_scale)
    global_state = get_adapter_state(model)

    # Preserve historical trainable-A+B filenames. New frozen-A runs add a
    # canonical result-affecting configuration hash so capped/full and other
    # scientifically distinct settings cannot overwrite one another.
    basis = f"weighted-{args.weight_by}" if args.weighted else "unweighted"
    if canonical_weight_by == "normmaxmin":
        if args.fedspan_step_policy == "fixed":
            step_tag = format(args.fedspan_step_norm, ".8g").replace(".", "p")
        else:
            step_tag = "median-active"
        basis += f"-s{step_tag}-dir{args.fedspan_direction_policy}"
    frozen_config_sha256 = None
    if args.lora_mode == "frozen-a":
        frozen_config_sha256 = _frozen_run_configuration_sha256(
            args, data_sha256=data_sha256, row_scale=row_scale)
        basis = (f"frozen-a-{row_scale}_{basis}"
                 f"-cfg{frozen_config_sha256[:12]}")
    tag = f"{basis}_r{args.num_rounds}"
    model_safe = args.model.replace("/", "_")
    jpath = os.path.join(args.out,
                         f"federated_{model_safe}_seed{args.seed}_{tag}.json")

    out = {"seed": args.seed, "slices": args.slices, "model": args.model,
           "metrics": args.metrics, "num_rounds": args.num_rounds,
           "weighted": args.weighted,
           "weight_by": args.weight_by if args.weighted else None,
           "weight_by_canonical": (canonical_weight_by
                                   if args.weighted else None),
           "split_fallback": [s for s in args.slices
                              if data[s].get("split_fallback")],
           "commit": commit, "use_amp": amp_enabled(),
           "lora_mode": args.lora_mode,
           "method_contract": {
               "coordinate": ("effective-b" if args.lora_mode == "frozen-a"
                              else "trainable-ab-factor-state"),
               "shared_row_orthonormal_a": args.lora_mode == "frozen-a",
               "frozen_a_row_scale": (row_scale
                                      if args.lora_mode == "frozen-a"
                                      else None),
               "frozen_a_row_scale_specified": (
                   args.frozen_a_row_scale is not None),
               "frozen_a_row_scale_records": (
                   getattr(module_scales, "records", None)
                   if args.lora_mode == "frozen-a" else None),
               "fedspan_step_policy": args.fedspan_step_policy,
               "fedspan_step_norm": args.fedspan_step_norm,
               "fedspan_direction_policy": args.fedspan_direction_policy,
               "dirty_provenance_override": args.allow_dirty_provenance,
               "initial_adapter_state_sha256": state_dict_sha256(global_state),
               "frozen_a_diagnostics": (
                   frozen_a_state_diagnostics(global_state, module_scales)
                   if args.lora_mode == "frozen-a" else None),
               "run_configuration_sha256": frozen_config_sha256,
           },
           "provenance": _runtime_provenance(
               commit, args.model, model_path, model, module_scales,
               args.data_root, data_sha256),
           "args": vars(args), "clients": {}, "R_matrix": {}, "BWT": None}
    R = out["R_matrix"]

    R["frozen"] = eval_global(model, global_state, data, args.slices,
                              q_prefix, d_prefix, args.metrics,
                              args.eval_batch_size)
    print_scores("frozen", R["frozen"], args.slices, args.metrics)
    dump_json(out, jpath)

    ADAPTIVE = ("maxmin", "rawmaxmin", "normmaxmin", "qffl", "afl",
                "mgda", "fednova")
    afl_lam = [1.0 / len(args.slices)] * len(args.slices)

    for rnd in range(args.num_rounds):
        print(f"  --- round {rnd+1}/{args.num_rounds} ---")
        label = f"round_{rnd+1}"
        round_broadcast = {key: value.clone()
                           for key, value in global_state.items()}
        # qffl/afl evaluate F_k at the broadcast point, BEFORE local training
        losses = None
        if args.weighted and args.weight_by in ("qffl", "afl"):
            losses = estimate_client_losses(
                model, round_broadcast, data, args.slices, q_prefix, d_prefix,
                args.loss_sample, args.eval_batch_size,
                rng_seed=args.seed * 1000 + rnd)
            out.setdefault("client_losses", {})[label] = \
                dict(zip(args.slices, [round(x, 5) for x in losses]))
            print(f"  broadcast-point losses: "
                  f"{[round(x, 3) for x in losses]}")
        states, n_examples, client_stats = [], [], {}
        for s in args.slices:
            st, n_ex, n_steps = client_train(model, round_broadcast, data[s],
                                             q_prefix, d_prefix,
                                             args.local_epochs,
                                             args.batch_size, args.lr, name=s,
                                             max_steps=args.max_steps_per_round)
            states.append(st)
            n_examples.append(n_ex)
            client_stats[s] = {"num_examples": n_ex, "num_steps": n_steps}
        if args.lora_mode == "frozen-a":
            validate_frozen_a_states(
                states, round_broadcast, module_scales=module_scales)
        # Per-client effective step magnitudes, recorded for EVERY arm and
        # coordinate. Without them the frozen-A vs trainable-A+B comparison
        # cannot be separated from a pure step-scale difference after the
        # fact. Cheap: the trace identity keeps this to r x r products.
        try:
            delta_norms = client_update_norms(
                states, round_broadcast, module_scales=module_scales)
            out.setdefault("client_delta_norms", {})[label] = dict(
                zip(args.slices, delta_norms))
        except Exception as exc:
            print(f"  WARNING: client delta norms unavailable: {exc}")
            out.setdefault("client_delta_norms", {})[label] = {
                "error": f"{type(exc).__name__}: {exc}"}
        if args.lora_mode == "trainable-ab":
            # A moves in this coordinate, so the effective step depends on
            # ||A|| as well as ||B||; both sides are needed to attribute it.
            out.setdefault("client_lora_a_norms", {})[label] = {
                name: lora_a_norms(state)
                for name, state in zip(args.slices, states)}
            out.setdefault("broadcast_lora_a_norms", {})[label] = \
                lora_a_norms(round_broadcast)
        # --- aggregation dispatch ---------------------------------------
        # Trainable-A+B simplex schemes retain historical FedAvg. Every
        # frozen-A arm applies raw-B deltas so the broadcast A is copied
        # bit-for-bit; normmaxmin supplies non-simplex true-step coefficients.
        delta_v = None
        precomputed_state = None
        scheme_result = None
        if not args.weighted:
            weights = None
        elif args.weight_by in ("maxmin", "rawmaxmin"):
            weights = maxmin_weights(
                states, round_broadcast, module_scales=module_scales)
            scheme_result = weights
        elif args.weight_by == "normmaxmin":
            fedspan = fedspan_delta_weights(
                states, round_broadcast, module_scales=module_scales,
                step_norm=args.fedspan_step_norm,
                step_policy=args.fedspan_step_policy,
                direction_policy=args.fedspan_direction_policy,
                active_abs_tol=args.fedspan_active_abs_tol,
                active_rel_tol=args.fedspan_active_rel_tol,
                mixture_norm_tol=args.fedspan_mixture_norm_tol,
                max_abs_delta_weight=args.fedspan_max_abs_delta_weight)
            precomputed_state, applied = apply_fedspan_update(
                round_broadcast, states, fedspan,
                module_scales=module_scales)
            fedspan_record = dict(fedspan)
            fedspan_record["application"] = applied
            out.setdefault("fedspan_diagnostics", {})[label] = fedspan_record
            delta_v = fedspan["delta_weights"]
            weights = None
            print("  normmaxmin status="
                  f"{fedspan['status']} gamma={fedspan['gamma']} "
                  f"mixture_norm={fedspan['mixture_norm']} "
                  f"applied_norm={applied['applied_step_norm']}")
            print(f"  normmaxmin direction={fedspan['direction_policy']} "
                  f"achieved={fedspan['achieved_min_direction_cosine']} "
                  f"optimal={fedspan['min_norm_value']} "
                  f"shortfall={fedspan['direction_solver_shortfall']}")
        elif args.weight_by == "mgda":
            weights = mgda_weights(
                states, round_broadcast, module_scales=module_scales)
            scheme_result = weights
        elif args.weight_by == "afl":
            afl_lam = afl_update(afl_lam, losses, args.afl_eta)
            weights = afl_lam
            scheme_result = afl_lam
        elif args.weight_by == "qffl":
            # float64: ||w_k - w^t||^2 is a difference of large nearly equal
            # inner products, and the float32 default loses the small client
            # delta once the broadcast dominates it.
            sq_norms = np.diag(update_gram(
                states, round_broadcast, module_scales=module_scales,
                dtype=torch.float64)).tolist()
            delta_v = qffl_delta_weights(losses, sq_norms, q=args.qffl_q,
                                         L=1.0 / args.lr)
            scheme_result = delta_v
        elif args.weight_by == "fednova":
            delta_v = fednova_delta_weights(
                n_examples, [client_stats[s]["num_steps"]
                             for s in args.slices])
            scheme_result = delta_v
        elif args.weight_by == "corpus":
            weights = [len(data[s]["corpus"]) for s in args.slices]
        else:
            weights = n_examples
        if scheme_result is not None:
            # Symmetric with fedspan_diagnostics: a solver failure that
            # degrades an arm to uniform must stay detectable in the record
            # rather than only in the console log.
            record = (scheme_result.record()
                      if isinstance(scheme_result, SchemeResult)
                      else {"weights": [float(w) for w in scheme_result],
                            "status": "unreported", "solver_status": None,
                            "solver_message": "scheme reported no status",
                            "fallback": None})
            record["scheme"] = canonical_weight_by
            out.setdefault("scheme_diagnostics", {})[label] = record
            if record["fallback"] is not None:
                print(f"  WARNING: {canonical_weight_by} fell back to "
                      f"{record['fallback']} ({record['status']}): "
                      f"{record['solver_message']}")
        if args.weighted and args.weight_by in ADAPTIVE:
            wn = (delta_v if delta_v is not None else
                  (weights if weights is not None
                   else [1.0 / len(states)] * len(states)))
            out.setdefault("round_weights", {})[label] = \
                [round(w, 5) for w in wn]
            if delta_v is not None:
                out["weight_space"] = "delta"
            print(f"  {args.weight_by} weights: {[round(w, 3) for w in wn]}")
        if precomputed_state is not None:
            global_state = precomputed_state
            out["weight_space"] = "raw-b-delta/true-effective-step-norm"
        elif delta_v is not None:
            if args.lora_mode == "frozen-a":
                global_state = apply_frozen_b_delta_weights(
                    round_broadcast, states, delta_v,
                    module_scales=module_scales)
                out["weight_space"] = "raw-b-delta/shared-frozen-a"
            else:
                global_state = apply_delta_weights(
                    round_broadcast, states, delta_v)
                _assert_finite_state(global_state, "delta-space aggregation")
        elif args.lora_mode == "frozen-a":
            frozen_weights = ([1.0] * len(states)
                              if weights is None else list(weights))
            frozen_total = float(sum(frozen_weights))
            if (not np.isfinite(frozen_total) or frozen_total <= 0
                    or not np.all(np.isfinite(frozen_weights))
                    or np.min(frozen_weights) < 0):
                raise ValueError(
                    "frozen-A simplex aggregation received invalid weights")
            frozen_weights = [float(value) / frozen_total
                              for value in frozen_weights]
            global_state = apply_frozen_b_delta_weights(
                round_broadcast, states, frozen_weights,
                module_scales=module_scales)
            out["weight_space"] = "raw-b-delta/shared-frozen-a"
        else:
            global_state = fedavg(states, weights=weights)
        label = f"round_{rnd+1}"
        out["clients"][label] = client_stats
        if args.save_states:
            spath = os.path.join(
                args.out,
                f"states_{model_safe}_seed{args.seed}_{tag}_round{rnd+1}.pt")
            dump_torch({"clients": dict(zip(args.slices, states)),
                        "broadcast": round_broadcast,
                        "global": global_state,
                        "broadcast_state_sha256": state_dict_sha256(
                            round_broadcast),
                        "global_state_sha256": state_dict_sha256(global_state)},
                       spath)
            print(f"  saved adapter states -> {spath}")
        # Persist aggregation/provenance diagnostics before the expensive
        # evaluation so a later evaluation failure cannot erase the method
        # correctness record for this completed server update.
        dump_json(out, jpath)
        R[label] = eval_global(model, global_state, data, args.slices,
                               q_prefix, d_prefix, args.metrics,
                               args.eval_batch_size)
        print_scores(label, R[label], args.slices, args.metrics)
        dump_json(out, jpath)
        torch.cuda.empty_cache()

    bwt = {}
    anchor = "round_1"
    final = f"round_{args.num_rounds}"
    for m in args.metrics:
        terms = [R[final][s][m] - R[anchor][s][m] for s in args.slices]
        bwt[m] = float(np.mean(terms))
        print(f">>> Federated BWT[{m}] (round1->final) = {bwt[m]:+.4f}")

    out["BWT"] = bwt
    dump_json(out, jpath)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
