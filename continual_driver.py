"""Continual training of one shared retriever over a manifest of client experiences: the
baseline arms, per-round records, acquisition references and the evaluation matrix."""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from sentence_transformers import InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from torch.nn import functional as F

import experiences
import regression
from aggregation_schemes import state_dict_sha256
from fedcrag_common import doc_text, resolve_local
from federated_forgetting import (_runtime_provenance, _sha256_file, amp_enabled,
                                  client_train, dump_json, dump_torch, fedavg,
                                  get_adapter_state, get_git_commit, new_model,
                                  response_encode, set_adapter_state)
from memory import ReplayMemory

ARMS = ("frozen", "local", "fedavg", "fedavg-replay", "fedavg-replay-distill")
REPLAY_ARMS = ("fedavg-replay", "fedavg-replay-distill")
SOURCE_FILES = ("continual_driver.py", "experiences.py", "memory.py", "regression.py",
                "validate_continual.py")
SPLITS = ("guard", "test")


def _clone(state):
    return {key: value.clone() for key, value in state.items()}


def _digest(items):
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def training_data(cells, current, replay_ids):
    """The current experience's training queries plus the replay queries, each with the
    judgements of the experience it belongs to."""
    data = {"corpus": cells[current]["corpus"],
            "train_q": dict(cells[current]["train_q"]),
            "train_qrels": dict(cells[current]["train_qrels"])}
    owner = {q: e for e, cell in cells.items() for q in cell["train_q"]}
    for q in replay_ids:
        cell = cells[owner[q]]
        data["train_q"][q] = cell["train_q"][q]
        data["train_qrels"][q] = cell["train_qrels"][q]
    return data


class ReplayDistillLoss(torch.nn.Module):
    """In-batch contrastive loss on every row, plus lambda times the mean squared
    difference between the student's and the teacher's scaled cosine scores on the
    replay rows (label 1)."""

    def __init__(self, model, teacher, lam, scale=20.0):
        super().__init__()
        self.model, self.teacher, self.lam, self.scale = model, teacher, lam, scale

    def _scores(self, network, sentence_features):
        anchors, positives = [network(f)["sentence_embedding"] for f in sentence_features]
        return self.scale * F.normalize(anchors, dim=-1) @ F.normalize(positives, dim=-1).T

    def forward(self, sentence_features, labels):
        scores = self._scores(self.model, sentence_features)
        loss = F.cross_entropy(scores, torch.arange(len(scores), device=scores.device))
        replay = labels > 0.5
        if replay.any():
            with torch.no_grad():
                teacher_scores = self._scores(self.teacher, sentence_features)
            loss = loss + self.lam * ((scores - teacher_scores)[replay] ** 2).mean()
        return loss


def client_train_distill(model, teacher, start_state, teacher_state, data, replay_ids,
                         q_prefix, d_prefix, batch_size, lr, name, lam):
    """One local epoch with the replay rows distilled from the acquisition reference."""
    set_adapter_state(model, start_state)
    set_adapter_state(teacher, teacher_state)
    teacher.eval()
    replay = set(replay_ids)
    examples = [InputExample(texts=[q_prefix + data["train_q"][qid],
                                    d_prefix + doc_text(data["corpus"][did])],
                             label=float(qid in replay))
                for qid, rels in data["train_qrels"].items()
                for did, rel in rels.items() if rel > 0 and did in data["corpus"]]
    loader = NoDuplicatesDataLoader(examples, batch_size=min(batch_size, len(examples)))
    steps = len(loader)
    model.fit(train_objectives=[(loader, ReplayDistillLoss(model, teacher, lam))],
              epochs=1, steps_per_epoch=steps, optimizer_params={"lr": lr},
              warmup_steps=max(1, int(0.1 * steps)), show_progress_bar=False,
              use_amp=amp_enabled())
    return get_adapter_state(model), len(examples), steps


def _summarise(scores):
    per_query = {m: {q: float(v[m]) for q, v in scores.items()} for m in regression.MEASURES}
    means = {m: (float(np.mean(list(per_query[m].values()))) if per_query[m] else float("nan"))
             for m in regression.MEASURES}
    return {"per_query": per_query, **means}


def evaluate(model, state, corpus, cells, experience_ids, q_prefix, d_prefix, batch_size):
    """Per-query scores of ``state`` on the guard and test queries of the given experiences
    of one client, against its fixed corpus encoded once."""
    cids = list(corpus)
    c_emb = response_encode(model, state, [d_prefix + doc_text(corpus[c]) for c in cids],
                            batch_size)
    out = {}
    for e in experience_ids:
        cell = cells[e]
        out[str(e)] = {}
        for split in SPLITS:
            qids = list(cell[f"{split}_q"])
            q_emb = response_encode(model, state, [q_prefix + cell[f"{split}_q"][q] for q in qids],
                                    batch_size)
            out[str(e)][split] = _summarise(regression.per_query_scores(
                cids, c_emb, qids, q_emb, cell[f"{split}_qrels"]))
    return out


def summarise(out):
    """Acquisition per (client, experience), positive-part regression, backward transfer
    and peak forgetting per historical cell, and the gate scalars A and G."""
    orders, T = out["order"], out["experiences_per_client"]
    matrix = out["matrix"]
    acquisition, cells = {}, {}
    for c in out["clients"]:
        acquisition[c] = {}
        for t in range(T):
            e = str(orders[c][t])
            current = matrix[str(t)][c][e]["test"]["per_query"]["ndcg@10"]
            frozen = out["frozen"][c][e]["test"]["per_query"]["ndcg@10"]
            acquisition[c][e] = regression.mean_difference(current, frozen)
            reference = out["references"][c][e]["scores"]["test"]["per_query"]["ndcg@10"]
            history = [matrix[str(v)][c][e]["test"]["per_query"]["ndcg@10"] for v in range(t, T)]
            for offset, later in enumerate(history[1:], start=t + 1):
                cells[f"{c}:{e}:{offset}"] = {
                    "regression": regression.positive_regression(reference, later),
                    "bwt": regression.mean_difference(later, reference),
                    "peak_forgetting": regression.peak_forgetting(history[:offset - t + 1])}
    summary = {"acquisition": acquisition,
               "A": float(np.mean([a for by in acquisition.values() for a in by.values()])),
               "regression": {"cells": cells}}
    if cells:
        by_cell = {tuple(k.split(":")): v["regression"] for k, v in cells.items()}
        summary["regression"].update(regression.cell_summary(by_cell, threshold=0.010))
        summary["G"] = summary["regression"]["mean"]
    return summary


def _hashes(payload):
    return {key: ({c: state_dict_sha256(s) for c, s in value.items()}
                  if key in ("clients", "clients_before") else state_dict_sha256(value))
            for key, value in payload.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", default="./beir_data")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--model", default="contriever")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True, help="rounds per experience")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--memory_budget", type=int, default=256)
    ap.add_argument("--lambda_distill", type=float, default=1.0)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    experiences.verify_manifest(manifest)
    manifest_sha = _sha256_file(args.manifest)
    clients = sorted(manifest["clients"], key=int)
    T, R = manifest["experiences_per_client"], args.rounds
    orders = {c: list(manifest["clients"][c]["order"]) for c in clients}
    corpora = {c: experiences.client_corpus(manifest, args.data_root, c) for c in clients}
    cells = {c: {e: experiences.materialise(manifest, args.data_root, c, e, corpora[c])
                 for e in range(T)} for c in clients}

    model_path, q_prefix, d_prefix, fp16 = resolve_local(args.model)
    commit = get_git_commit()
    model, module_scales = new_model(args.model, model_path, args.lora_rank, fp16,
                                     "trainable-ab", grad_ckpt=not args.no_grad_ckpt)
    teacher = (new_model(args.model, model_path, args.lora_rank, fp16, "trainable-ab",
                         grad_ckpt=False)[0] if args.arm == "fedavg-replay-distill" else None)
    initial = get_adapter_state(model)

    os.makedirs(args.out, exist_ok=True)
    tag = (f"continual_{args.model.replace('/', '_')}_seed{args.seed}_{args.arm}"
           f"_r{R}_m{manifest_sha[:8]}")
    jpath = os.path.join(args.out, tag + ".json")
    if os.path.exists(jpath):
        raise SystemExit(f"refusing to overwrite the existing result {jpath}")
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"arm": args.arm, "seed": args.seed, "manifest_path": os.path.abspath(args.manifest),
           "manifest_sha256": manifest_sha, "clients": clients, "order": orders,
           "experiences_per_client": T, "rounds_per_experience": R, "commit": commit,
           "args": vars(args), "initial_state_sha256": state_dict_sha256(initial),
           "provenance": _runtime_provenance(commit, args.model, model_path, model,
                                             module_scales, args.data_root,
                                             {"manifest": manifest_sha}),
           "continual_source_sha256": {n: _sha256_file(os.path.join(here, n))
                                       for n in SOURCE_FILES},
           "rounds": [], "references": {c: {} for c in clients}, "matrix": {},
           "frozen": {}, "eval_pool": {}, "summary": {}}
    for c in clients:
        out["frozen"][c] = evaluate(model, initial, corpora[c], cells[c], range(T),
                                    q_prefix, d_prefix, args.eval_batch_size)
    dump_json(out, jpath)

    local = args.arm == "local"
    replay = args.arm in REPLAY_ARMS
    memories = {c: ReplayMemory(args.memory_budget if replay else 0, args.seed * 100 + int(c))
                for c in clients}
    states = {c: _clone(initial) for c in clients} if local else None
    global_state = None if local else _clone(initial)
    teacher_state = None
    for t in range(T):
        current = {c: orders[c][t] for c in clients}
        if args.arm != "frozen":
            for c in clients:
                memories[c].refill({str(orders[c][s]): list(cells[c][orders[c][s]]["train_q"])
                                    for s in range(t)})
            for r in range(R):
                record = {"position": t, "round": r, "experience": current,
                          "memory": {c: {**memories[c].record(), "used": memories[c].used}
                                     for c in clients},
                          "training_ids_sha256": {}, "examples": {}, "steps": {}}
                broadcast = None if local else _clone(global_state)
                trained = {}
                for c in clients:
                    data = training_data(cells[c], current[c], memories[c].ids)
                    record["training_ids_sha256"][c] = _digest(sorted(data["train_q"]))
                    start = states[c] if local else broadcast
                    if args.arm == "fedavg-replay-distill" and teacher_state is not None:
                        new, n_examples, n_steps = client_train_distill(
                            model, teacher, start, teacher_state, data, memories[c].ids,
                            q_prefix, d_prefix, args.batch_size, args.lr, c,
                            args.lambda_distill)
                    else:
                        new, n_examples, n_steps = client_train(
                            model, start, data, q_prefix, d_prefix, 1, args.batch_size,
                            args.lr, c)
                    trained[c] = new
                    record["examples"][c], record["steps"][c] = n_examples, n_steps
                if local:
                    payload = {"clients_before": states, "clients": trained}
                    states = {c: _clone(trained[c]) for c in clients}
                else:
                    global_state = fedavg([trained[c] for c in clients])
                    payload = {"broadcast": broadcast, "clients": trained,
                               "global": global_state}
                record["hashes"] = _hashes(payload)
                record["state_file"] = f"{tag}_p{t}_r{r}.pt"
                dump_torch(payload, os.path.join(args.out, record["state_file"]))
                out["rounds"].append(record)
                dump_json(out, jpath)
                torch.cuda.empty_cache()
        out["matrix"][str(t)] = {}
        for c in clients:
            state = initial if args.arm == "frozen" else (states[c] if local else global_state)
            scored = evaluate(model, state, corpora[c], cells[c],
                              [orders[c][s] for s in range(t + 1)],
                              q_prefix, d_prefix, args.eval_batch_size)
            out["matrix"][str(t)][c] = scored
            out["references"][c][str(current[c])] = {
                "position": t, "sha256": state_dict_sha256(state),
                "state_file": out["rounds"][-1]["state_file"] if out["rounds"] else None,
                "scores": scored[str(current[c])]}
        if not local and args.arm != "frozen":
            teacher_state = _clone(global_state)
        dump_json(out, jpath)

    for c in clients:
        state = initial if args.arm == "frozen" else (states[c] if local else global_state)
        queries, qrels = experiences.eval_queries(manifest, args.data_root, c)
        qids = sorted(queries, key=int)
        cids = list(corpora[c])
        c_emb = response_encode(model, state, [d_prefix + doc_text(corpora[c][x]) for x in cids],
                                args.eval_batch_size)
        q_emb = response_encode(model, state, [q_prefix + queries[q] for q in qids],
                                args.eval_batch_size)
        scored = _summarise(regression.per_query_scores(cids, c_emb, qids, q_emb, qrels))
        out["eval_pool"][c] = {m: scored[m] for m in regression.MEASURES} | {
            "n": len(qids), "source": manifest["clients"][c]["eval_source"]}
    out["summary"] = summarise(out)
    dump_json(out, jpath)
    print(f"saved {jpath}")


if __name__ == "__main__":
    main()
