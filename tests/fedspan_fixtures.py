"""Frozen-A federations whose effective geometry is prescribed exactly.

A test that wants to assert *what* the direction solver should return needs a
federation whose cosine Gram it chose, not one it read back out of the solve.
``federation_with_cosine_gram`` realizes an arbitrary positive-definite cosine
Gram as genuine adapter states: shared row-orthonormal A, zero broadcast B, and
client raw-B blocks whose PEFT-scaled concatenation reproduces the requested
Gram and the requested per-client norms.
"""
import numpy as np
import torch

D_OUT = 8
RANK = 2
D_IN = 6
MODULES = ("layer0.attn.q", "layer1.ffn.up")
EFFECTIVE_DIM = len(MODULES) * D_OUT * RANK


def default_scales():
    return {MODULES[0]: 0.5, MODULES[1]: 2.0}


def _broadcast_state(rng, dtype):
    state = {}
    for module in MODULES:
        basis, _ = np.linalg.qr(rng.normal(size=(D_IN, RANK)))
        state[f"{module}.lora_A.weight"] = torch.tensor(
            basis[:, :RANK].T.copy(), dtype=dtype)
        state[f"{module}.lora_B.weight"] = torch.zeros(
            D_OUT, RANK, dtype=dtype)
    return state


def _client_from_effective_vector(broadcast, vector, scales, dtype):
    """Raw-B blocks whose PEFT-scaled concatenation equals ``vector``."""
    state = {key: value.clone() for key, value in broadcast.items()}
    offset = 0
    for module in MODULES:
        count = D_OUT * RANK
        block = vector[offset:offset + count].reshape(D_OUT, RANK)
        state[f"{module}.lora_B.weight"] = (
            broadcast[f"{module}.lora_B.weight"]
            + torch.tensor(block / scales[module], dtype=dtype))
        offset += count
    return state


def federation_with_cosine_gram(cosine_gram, radii, seed=11,
                                scales=None, dtype=torch.float64):
    """(broadcast, clients, scales) realizing an exact effective geometry.

    ``cosine_gram`` must be positive definite. Client k's effective update is
    ``radii[k]`` long and its unit direction has the requested pairwise
    cosines with every other client.
    """
    scales = dict(scales or default_scales())
    C = np.asarray(cosine_gram, dtype=np.float64)
    factor = np.linalg.cholesky(C)
    rng = np.random.default_rng(seed)
    broadcast = _broadcast_state(rng, dtype)
    basis, _ = np.linalg.qr(rng.normal(size=(EFFECTIVE_DIM, C.shape[0])))
    clients = [
        _client_from_effective_vector(
            broadcast, basis @ factor[index] * float(radii[index]),
            scales, dtype)
        for index in range(C.shape[0])
    ]
    return broadcast, clients, scales


def federation_from_unit_directions(directions, radii, seed=11,
                                    scales=None, dtype=torch.float64):
    """(broadcast, clients, scales) whose effective updates are the rows given.

    Rows of ``directions`` are unit vectors in any dimension up to
    ``EFFECTIVE_DIM``; they are placed in an orthonormal basis of the
    concatenated effective space, so every inner product is preserved.
    """
    scales = dict(scales or default_scales())
    U = np.asarray(directions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    broadcast = _broadcast_state(rng, dtype)
    basis, _ = np.linalg.qr(rng.normal(size=(EFFECTIVE_DIM, U.shape[1])))
    clients = [
        _client_from_effective_vector(
            broadcast, basis @ U[index] * float(radii[index]), scales, dtype)
        for index in range(U.shape[0])
    ]
    return broadcast, clients, scales


def federation_with_norms(norms, seed=23, scales=None, dtype=torch.float64):
    """A federation of mutually orthogonal clients with the exact norms given.

    Orthogonality keeps the cosine Gram the identity, so activity-gate tests
    are not confounded by direction geometry.
    """
    count = len(norms)
    return federation_with_cosine_gram(
        np.eye(count), norms, seed=seed, scales=scales, dtype=dtype)


def effective_delta_vector(state, broadcast, scales):
    """Independent materialization of client k's effective update, flattened.

    Uses the shared frozen A explicitly, so it exercises the same definition
    the method claims rather than the factor-space shortcut the solver uses.
    """
    parts = []
    for module in MODULES:
        delta_b = (state[f"{module}.lora_B.weight"].detach().cpu().double()
                   - broadcast[f"{module}.lora_B.weight"].detach().cpu().double())
        a = broadcast[f"{module}.lora_A.weight"].detach().cpu().double()
        parts.append(scales[module] * (delta_b @ a).reshape(-1))
    return torch.cat(parts)


def effective_cosine_gram(clients, broadcast, scales):
    """Cosine Gram measured from materialized dense updates, not the solver."""
    vectors = [effective_delta_vector(state, broadcast, scales)
               for state in clients]
    stacked = torch.stack(vectors).numpy()
    norms = np.linalg.norm(stacked, axis=1)
    return (stacked @ stacked.T) / np.outer(norms, norms), norms
