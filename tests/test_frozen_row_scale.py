"""The frozen-A row scale ``c`` in ``A A^T = c^2 I``, and who is allowed to
carry it.

Two different scale families exist and are NOT interchangeable:

  * the geometry scale ``sigma * c``, correct in raw-B space, where
    ``||sigma*c*dB||_F == ||sigma * dB @ A||_F``;
  * the bare PEFT scale ``sigma``, correct in any space that already
    materializes ``A`` and therefore already carries ``c``.

Every fixture here uses a genuine PEFT initialization, so ``c != 1`` and the
two families are numerically distinguishable. Fixtures with row-orthonormal
A (c = 1) cannot tell them apart at all.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregation_schemes import (  # noqa: E402
    client_update_norms,
    configure_frozen_lora_a,
    peft_scales,
)

HIDDEN, INTERMEDIATE, RANK, ALPHA = 128, 256, 8, 16
SIGMA = ALPHA / RANK


def peft_model(seed=42):
    """A real PEFT LoRA BERT; its A is kaiming-init, so its row RMS is not 1."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertConfig, BertModel

    torch.manual_seed(seed)
    base = BertModel(BertConfig(
        hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=2, num_attention_heads=4, vocab_size=64))
    return get_peft_model(base, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=RANK, lora_alpha=ALPHA,
        lora_dropout=0.0, target_modules=["query", "value"]))


def lora_a_modules(model, adapter_name="default"):
    return {name: module.lora_A[adapter_name].weight
            for name, module in model.named_modules()
            if getattr(module, "lora_A", None) is not None
            and adapter_name in module.lora_A}


def measured_row_rms(weight):
    """The pre-orthogonalization row RMS, measured the way the cure defines it."""
    source = weight.detach().cpu().double().T
    return float(torch.sqrt(torch.mean(torch.sum(source ** 2, dim=0))).item())


def frozen_a_states(model, client_multipliers=(0.25, -1.75, 0.6)):
    """(broadcast, clients) sharing the model's frozen A with distinct raw B."""
    broadcast = {}
    for name, weight in lora_a_modules(model).items():
        broadcast[f"{name}.lora_A.weight"] = weight.detach().cpu().clone()
        broadcast[f"{name}.lora_B.weight"] = torch.zeros(
            HIDDEN, RANK, dtype=weight.dtype)
    clients = []
    generator = torch.Generator().manual_seed(5)
    for multiplier in client_multipliers:
        state = {key: value.clone() for key, value in broadcast.items()}
        for key in sorted(state):
            if key.endswith(".lora_B.weight"):
                state[key] = state[key] + multiplier * torch.rand(
                    state[key].shape, generator=generator)
        clients.append(state)
    return broadcast, clients


def materialized_effective_norms(clients, broadcast, sigma):
    """||sigma * (B_k A - B_g A_g)||_F per client, built densely from A itself."""
    norms = []
    for state in clients:
        total = 0.0
        for key in sorted(broadcast):
            if not key.endswith(".lora_B.weight"):
                continue
            a_key = key.replace(".lora_B.weight", ".lora_A.weight")
            update = (state[key].double() @ state[a_key].double()
                      - broadcast[key].double() @ broadcast[a_key].double())
            total += float(torch.sum((sigma * update) ** 2).item())
        norms.append(math.sqrt(total))
    return norms


# ------------------------------------------------- what configure_ returns


def test_peft_init_scale_is_sigma_times_the_measured_init_row_rms():
    model = peft_model()
    expected_c = {name: measured_row_rms(weight)
                  for name, weight in lora_a_modules(model).items()}
    scales = configure_frozen_lora_a(model, row_scale="peft-init")

    assert sorted(scales) == sorted(expected_c)
    for name, c in expected_c.items():
        assert scales[name] == pytest.approx(SIGMA * c, rel=1e-12)
        record = scales.records[name]
        assert record["peft_scale"] == pytest.approx(SIGMA)
        assert record["row_scale_mode"] == "peft-init"
        assert record["row_scale_c"] == pytest.approx(c, rel=1e-12)
        assert record["measured_init_row_rms"] == pytest.approx(c, rel=1e-12)
        assert record["geometry_scale"] == pytest.approx(SIGMA * c, rel=1e-12)


def test_peft_init_rows_are_kaiming_scale_not_unit_scale():
    """The whole point of 'peft-init': c ~ 1/sqrt(3), not 1."""
    model = peft_model()
    scales = configure_frozen_lora_a(model, row_scale="peft-init")
    measured = [record["row_scale_c"] for record in scales.records.values()]

    assert len(measured) == 4
    # Kaiming-uniform A over fan_in columns has row RMS 1/sqrt(3) in
    # expectation; measured range over these four modules at seed 42 is
    # [0.568246, 0.579652], i.e. within 2% of 0.577350.
    for c in measured:
        assert c == pytest.approx(1.0 / math.sqrt(3.0), rel=0.02)
        assert c < 0.99


def test_unit_rows_exceed_peft_init_rows_by_exactly_one_over_c():
    """The sqrt(3) rescale the unit-scale E0 control row exists to measure."""
    unit = configure_frozen_lora_a(peft_model(), row_scale="unit")
    peft_init = configure_frozen_lora_a(peft_model(), row_scale="peft-init")

    assert sorted(unit) == sorted(peft_init)
    for name in unit:
        c = peft_init.records[name]["row_scale_c"]
        assert unit[name] == pytest.approx(SIGMA, rel=1e-12)
        assert unit.records[name]["row_scale_c"] == 1.0
        assert unit[name] / peft_init[name] == pytest.approx(1.0 / c, rel=1e-12)
        assert unit[name] / peft_init[name] == pytest.approx(
            math.sqrt(3.0), rel=0.02)


@pytest.mark.parametrize("row_scale", ["unit", "peft-init", 3.0])
def test_frozen_rows_satisfy_a_a_transpose_equals_c_squared_identity(row_scale):
    model = peft_model()
    scales = configure_frozen_lora_a(model, row_scale=row_scale)

    for name, weight in lora_a_modules(model).items():
        c = scales.records[name]["row_scale_c"]
        A = weight.detach().cpu().double()
        gram = A @ A.T
        identity = (c ** 2) * torch.eye(RANK, dtype=torch.float64)
        # A is stored back in the model's float32 dtype, so the achievable
        # deviation is float32 roundoff: measured worst |error|/c^2 over these
        # three modes is 1.45e-08.
        tolerance = 1e-6 * max(1.0, c ** 2)
        assert torch.allclose(gram, identity, atol=tolerance, rtol=0.0)
        assert scales[name] == pytest.approx(SIGMA * c, rel=1e-12)


# ------------------------------------- who may carry c into which space


def test_peft_scales_strips_c_and_leaves_scale_free_mappings_alone():
    scales = configure_frozen_lora_a(peft_model(), row_scale="peft-init")
    bare = peft_scales(scales)

    for name in scales:
        assert bare[name] == pytest.approx(SIGMA, rel=1e-12)
        # The geometry mapping must not be mutated by deriving the bare one.
        assert scales[name] == pytest.approx(
            SIGMA * scales.records[name]["row_scale_c"], rel=1e-12)
    assert peft_scales(2.0) == 2.0
    assert peft_scales({"m": 2.0}) == {"m": 2.0}
    assert peft_scales(None) is None


def test_client_update_norms_equal_the_materialized_step_at_non_unit_c():
    """Regression for the c double-count.

    ``update_gram`` contracts the real A tensors, so the materialized update
    already carries c; scaling it again by the geometry scale sigma*c reports
    c times the truth. Only the bare PEFT scale reproduces
    ``||sigma * dB @ A||_F``.
    """
    model = peft_model()
    scales = configure_frozen_lora_a(model, row_scale="peft-init")
    broadcast, clients = frozen_a_states(model)
    truth = materialized_effective_norms(clients, broadcast, SIGMA)

    reported = client_update_norms(
        clients, broadcast, module_scales=peft_scales(scales))

    assert len(truth) == 3 and min(truth) > 0.0
    for got, want in zip(reported, truth):
        assert got == pytest.approx(want, rel=1e-9)


def test_geometry_scales_in_materialized_space_deflate_the_norm_by_c():
    """The fixture must be able to tell the two scale families apart.

    If this ratio were 1 the regression test above would be vacuous, which is
    exactly the state a c = 1 fixture leaves it in.
    """
    model = peft_model()
    scales = configure_frozen_lora_a(model, row_scale="peft-init")
    broadcast, clients = frozen_a_states(model)
    truth = materialized_effective_norms(clients, broadcast, SIGMA)

    deflated = client_update_norms(
        clients, broadcast, module_scales=scales)

    # Uniform c would give exactly c; the per-module spread makes it an
    # energy-weighted RMS of the c_l, measured here in [0.568, 0.580].
    for got, want in zip(deflated, truth):
        ratio = got / want
        assert 0.568 <= ratio <= 0.580
        assert ratio == pytest.approx(1.0 / math.sqrt(3.0), rel=0.02)
