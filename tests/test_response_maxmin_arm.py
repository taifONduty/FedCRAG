"""The response-maxmin arm through the real driver (design note 2026-09-08)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver_harness  # noqa: E402
import federated_forgetting as driver  # noqa: E402
from validate_e0 import E0ValidationError, validate_run_directory  # noqa: E402

COUNTS = {"c0": 1000, "c1": 250, "c2": 40}
STEPS = {"c0": 31, "c1": 8, "c2": 2}
ARM_FLAGS = ("--response_dev_fraction", "0.25", "--response_dev_min", "2",
             "--response_lattice_step", "0.5", "--response_scales", "1.0",
             "--response_verify", "2")


def test_harness_dev_data_has_forty_queries_with_one_positive_each():
    loader = driver_harness.dev_mock_data()
    data = loader("c0", "/nowhere")
    assert len(data["train_q"]) == 40 and len(data["train_qrels"]) == 40
    assert all(len(v) == 1 for v in data["train_qrels"].values())
    assert data["corpus"]["c0-d3"]["text"] == "c0-d3"
    emb = driver_harness.fake_response_encoder(
        None, driver_harness.broadcast_state(), ["c0-d3", "c0-q3"], 8)
    assert emb.shape == (2, 8) and np.allclose(np.linalg.norm(emb, axis=1), 1.0)
