"""server_average.py: window 1 reproduces a run; a wider window deploys the average of the
last rounds' global models; the running blend adds LoRA updates exactly."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server_average  # noqa: E402
from aggregation_schemes import state_dict_sha256  # noqa: E402
from test_continual_driver import run, stream  # noqa: E402,F401


def average_run(source, root, window, out, rule="--window"):
    server_average.main(["--source", str(source), rule, str(window),
                         "--data_root", str(root), "--out", str(out)])
    return json.loads(next(Path(out).glob("continual_*.json")).read_text())


def test_window_one_reproduces_the_run_and_window_two_deploys_the_average(stream, tmp_path):
    source, _, _ = run(stream, "fedavg-replay", tmp_path / "run")
    one = average_run(tmp_path / "run", stream[0], 1, tmp_path / "k1")
    two = average_run(tmp_path / "run", stream[0], 2, tmp_path / "k2")
    assert one["matrix"] == source["matrix"] and one["reproduction"] == {"A": 0.0, "G": 0.0}
    for t in range(2):
        states = [torch.load(tmp_path / "run" / r["state_file"], weights_only=True)["global"]
                  for r in source["rounds"] if r["position"] == t]
        deployed = {ref["sha256"] for refs in two["references"].values()
                    for ref in refs.values() if ref["position"] == t}
        assert deployed == {state_dict_sha256(server_average.average(states))}
        assert deployed != {ref["sha256"] for refs in source["references"].values()
                            for ref in refs.values() if ref["position"] == t}
    with pytest.raises(SystemExit, match="cannot average"):
        average_run(tmp_path / "run", stream[0], 3, tmp_path / "k3")


def test_the_blend_stacks_updates_exactly_with_running_weights(stream, tmp_path):
    torch.manual_seed(0)
    states = [{"m.lora_A.weight": torch.randn(2, 5), "m.lora_B.weight": torch.randn(3, 2)}
              for _ in range(3)]
    weights = server_average.blend_weights(2, 0.25)
    stacked = server_average.stack(list(zip(weights, states)))
    dense = sum(w * s["m.lora_B.weight"] @ s["m.lora_A.weight"] for w, s in zip(weights, states))
    assert stacked["m.lora_B.weight"] @ stacked["m.lora_A.weight"] == pytest.approx(dense.numpy())
    assert weights == pytest.approx([0.25 ** 2, 0.25 * 0.75, 0.75])
    with pytest.raises(ValueError, match="non-LoRA"):
        server_average.stack([(1.0, {"m.bias": torch.zeros(3)})])
    run(stream, "fedavg-replay", tmp_path / "run")
    blend = average_run(tmp_path / "run", stream[0], 0.5, tmp_path / "b", rule="--blend")
    assert blend["blend"] == 0.5 and blend["window"] is None
