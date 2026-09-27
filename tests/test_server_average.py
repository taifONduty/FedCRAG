"""server_average.py: window 1 reproduces a run; a wider window deploys the average of the
last rounds' global models."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server_average  # noqa: E402
from aggregation_schemes import state_dict_sha256  # noqa: E402
from test_continual_driver import run, stream  # noqa: E402,F401


def average_run(source, root, window, out):
    server_average.main(["--source", str(source), "--window", str(window),
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
