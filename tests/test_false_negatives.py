"""false_negatives.py: every earlier test query at every later evaluation is counted once,
flagged or not, in the run and in the control."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import false_negatives  # noqa: E402
from test_continual_driver import run, stream  # noqa: E402,F401


def test_every_earlier_test_query_is_counted_once_in_both_runs(stream, tmp_path, monkeypatch):
    def fake_anchor(model, start, data, anchored, *args, **kwargs):
        new = {k: v.clone() for k, v in start.items()}
        new["lora_B"] = new["lora_B"] + 0.1
        return new, len(data["train_q"]), 3

    monkeypatch.setattr(false_negatives.driver.rank, "client_train_anchor", fake_anchor)
    hard, _, _ = run(stream, "fedavg-replay-anchor", tmp_path / "hard",
                     ["--anchor_k", "3", "--anchor_loss", "hard"])
    run(stream, "fedavg-replay", tmp_path / "replay")
    out = tmp_path / "fn.json"
    false_negatives.main(["--run", str(tmp_path / "hard"), "--control", str(tmp_path / "replay"),
                          "--data_root", str(stream[0]), "--out", str(out)])
    result = json.loads(out.read_text())
    tests = sum(len(hard["references"][c][str(hard["order"][c][0])]["scores"]["test"]
                    ["per_query"]["ndcg@10"]) for c in hard["clients"])
    for which in ("run", "control"):
        assert result["flagged"][which]["n"] + result["unflagged"][which]["n"] == tests
