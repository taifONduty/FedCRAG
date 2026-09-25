"""memory.py: one retained-query budget per client, refilled deterministically."""
import pytest

from memory import ReplayMemory

PAST = {"e0": [f"q{i}" for i in range(50)], "e1": [f"r{i}" for i in range(50)]}


def test_refill_takes_an_equal_share_from_each_earlier_experience():
    memory = ReplayMemory(budget=10, seed=7)
    memory.refill(PAST)
    ids = memory.ids
    assert len(ids) == 10 and len(set(ids)) == 10
    assert sum(q.startswith("q") for q in ids) == 5


def test_refill_is_deterministic_and_seed_dependent():
    first = ReplayMemory(budget=10, seed=7); first.refill(PAST)
    again = ReplayMemory(budget=10, seed=7); again.refill(PAST)
    other = ReplayMemory(budget=10, seed=8); other.refill(PAST)
    assert first.ids == again.ids and first.ids != other.ids


def test_a_small_experience_yields_its_share_to_the_others():
    memory = ReplayMemory(budget=10, seed=1)
    memory.refill({"e0": ["a", "b"], "e1": [f"r{i}" for i in range(50)]})
    assert set(memory.ids) >= {"a", "b"} and len(memory.ids) == 10


def test_guard_queries_consulted_online_count_toward_the_budget():
    memory = ReplayMemory(budget=10, seed=1)
    memory.reserve([f"g{i}" for i in range(4)])
    memory.refill(PAST)
    assert len(memory.ids) == 6 and memory.used == 10
    with pytest.raises(ValueError, match="budget"):
        memory.reserve([f"h{i}" for i in range(7)])


def test_record_round_trips():
    memory = ReplayMemory(budget=10, seed=7)
    memory.reserve(["g0"])
    memory.refill(PAST)
    restored = ReplayMemory.from_record(memory.record())
    assert restored.ids == memory.ids and restored.used == memory.used
    assert restored.record() == memory.record()


def test_guard_redraw_replaces_the_guard_and_leaves_replay_the_rest_of_the_budget():
    memory = ReplayMemory(budget=10, seed=7)
    memory.redraw_guard({"e0": [f"g{i}" for i in range(20)]}, 4)
    first = memory.reserved
    memory.redraw_guard({"e0": [f"g{i}" for i in range(20)], "e1": [f"h{i}" for i in range(20)]}, 4)
    memory.refill(PAST)
    assert sum(q.startswith("g") for q in memory.reserved) == 2 and memory.reserved != first
    assert len(memory.ids) == 6 and memory.used == 10
    plain = ReplayMemory(budget=6, seed=7)
    plain.refill(PAST)
    assert memory.ids == plain.ids
