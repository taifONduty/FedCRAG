"""acceptance.py: arm F's check pool and step rule."""
import pytest

import acceptance


def test_the_first_step_within_the_threshold_is_kept_and_none_keeps_the_broadcast():
    step, tried = acceptance.choose_step(lambda s: {1.0: 0.03, 0.5: 0.008, 0.25: 0.0}[s])
    assert step == 0.5 and [s for s, _ in tried] == [1.0, 0.5]
    assert acceptance.rule_holds(step, tried)
    step, tried = acceptance.choose_step(lambda s: 0.02)
    assert step == 0.0 and len(tried) == 3 and acceptance.rule_holds(step, tried)
    assert not acceptance.rule_holds(1.0, [[1.0, 0.03], [0.5, 0.008]])
    assert not acceptance.rule_holds(0.0, [[1.0, 0.03]])


def test_the_check_pool_holds_the_guard_passages_and_refuses_outside_ones():
    corpus = [str(i) for i in range(50)]
    pool = acceptance.check_pool(corpus, {"q": {"3": 1}}, {"q": ["4"]}, seed=[1])
    assert {"3", "4"} <= set(pool) and pool == sorted(set(pool))
    with pytest.raises(ValueError, match="outside the corpus"):
        acceptance.check_pool(corpus, {"q": {"99": 1}}, {"q": []}, seed=[1])
