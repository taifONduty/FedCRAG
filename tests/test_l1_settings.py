"""l1_settings.py: the rules of registration section 15 that fix the LoTTE block."""
import l1_settings


def test_lambda_rule_takes_the_best_earlier_guard_score_and_the_smaller_on_a_tie():
    assert l1_settings.choose_lambda({0.1: 0.5, 0.25: 0.6, 2.0: 0.6}) == 0.25
    assert l1_settings.choose_lambda({0.1: 0.7, 0.25: 0.6, 2.0: 0.6}) == 0.1
    record = {"experiences_per_client": 2, "clients": ["0"], "order": {"0": [1, 0]},
              "matrix": {"1": {"0": {"1": {"guard": {"ndcg@10": 0.4}},
                                     "0": {"guard": {"ndcg@10": 0.9}}}}}}
    assert l1_settings.earlier_guard_ndcg(record) == 0.4
