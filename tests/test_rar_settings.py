"""rar_settings.py: the rule of registration section 17."""
import rar_settings


def test_the_best_eligible_configuration_is_chosen_only_if_it_beats_replay():
    baseline = (0.50, 0.100)
    candidates = {(0.5, "random"): (0.52, 0.099), (2.0, "random"): (0.53, 0.090),
                  (0.5, "fragile"): (0.52, 0.101), (2.0, "fragile"): (0.49, 0.100)}
    assert rar_settings.choose(candidates, baseline) == (0.5, "random")
    assert rar_settings.choose({(0.5, "random"): (0.49, 0.100)}, baseline) is None
    assert rar_settings.choose({(2.0, "random"): (0.60, 0.050)}, baseline) is None
