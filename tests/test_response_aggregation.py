"""Pure pieces of the response-maxmin arm (design note 2026-09-08, sections 4.1 and 4.3)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import response_aggregation as ra  # noqa: E402


def test_dev_split_is_deterministic_disjoint_and_sized():
    qids = [f"q{i}" for i in range(40)]
    train, dev = ra.dev_split(qids, fraction=0.1, seed=123, slice_name="fiqa", min_dev=2)
    train2, dev2 = ra.dev_split(list(reversed(qids)), fraction=0.1, seed=123,
                                slice_name="fiqa", min_dev=2)
    assert (train, dev) == (train2, dev2)          # input order does not matter
    assert len(dev) == 4 and len(train) == 36
    assert not set(train) & set(dev) and set(train) | set(dev) == set(qids)
    assert dev == sorted(dev) and train == sorted(train)
    _, dev_other_seed = ra.dev_split(qids, 0.1, 124, "fiqa", 2)
    _, dev_other_slice = ra.dev_split(qids, 0.1, 123, "nfcorpus", 2)
    assert dev != dev_other_seed and dev != dev_other_slice


def test_dev_split_respects_minimum_and_half_cap():
    qids = [f"q{i}" for i in range(40)]
    _, dev = ra.dev_split(qids, fraction=0.01, seed=1, slice_name="s", min_dev=5)
    assert len(dev) == 5                             # the minimum wins over the fraction
    _, dev = ra.dev_split(qids, fraction=0.9, seed=1, slice_name="s", min_dev=5)
    assert len(dev) == 20                            # never more than half
    assert ra.dev_split(["q0"], 0.5, 1, "s", 1) == (["q0"], [])


def test_simplex_lattice_count_and_sum():
    pts = ra.simplex_lattice(4, 0.125)
    assert len(pts) == 165                           # C(8 + 3, 3)
    assert all(abs(sum(p) - 1) < 1e-12 and min(p) >= 0 for p in pts)
    assert len(set(pts)) == 165
    with pytest.raises(AssertionError):
        ra.simplex_lattice(3, 0.3)                   # 1/0.3 is not an integer


def test_candidate_grid_keeps_fixed_points_first_and_dedups():
    fixed = {"uniform": [1 / 3] * 3, "solo_a": [1.0, 0.0, 0.0]}
    grid = ra.candidate_grid(3, fixed, lattice_step=0.5, scales=(1.0, 2.0))
    names = list(grid)
    assert names[:2] == ["uniform", "solo_a"]
    # the lattice at step 0.5 has 6 points; (1,0,0) at scale 1 duplicates solo_a
    assert sum(n.startswith("lat") for n in names) == 6 * 2 - 1
    assert grid["solo_a"] == [1.0, 0.0, 0.0]
    assert all(len(v) == 3 and min(v) >= 0 for v in grid.values())
    assert any(abs(sum(v) - 2.0) < 1e-12 for v in grid.values())


def test_predict_embeddings_is_exact_at_zero_and_at_vertices():
    rng = np.random.default_rng(0)
    base = ra.normalise_rows(rng.normal(size=(7, 5)))
    solos = [ra.normalise_rows(rng.normal(size=(7, 5))) for _ in range(3)]
    responses = [s - base for s in solos]
    assert np.allclose(ra.predict_embeddings(base, responses, [0, 0, 0]), base)
    for j in range(3):
        v = [0.0] * 3
        v[j] = 1.0
        assert np.allclose(ra.predict_embeddings(base, responses, v), solos[j], atol=1e-12)
    mixed = ra.predict_embeddings(base, responses, [0.5, 0.25, 0.0])
    assert np.allclose(np.linalg.norm(mixed, axis=1), 1.0)


def test_ndcg10_matches_pytrec_eval_including_identical_id_rule():
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = np.random.default_rng(1)
    cids = [f"d{i}" for i in range(50)] + ["q3"]        # a document named like query q3
    qids = [f"q{i}" for i in range(6)]
    sims = rng.normal(size=(6, 51))
    sims[3, 50] = 10.0                                  # q3's own document would rank first
    qrels = {q: {f"d{int(rng.integers(0, 50))}": int(g) for g in rng.integers(1, 4, size=3)}
             for q in qids}
    qrels["q5"] = {}                                    # no relevant documents
    ours = ra.ndcg10(sims, cids, qids, qrels)
    run = {}
    for i, q in enumerate(qids):
        order = np.argsort(-sims[i])
        run[q] = {cids[j]: float(sims[i, j]) for j in order[:60] if cids[j] != q}
    ev = pytrec_eval.RelevanceEvaluator({q: r for q, r in qrels.items() if r},
                                        {"ndcg_cut.10"})
    ref = ev.evaluate(run)
    for i, q in enumerate(qids):
        expected = ref[q]["ndcg_cut_10"] if q in ref else 0.0
        assert ours[i] == pytest.approx(expected, abs=1e-9), q


def test_rank_candidates_orders_by_worst_client_then_mean_and_applies_floor():
    current = [0.30, 0.20]
    pred = {"a": [0.33, 0.23],      # gains +0.03 / +0.03: min 0.03, mean 0.03
            "b": [0.36, 0.23],      # min 0.03, mean 0.045: ahead of a on the tie
            "c": [0.40, 0.21],      # min 0.01
            "d": [0.50, 0.19]}      # below a floor of 0.20 on client 2
    assert ra.rank_candidates(pred, current, floors=None, n_top=4) == ["b", "a", "c", "d"]
    assert ra.rank_candidates(pred, current, floors=[0.30, 0.20], n_top=2) == ["b", "a"]
    assert ra.rank_candidates(pred, current, floors=[0.0, 0.20], n_top=9) == ["b", "a", "c"]
    assert ra.rank_candidates(pred, current, floors=[0.0, 0.99], n_top=2) == []


def test_choose_applied_uses_measured_values_and_reports_min_gain():
    current = [0.30, 0.20]
    measured = {"a": [0.31, 0.25], "b": [0.36, 0.19]}
    name, gain = ra.choose_applied(measured, current, floors=None)
    assert name == "a" and gain == pytest.approx(0.01)
    name, gain = ra.choose_applied(measured, current, floors=[0.32, 0.0])
    assert name == "b" and gain == pytest.approx(-0.01)
    assert ra.choose_applied(measured, current, floors=[0.9, 0.9]) == (None, None)


def test_paired_lower_bound_is_below_the_mean_and_deterministic():
    rng = np.random.default_rng(2)
    diffs = rng.normal(0.02, 0.1, size=300)
    lo = ra.paired_lower_bound(diffs, alpha=0.05, n_boot=500, seed=0)
    assert lo < diffs.mean()
    assert lo == ra.paired_lower_bound(diffs, alpha=0.05, n_boot=500, seed=0)
    assert np.isnan(ra.paired_lower_bound([], alpha=0.05))


def test_response_config_tag_is_short_and_configuration_sensitive():
    cfg = dict(dev_fraction=0.1, dev_min=30, lattice_step=0.125, scales=[0.5, 1.0, 1.5],
               n_verify=2, floor="frozen", floor_delta=0.0, halvings=2)
    tag = ra.response_config_tag(cfg)
    assert len(tag) == 8 and tag == ra.response_config_tag(dict(cfg))
    assert tag != ra.response_config_tag({**cfg, "n_verify": 3})


def test_pessimistic_gain_is_mean_minus_one_standard_error():
    d = np.array([0.1, 0.3, -0.1, 0.2])
    assert ra.pessimistic_gain(d) == pytest.approx(d.mean() - d.std(ddof=1) / 2.0)
    assert ra.pessimistic_gain([0.25]) == 0.25
    assert np.isnan(ra.pessimistic_gain([]))


def test_magnitude_candidates_equalise_shares_and_drop_idle_clients():
    norms = [4.0, 1.0, 0.5, 0.0]            # the last client moved nowhere
    game = [0.4, 0.3, 0.3, 0.0]
    cands = ra.magnitude_candidates(norms, game, eq_scales=[1.0, 2.0], game_scales=[1.0])
    assert set(cands) == {"eq_x1", "eq_x2", "game_x1"}
    rbar = np.mean([4.0, 1.0, 0.5])
    # equal magnitude shares: v_k r_k equal for active clients, total rbar at scale 1
    shares = [v * r for v, r in zip(cands["eq_x1"], norms)]
    assert shares[:3] == pytest.approx([rbar / 3] * 3) and shares[3] == 0.0
    assert sum(v * r for v, r in zip(cands["eq_x2"], norms)) == pytest.approx(2 * rbar)
    # the game family mixes unit directions by w*: magnitude share k is w*_k
    gshares = [v * r for v, r in zip(cands["game_x1"], norms)]
    assert np.asarray(gshares[:3]) / rbar == pytest.approx([0.4, 0.3, 0.3])
    assert cands["game_x1"][3] == 0.0
    assert ra.magnitude_candidates([0.0, 0.0], [0.5, 0.5], [1.0], [1.0]) == {}


def test_uniform_subsets_enumerate_every_nonempty_subset():
    subs = ra.uniform_subsets(3)
    assert len(subs) == 7 and subs["sub_02"] == [0.5, 0.0, 0.5] and subs["sub_1"] == [0.0, 1.0, 0.0]


def test_greedy_soup_adds_a_vertex_only_when_every_client_improves():
    # client gains of the uniform soup over a set S: client 0 likes vertex 0 only
    def gains_of(v):
        kept = [j for j, x in enumerate(v) if x > 0]
        return [1.0 if kept == [0] else 0.5 if 0 in kept else 0.0,   # client 0 hurt by others
                0.2 * len(kept)]                                       # client 1 likes everyone
    v, kept = ra.greedy_soup([0, 1, 2], gains_of)
    assert kept == [0] and v == [1.0, 0.0, 0.0]
    v, kept = ra.greedy_soup([1, 2, 0], lambda v: [sum(v) * 0 + len([x for x in v if x > 0])] * 2)
    assert kept == [1, 2, 0] and v == pytest.approx([1 / 3] * 3)


def test_rank_candidates_uses_scores_when_given_and_floors_on_means():
    pred = {"a": [0.5, 0.5], "b": [0.6, 0.6], "c": [0.9, 0.1]}
    current = [0.4, 0.4]
    scores = {"a": [0.2, 0.2], "b": [0.05, 0.3], "c": [0.5, -0.1]}
    assert ra.rank_candidates(pred, current, None, 3) == ["b", "a", "c"]          # mean rule
    assert ra.rank_candidates(pred, current, None, 3, scores) == ["a", "b", "c"]  # pessimistic
    assert ra.rank_candidates(pred, current, [0.55, 0.0], 3, scores) == ["b", "c"]
    name, stat = ra.choose_applied(pred, current, None, scores)
    assert name == "a" and stat == 0.2


def test_response_config_tag_is_sensitive_to_the_v2_keys():
    cfg = dict(dev_fraction=0.1, dev_min=30, lattice_step=0.125, scales=[0.5, 1.0, 1.5],
               n_verify=2, floor="frozen", floor_delta=0.0, halvings=2)
    v1 = ra.response_config_tag(cfg)
    v2 = ra.response_config_tag({**cfg, "candidates": "compact", "select": "pessimistic",
                                 "eq_scales": [0.5, 1.0, 2.0], "game_scales": [1.0, 2.0],
                                 "model_pick": False})
    assert v1 != v2 and v2 != ra.response_config_tag({**cfg, "candidates": "compact",
                                                      "select": "mean", "eq_scales": [1.0],
                                                      "game_scales": [1.0], "model_pick": True})


def test_pareto_objective_prefers_all_positive_candidates_by_total_then_degrades_to_maxmin():
    pred = {"a": [0.5, 0.5], "b": [0.6, 0.6], "c": [0.9, 0.1], "d": [0.7, 0.7]}
    current = [0.4, 0.4]
    scores = {"a": [0.02, 0.02], "b": [0.05, 0.01], "c": [0.5, -0.1], "d": [0.09, 0.0]}
    # feasible (all >= 0): a total 0.04, b total 0.06, d total 0.09; c is not
    assert ra.rank_candidates(pred, current, None, 4, scores, "pareto") == ["d", "b", "a", "c"]
    assert ra.rank_candidates(pred, current, None, 4, scores) == ["a", "b", "d", "c"]  # max-min
    # nothing feasible: max-min order
    neg = {n: [x - 1.0 for x in v] for n, v in scores.items()}
    assert ra.rank_candidates(pred, current, None, 4, neg, "pareto") == ["a", "b", "d", "c"]
    assert ra.pareto_feasible([0.0, 0.1]) and not ra.pareto_feasible([-1e-9, 0.1]) and not ra.pareto_feasible([])
    with pytest.raises(ValueError):
        ra.rank_candidates(pred, current, None, 1, scores, "other")


def test_magnitude_candidates_top_m_equalise_the_largest_updates_only():
    norms = [4.0, 1.0, 0.5, 0.25]
    cands = ra.magnitude_candidates(norms, [0.25] * 4, eq_scales=[1.0], game_scales=[], eq_top=[2, 3, 4, 9])
    assert set(cands) == {"eq_x1", "eq2_x1", "eq3_x1"}          # m must be below the active count
    rbar = np.mean(norms)
    shares2 = [v * r for v, r in zip(cands["eq2_x1"], norms)]
    assert shares2 == pytest.approx([rbar / 2, rbar / 2, 0.0, 0.0])
    shares3 = [v * r for v, r in zip(cands["eq3_x1"], norms)]
    assert shares3 == pytest.approx([rbar / 3] * 3 + [0.0])


def test_parse_number_list_treats_none_and_empty_as_off():
    assert ra.parse_number_list("none", float) == [] and ra.parse_number_list("", float) == []
    assert ra.parse_number_list(" None ", int) == []
    assert ra.parse_number_list("0.5, 1.0,2", float) == [0.5, 1.0, 2.0]
    assert ra.parse_number_list("2,3", int) == [2, 3]
    with pytest.raises(ValueError):
        ra.parse_number_list("1.0,abc", float)
