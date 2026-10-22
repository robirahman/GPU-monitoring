"""Draw mechanics on a SYNTHETIC register only; the real pool is drawn once, after pre-registration."""
import pytest

from h200_campaign.partition.draw import run

REG = {"candidates": {
    "training_capable": {"a": {"stratum": "llm_decoder"}, "b": {"stratum": "llm_decoder"},
                         "c": {"stratum": "llm_decoder"}, "d": {"stratum": "vision"}, "e": {"stratum": "graph"}},
    "inference_only": {"p": {}, "q": {}, "r": {}},
    "other": {"x": {}, "y": {}, "z": {}}}}
CONS = {"seed": 123, "test_counts": {"training_capable": 3, "inference_only": 2, "other": 2},
        "test_training_must_include_strata_any_of": [["llm_decoder"], ["vision", "seq2seq", "graph"]],
        "test_training_min_distinct_strata": 2}
FEAS = {k: {"feasible": True} for k in "abcdepqrxyz"}


def test_deterministic_and_valid():
    r1, r2 = run(REG, CONS, FEAS), run(REG, CONS, FEAS)
    assert r1["test"] == r2["test"]
    t = r1["test"]["training_capable"]
    assert len(t) == 3 and any(x in "abc" for x in t) and any(x in "de" for x in t)
    test = {x for v in r1["test"].values() for x in v}
    for pool, rest in r1["calibration_additions"].items():
        assert not test & set(rest)


def test_rejections_are_logged():
    # with only one non-LLM lineage feasible, many attempts draw three LLMs and must be rejected and logged
    feas = dict(FEAS, e={"feasible": False})
    r = run(REG, CONS, feas)
    assert all(a["violations"] for a in r["attempts"][:-1])
    assert not r["attempts"][-1]["violations"]


def test_infeasible_pool_stops_instead_of_relaxing():
    feas = dict(FEAS, x={"feasible": False}, y={"feasible": False})
    with pytest.raises(SystemExit):
        run(REG, CONS, feas)
