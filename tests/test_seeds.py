"""Shared seed (seeds.py): one seed per comparison, applied where the target reads it, recorded where not."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import seeds  # noqa: E402
from targets import VLA_TARGETS  # noqa: E402


def test_every_target_has_a_seed_entry():
    assert set(VLA_TARGETS) <= set(seeds.SEED_CONTROL)
    for t, c in seeds.SEED_CONTROL.items():
        assert c["where"] in ("tasks", "eval", None), t
        assert c["env"] in ("yes", "no", "fixed") and c["policy"] in ("yes", "no"), t
        if c["where"] is None:
            assert c.get("reason"), f"{t}: a target the tool cannot seed must say why"


@pytest.mark.parametrize("value,expect", [(None, "model"), ("model", "model"), ("AUTO", "auto"), ("0", 0),
                                          ("2147483647", 2 ** 31 - 1), (42, 42)])
def test_parse(value, expect):
    assert seeds.parse(value) == expect


@pytest.mark.parametrize("bad", ["-1", "2147483648", "1e3", "seed", "", "12 3"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        seeds.parse(bad)


def test_auto_draws_once_in_range():
    a, b = seeds.resolve("auto"), seeds.resolve("auto")
    assert a["mode"] == "drawn" and 0 <= a["value"] < 2 ** 31 and a["drawn_utc"].endswith("Z")
    assert a["value"] != b["value"] or a["value"] == b["value"]  # draws are independent (no state kept)


def test_same_seed_reaches_every_task_of_every_target():
    plan = seeds.resolve("424242")
    cfg = {"tasks": [{"suite": "libero_spatial", "seed": 1000}, {"suite": "libero_object"}]}
    for t in ("smolvla", "xvla", "pi", "molmoact2"):
        out = seeds.apply(cfg, t, plan)
        assert [x["seed"] for x in out["tasks"]] == [424242, 424242]
        assert out["seed_plan"]["status"] == "applied"
    assert cfg["tasks"][0]["seed"] == 1000  # the input config is not changed


def test_tasks_by_suite_and_eval_seed():
    plan = seeds.resolve(9)
    out = seeds.apply({"tasks": {"10": [{"seed": 7}], "spatial": [{}]}}, "openvla-oft", plan)
    assert out["tasks"] == {"10": [{"seed": 9}], "spatial": [{"seed": 9}]}
    out = seeds.apply({"eval": {"seed": 0}}, "robolab-pi05", plan)
    assert out["eval"]["seed"] == 9


def test_not_controllable_is_recorded_not_faked():
    out = seeds.apply({"tasks": [{"seed": 1}]}, "gr00t", seeds.resolve(9))
    assert out["tasks"] == [{"seed": 1}]
    sp = out["seed_plan"]
    assert sp["status"] == "not-controllable" and sp["value"] == 9 and "4b1dca9" in sp["reason"]


def test_model_mode_changes_nothing():
    cfg = {"tasks": [{"seed": 1000}]}
    out = seeds.apply(cfg, "smolvla", seeds.resolve("model"))
    assert out["tasks"] == cfg["tasks"] and out["seed_plan"]["status"] == "model"


def test_a_handed_on_draw_stays_drawn():
    p = seeds.resolve("17", drawn_utc="2026-10-31T05:03:57Z")
    assert (p["mode"], p["value"], p["drawn_utc"]) == ("drawn", 17, "2026-10-31T05:03:57Z")
