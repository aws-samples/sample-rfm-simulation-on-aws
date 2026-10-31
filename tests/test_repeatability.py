"""Re-run null per simulator (repeatability.py) and the rule `results --compare` applies with it.

tests/fixtures/rerun_cells.csv holds measured success counts of two simulators that do not repeat an identical
run: NVIDIA RoboLab on Isaac Lab (pi0.5-DROID vs Cosmos3, 12 tasks x 10 episodes; condition C0 was run twice per
policy, C1 and Cs once) and Isaac Lab Arena (same seed, same machine, two runs). tests/fixtures/rerun_episodes.json
holds SYNTHETIC per-episode outcomes built to match the measured counts and the measured number of episodes that
changed between the two runs of a policy (25 of 120, 19 of 120, 13, 9 and 46 of 100); between-policy pairings in
it are not measured and are not asserted.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import compare  # noqa: E402
import repeatability  # noqa: E402
import results  # noqa: E402
from targets import VLA_TARGETS  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
CSV = FIX / "rerun_cells.csv"
EPS = FIX / "rerun_episodes.json"


def _cells(model):
    return compare.attach_episodes(compare.cells_from_csv(CSV, model), EPS, model)


def test_every_target_has_a_simulator():
    assert set(repeatability.TARGET_SIMULATOR) == set(VLA_TARGETS)
    for sim in repeatability.TARGET_SIMULATOR.values():
        assert repeatability.SIMULATORS[sim]["repeatable"] in repeatability.REPEATABLE


def test_measured_classes():
    assert repeatability.for_target("smolvla")["repeatable"] == "yes"
    assert repeatability.for_target("robolab-pi05")["repeatable"] == "no"
    assert repeatability.for_target("openarm-isaac")["repeatable"] == "no"
    assert repeatability.for_target("lingbot-vla2")["repeatable"] == "unmeasured"
    assert repeatability.for_target("not-a-target")["repeatable"] == "unmeasured"
    # a stack name is matched on the longest target it starts with
    assert repeatability.for_target("robolab-pi05-demo")["simulator"] == "robolab"
    assert repeatability.for_target("pi-libero-10")["simulator"] == "libero"
    assert repeatability.worst("yes", "no") == "no" and repeatability.worst("yes", "unmeasured") == "unmeasured"


def test_csv_runs_and_spread():
    cells = _cells("robolab-pi05")
    c0 = cells[("robolab-12", "C0")]
    assert (c0["k"], c0["run"], len(c0["runs"])) == (54, "2026-10-31-0248", 2)  # the latest run is the cell
    sp = compare.rerun_spread(c0["runs"])
    assert (sp["k_min"], sp["k_max"], sp["n"], sp["paired"], sp["changed"]) == (45, 54, 120, 120, 25)
    assert "runs" not in cells[("robolab-12", "Cs")]
    sp = compare.rerun_spread(_cells("arena-pi05")[("arena-microwave", "b1-s1000")]["runs"])
    assert (sp["k_min"], sp["k_max"], sp["changed"]) == (69, 77, 46)


def test_robolab_one_run_needs_rerun_and_two_runs_are_unstable():
    rows = {r["condition_a"]: r for r in compare.compare(_cells("robolab-pi05"), _cells("robolab-cosmos3"),
                                                          repeat=("no", "no"))}
    # Cs: the instruction-wording result is one run per policy -> not a verdict yet
    assert rows["Cs"]["verdict"] == "needs re-run" and rows["Cs"]["single_run_verdict"] == "B>A"
    assert rows["Cs"]["needs_rerun"] == "AB"
    assert rows["C1"]["verdict"] == "needs re-run" and rows["C1"]["single_run_verdict"] == "A>B"
    # C0 ran twice per policy: 45/54 vs 41/36 -> tie on three pairs of runs, A>B on one (54 vs 36, p = 0.023)
    c0 = rows["C0"]
    assert c0["pair_verdicts"] == {"tie": 3, "A>B": 1}
    assert c0["verdict"] == "unstable" and c0["one_run_verdict"] == "A>B"
    assert compare.fisher_two_sided(54, 120, 36, 120) == pytest.approx(0.0232, abs=5e-4)
    assert compare.fisher_two_sided(45, 120, 41, 120) == pytest.approx(0.6864, abs=5e-4)


def test_arena_two_runs_agree_keeps_the_verdict():
    rows = compare.compare(_cells("arena-pi05"), _cells("arena-smolvla"), repeat=("no", "no"))
    (r,) = rows
    assert r["verdict"] == "tie" and r["pair_verdicts"] == {"tie": 4}
    assert (r["rerun_a"]["changed"], r["rerun_b"]["changed"]) == (13, 9)


def test_repeatable_and_unmeasured_keep_one_run_verdicts():
    a, b = _cells("robolab-pi05"), _cells("robolab-cosmos3")
    plain = {r["condition_a"]: r["verdict"] for r in compare.compare(a, b)}
    for cls in ("yes", "unmeasured"):
        got = {r["condition_a"]: r for r in compare.compare(a, b, repeat=(cls, cls))}
        assert {k: r["verdict"] for k, r in got.items()} == plain
        assert all(r["rerun_class"] == cls for r in got.values())
    # one side GPU physics with one run is enough to hold the verdict
    got = {r["condition_a"]: r for r in compare.compare(a, b, repeat=("yes", "no"))}
    assert got["Cs"]["verdict"] == "needs re-run" and got["Cs"]["needs_rerun"] == "B"


def test_study_verdicts_unchanged_with_the_rule_on():
    """LIBERO study table (test_compare.py): the 13 official verdicts survive the rule (LIBERO repeats)."""
    expected = json.loads((FIX / "compare_expected.json").read_text())["official_pairs"]
    study = FIX / "study_cells.csv"
    got = {}
    for r in expected:
        key = (r["a"], r["b"])
        if key not in got:
            got[key] = {x["suite"]: x for x in compare.compare(compare.cells_from_csv(study, r["a"]),
                                                              compare.cells_from_csv(study, r["b"]), "S0", "S0",
                                                              repeat=("yes", "yes"))}
        assert got[key][r["suite"]]["verdict"] == r["verdict"]


def test_cli_prints_the_rule(capsys, tmp_path):
    out_json = tmp_path / "c.json"
    rc = results.main(["--compare", "robolab-pi05", "robolab-cosmos3", "--from-csv", str(CSV),
                       "--episodes", str(EPS), "--json", str(out_json)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "needs re-run 2" in text and "unstable 1" in text
    assert "25 of 120 episodes changed (21%)" in text and "run pairs A>B 1 · tie 3" in text
    assert "re-run null, A and B: RoboLab (Isaac Lab, GPU physics) — no" in text
    data = json.loads(out_json.read_text())
    assert data["simulators"]["A and B"]["repeatable"] == "no"
    rc = results.main(["--compare", "robolab-pi05", "robolab-cosmos3", "--from-csv", str(CSV), "--assume-repeatable"])
    text = capsys.readouterr().out
    assert rc == 0 and "needs re-run" not in text and "re-run null, " not in text


def test_cli_simulator_column_overrides_the_target_table(capsys):
    rc = results.main(["--compare", "arena-pi05", "arena-smolvla", "--from-csv", str(CSV), "--episodes", str(EPS)])
    text = capsys.readouterr().out
    assert rc == 0 and "Isaac Lab (GPU physics) — no" in text and "run pairs tie 4" in text
