"""Offline tests for compare.py (`vlasim results --compare A B`).

The fixture tests/fixtures/study_cells.csv is the public study table of the AMLC 2026 study
(github.com/byongwu/pai-same-policy-different-numbers, results/cells.csv, MIT-0). Expected p-values in
tests/fixtures/compare_expected.json were computed with SciPy / statsmodels, so these tests pin the pure-Python
statistics to the reference implementations without adding SciPy as a dependency.
"""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import compare  # noqa: E402
import results  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
CSV = FIX / "study_cells.csv"
EXPECTED = json.loads((FIX / "compare_expected.json").read_text())


def test_fisher_matches_reference_on_every_official_pair():
    for r in EXPECTED["official_pairs"]:
        assert compare.fisher_two_sided(r["k_a"], r["n_a"], r["k_b"], r["n_b"]) == pytest.approx(r["p"], abs=1e-12)


def test_official_verdicts_match_reference():
    """60 policy pairs x suites under the official scripts; 13 resolved (BH over each pair's four suites)."""
    models = sorted({r["a"] for r in EXPECTED["official_pairs"]} | {r["b"] for r in EXPECTED["official_pairs"]})
    cells = {m: compare.cells_from_csv(CSV, m) for m in models}
    got = {}
    for r in EXPECTED["official_pairs"]:
        key = (r["a"], r["b"])
        if key not in got:
            got[key] = {x["suite"]: x for x in compare.compare(cells[r["a"]], cells[r["b"]], "S0", "S0")}
        row = got[key][r["suite"]]
        assert row["q"] == pytest.approx(r["q"], abs=1e-12)
        assert row["verdict"] == r["verdict"]
    assert len(EXPECTED["official_pairs"]) == 60
    assert sum(r["verdict"] != "tie" for r in EXPECTED["official_pairs"]) == 13


@pytest.mark.parametrize("case", EXPECTED["mcnemar"])
def test_mcnemar_matches_reference(case):
    assert compare.mcnemar_exact(case["only_a"], case["only_b"]) == pytest.approx(case["p"], abs=1e-12)


def test_fisher_and_binom_edge_cases():
    assert compare.fisher_two_sided(50, 50, 50, 50) == 1.0
    assert compare.fisher_two_sided(0, 0, 3, 5) == 1.0
    assert compare.mcnemar_exact(0, 0) == 1.0
    assert compare.binom_two_sided(0, 10) == pytest.approx(2 * 0.5 ** 10)


def test_bh_is_monotone_and_bounded():
    random.seed(3)
    ps = [random.random() ** 3 for _ in range(40)]
    q = compare.bh(ps)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    assert all(q[order[i]] <= q[order[i + 1]] + 1e-15 for i in range(len(ps) - 1))
    assert all(p - 1e-12 <= x <= 1 for p, x in zip(ps, q))


def test_same_policy_two_settings_matches_on_suite():
    cells = compare.cells_from_csv(CSV, "oft")
    rows = compare.compare(cells, cells, "S0", "S3b")
    assert {r["suite"] for r in rows} == {"libero_10", "libero_goal", "libero_object", "libero_spatial"}
    assert all(r["condition_a"] == "S0" and r["condition_b"] == "S3b" for r in rows)
    assert all(r["verdict"] == "A>B" for r in rows)  # OpenVLA-OFT running the chunk tail collapses on every suite


def test_paired_episodes_by_task_occurrence():
    a = [dict(task="t1", ep=i, success=True) for i in range(5)] + [dict(task="t2", ep=i, success=i < 2) for i in range(5)]
    b = [dict(task="t1", ep=i, success=i != 0) for i in range(5)] + [dict(task="t2", ep=i, success=True) for i in range(5)]
    assert compare.pair_episodes(a, b) == (10, 1, 3)


def test_episodes_from_video_keys():
    keys = ["r/task-0/videos/episode=3--success=True--task=open_the_drawer.mp4",
            "r/task-0/videos/episode=4--success=False--task=open_the_drawer.mp4", "r/task-0/videos/notes.txt"]
    assert compare.episodes_from_keys(keys) == [dict(ep=3, success=True, task="open_the_drawer"),
                                                dict(ep=4, success=False, task="open_the_drawer")]


def test_rerun_null_compares_consecutive_runs():
    runs = {("libero_10", "S0", "fp"): [("2026-10-02-0000", 40, 50), ("2026-10-01-0000", 41, 50),
                                        ("2026-10-03-0000", 20, 50)]}
    null = compare.rerun_null(runs)
    assert [(x["run_1"], x["run_2"]) for x in null] == [("2026-10-01-0000", "2026-10-02-0000"),
                                                        ("2026-10-02-0000", "2026-10-03-0000")]
    assert [x["differs"] for x in null] == [False, True]


def test_cli_offline_compare(tmp_path, capsys):
    out = tmp_path / "c.json"
    assert results.main(["--compare", "oft:S0", "oft:S3b", "--from-csv", str(CSV), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "4 rows: A>B 4" in text
    data = json.loads(out.read_text())
    assert data["a"] == "oft:S0" and len(data["rows"]) == 4


def test_cli_no_common_rows(capsys):
    assert results.main(["--compare", "oft:S0", "nosuchmodel", "--from-csv", str(CSV)]) == 1
