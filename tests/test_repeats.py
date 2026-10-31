"""Re-runs per deploy (deploy.py / generate.py --repeats; repeatability.default_repeats).

On simulators where an identical re-run does not repeat (repeatability.py: RoboLab on Isaac Lab), one run per
policy cannot call a winner, so a deploy runs the evaluation twice by default (a new RUN_ID each, same stack,
seed and start states) and `results --compare` finds two runs of one fingerprint.

tests/fixtures/robolab_cs_two_runs.{csv,json} are MEASURED (2026-10-31, ap-northeast-1, g6e.2xlarge): RoboLab
12 tasks x 10 episodes with the "specific" instructions, pi0.5-DROID and Cosmos3, two runs per policy on two
different instances, per-episode outcomes from episode_results.jsonl. One run per policy said Cosmos3 better
(51 vs 27 of 120, p = 0.001); the second runs (48 and 39) did not repeat it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import compare  # noqa: E402
import generate  # noqa: E402
import repeatability  # noqa: E402
import results  # noqa: E402
from targets import VLA_TARGETS  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
CS_CSV = FIX / "robolab_cs_two_runs.csv"
CS_EPS = FIX / "robolab_cs_two_runs.json"


def _render(vla, out_dir, *extra):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return proc, (out_dir / f"{vla}.sh").read_text() if proc.returncode == 0 else ""


def test_default_is_two_runs_only_where_reruns_do_not_repeat():
    for t in VLA_TARGETS:
        n = repeatability.default_repeats(t)
        if t in repeatability.REPEATS_SUPPORTED:
            assert repeatability.for_target(t)["repeatable"] == "no" and n == 2, t
        else:
            assert n == 1, t
    # every target that supports repeats has a RoboLab-style template that loops over them
    assert set(repeatability.REPEATS_SUPPORTED) <= set(generate.ROBOLAB_TARGETS)


def test_resolve_repeats():
    assert repeatability.resolve_repeats("robolab-pi05", "auto")[0] == 2
    n, why = repeatability.resolve_repeats("robolab-pi05", "1")
    assert n == 1 and "needs re-run" in why
    assert repeatability.resolve_repeats("robolab-cosmos3", 3)[0] == 3
    assert repeatability.resolve_repeats("smolvla", "auto") == (1, "LIBERO (MuJoCo, CPU physics): repeatable = yes")
    n, why = repeatability.resolve_repeats("openarm-isaac", "auto")
    assert n == 1 and "deploy it again" in why
    for target, bad in (("smolvla", "2"), ("openarm-isaac", "2"), ("robolab-pi05", "0"),
                        ("robolab-pi05", str(repeatability.MAX_REPEATS + 1)), ("robolab-pi05", "two")):
        with pytest.raises(ValueError):
            repeatability.resolve_repeats(target, bad)


def test_plan_line_and_fit():
    line = repeatability.plan_line("robolab-pi05", 2, 60)
    assert line.startswith("2 run(s) x 60 episodes = 120 episodes, ~2.1-2.2 instance-h")
    assert repeatability.plan_line("smolvla", 1, 50) == "1 run(s) x 50 episodes = 50 episodes"
    assert repeatability.plan_line("smolvla", 1, None) is None
    sub = generate.robolab_plan(generate._load_merged_config(ROOT / "simulator-config.yaml", "robolab-pi05"),
                                "robolab-pi05")
    assert sub == dict(episodes=60, shards=1, eval_timeout_min=300, deploy_timeout_min=480)
    assert repeatability.check_fit("robolab-pi05", 2, sub) is None
    assert repeatability.check_fit("robolab-pi05", 1, {"shards": 8}) is None
    assert "8 shards" in repeatability.check_fit("robolab-pi05", 2, dict(sub, shards=8))
    # 3 runs of the full 120-task protocol on one shard would not fit 8 h
    assert "deploy limit" in repeatability.check_fit("robolab-pi05", 3, dict(sub, episodes=1200))


def test_robolab_userdata_loops_over_repeats(tmp_path):
    proc, text = _render("robolab-pi05", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "Runs:   2 per deploy" in proc.stdout
    assert "REPEATS=2" in text and 'for REP in $(seq 1 "$REPEATS"); do' in text
    assert 'PC_REPEAT_INDEX="${REP}/${REPEATS}"' in text
    assert "  repeats: 2" in text
    # a new RUN_ID per repeat, the previous run's output cleared, servers removed once after the loop
    loop = text[text.index('for REP in $(seq 1 "$REPEATS")'):text.index("# ── [9/9]")]
    assert 'RUN_ID=$(date ' in loop and 'rm -rf "${OUT_DIR}/shard-${_s}/run"' in loop
    assert loop.index("\ndone\n") < loop.index('docker rm -f "policy-shard-${_s}"')
    # the first run keeps the old failure path (cfn FAILURE); a failed repeat is reported, not fatal
    assert 'if [ "$REP" -eq 1 ]; then' in loop and 'REPEAT_FAILED+=("${RUN_ID}")' in loop
    r = subprocess.run(["bash", "-n", str(tmp_path / "robolab-pi05.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_repeats_one_turns_it_off(tmp_path):
    proc, text = _render("robolab-cosmos3", tmp_path, "--repeats", "1")
    assert proc.returncode == 0, proc.stderr
    assert "REPEATS=1" in text and "  repeats: 1" in text
    assert "[warn] one run per deploy" in proc.stderr and "needs re-run" in proc.stderr


def test_repeats_refused_where_the_template_runs_once(tmp_path):
    proc, _ = _render("smolvla", tmp_path, "--repeats", "2")
    assert proc.returncode == 2 and "runs once per deploy" in proc.stderr
    proc, text = _render("smolvla", tmp_path)
    assert proc.returncode == 0 and "  repeats: 1" in text and "Runs:" not in proc.stdout


def test_deploy_refuses_before_any_aws_call():
    proc = subprocess.run([sys.executable, str(ROOT / "deploy.py"), "--vla", "smolvla", "--repeats", "2",
                           "--email", "someone@example.org", "--region", "us-east-1"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "runs once per deploy" in proc.stderr
    assert "[1/2]" not in proc.stdout


def test_card_records_repeats(tmp_path):
    static = generate.protocol_static({"model": {}, "repeat_plan": {"repeats": 2, "reason": "x'y"}},
                                      "robolab-pi05")["rerun_null"]
    assert (static["repeats"], static["repeats_reason"], static["runs_for_a_verdict"]) == (2, "xy", 2)
    assert generate.protocol_static({"model": {}}, "smolvla")["rerun_null"]["repeats"] == 1
    _, text = _render("robolab-pi05", tmp_path)
    assert re.search(r'echo "  repeat_index: \$\(pc_q "\$\{PC_REPEAT_INDEX:-1\}"\)"', text)


def test_measured_two_runs_are_unstable():
    """T4 (measured): one run per policy says B>A (51 vs 27, p = 0.001); with the second runs the tool says unstable."""
    a = compare.attach_episodes(compare.cells_from_csv(CS_CSV, "robolab-pi05"), CS_EPS, "robolab-pi05")
    b = compare.attach_episodes(compare.cells_from_csv(CS_CSV, "robolab-cosmos3"), CS_EPS, "robolab-cosmos3")
    (row,) = compare.compare(a, b, repeat=("no", "no"))
    assert (row["k_a"], row["k_b"]) == (39, 48)  # the latest run of each side is the cell
    assert row["verdict"] == "unstable" and row["pair_verdicts"] == {"B>A": 2, "tie": 2}
    assert (row["rerun_a"]["changed"], row["rerun_b"]["changed"]) == (28, 13)
    assert compare.fisher_two_sided(27, 120, 51, 120) < 0.002  # what the first runs alone would have said
    first = {k: dict(v, k=v["runs"][0]["k"], n=v["runs"][0]["n"]) for k, v in a.items()}
    first_b = {k: dict(v, k=v["runs"][0]["k"], n=v["runs"][0]["n"]) for k, v in b.items()}
    for side in (first, first_b):
        for v in side.values():
            v.pop("runs"), v.pop("episodes", None)
    (one,) = compare.compare(first, first_b, repeat=("yes", "yes"))
    assert one["verdict"] == "B>A"


def test_measured_two_runs_cli(capsys):
    rc = results.main(["--compare", "robolab-pi05", "robolab-cosmos3", "--from-csv", str(CS_CSV),
                       "--episodes", str(CS_EPS)])
    text = capsys.readouterr().out
    assert rc == 0 and "unstable 1" in text and "28 of 120 episodes changed (23%)" in text
