"""Post-run judge wiring: deploy.py/destroy.py/vlasim.py --judge argument handling, run selection,
stack/bucket names shared with the CDK app, and the rendered flora-judge UserData.

The CLI checks run deploy.py/destroy.py as subprocesses and stop at argument validation, before any
AWS call, so no credentials are needed.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from targets import JUDGE_TARGETS  # noqa: E402

JUDGE_TS = (ROOT / "cdk" / "lib" / "judge-stack.ts").read_text()


def _cli(script, *args):
    return subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=60)


# --- argument validation (no AWS) -------------------------------------------------------------------

def test_deploy_run_needs_judge():
    pytest.importorskip("boto3")
    out = _cli("deploy.py", "--vla", "smolvla", "--run", "2026-10-20-0130", "--email", "a@example.com")
    assert out.returncode == 1 and "--run is only used with --judge" in out.stderr


def test_deploy_judge_rejects_bad_run_id():
    pytest.importorskip("boto3")
    out = _cli("deploy.py", "--judge", "--vla", "smolvla", "--run", "latest*")
    assert out.returncode == 1 and "YYYY-MM-DD-HHMM" in out.stderr


def test_deploy_judge_rejects_bridge():
    pytest.importorskip("boto3")
    out = _cli("deploy.py", "--judge", "--vla", "pi", "--bridge")
    assert out.returncode == 1 and "cannot be combined" in out.stderr


def test_deploy_judge_rejects_unknown_judge():
    pytest.importorskip("boto3")
    out = _cli("deploy.py", "--judge", "other", "--vla", "smolvla")
    assert out.returncode == 2 and "invalid choice" in out.stderr


def test_destroy_judge_has_no_delete_all():
    pytest.importorskip("yaml")
    out = _cli("destroy.py", "--judge", "--vla", "smolvla", "--delete-all", "--yes")
    assert out.returncode == 1 and "no data of its own" in out.stderr


def test_vlasim_judge_help_and_usage():
    out = _cli("vlasim.py", "judge", "--help")
    assert out.returncode == 0 and "deploy.py --judge" in out.stdout
    assert "vlasim.py judge" in _cli("vlasim.py", "--help").stdout


# --- run selection and names --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def deploy():
    pytest.importorskip("boto3")
    pytest.importorskip("yaml")
    import deploy as mod  # noqa: PLC0415
    return mod


def test_pick_run_latest_with_videos(deploy):
    runs = ["2026-10-18-0100", "2026-10-20-0130", "2026-10-21-0900", "scratch"]
    counts = {"2026-10-18-0100": 4, "2026-10-20-0130": 10}
    assert deploy.pick_run(runs, counts) == ("2026-10-20-0130", None)  # newest run has no videos yet
    assert deploy.pick_run(runs, counts, "2026-10-18-0100") == ("2026-10-18-0100", None)
    run, err = deploy.pick_run(runs, counts, "2026-10-21-0900")
    assert run is None and "no task-N/videos" in err
    run, err = deploy.pick_run(runs, counts, "2026-01-01-0000")
    assert run is None and "not found" in err
    assert deploy.pick_run([], {})[0] is None


def test_video_key_re(deploy):
    assert deploy.VIDEO_KEY_RE.match("2026-10-20-0130/task-0/videos/libero_spatial_0--eval_episode_0.mp4")
    assert not deploy.VIDEO_KEY_RE.match("2026-10-20-0130/task-0/videos/a.txt")
    assert not deploy.VIDEO_KEY_RE.match("2026-10-20-0130/diag-fail/a.mp4")


def test_results_bucket_name_matches_cdk_rule(deploy):
    acct = "123456789012"
    assert deploy.results_bucket_name("vla-sim-results", "SmolVLA-Demo", "ap-northeast-2", acct) == \
        "vla-sim-results-smolvla-demo-ap-northeast-2-123456789012"
    # 63-char cap: same expectation as cdk/test/cdk.test.ts
    assert deploy.results_bucket_name("vla-sim-results", "OpenArm-Lift-ACT-Demo", "ap-northeast-2", acct) == \
        "vla-sim-results-openarm-lift-act-ap-northeast-2-123456789012"


def test_judge_stack_name_matches_cdk(deploy):
    m = re.search(r"JUDGE_STACK_PREFIX: Record<string, string> = \{(.*?)\};", JUDGE_TS, re.S)
    ts = dict(re.findall(r"'([a-z0-9-]+)':\s*'([A-Za-z0-9]+)'", m.group(1)))
    assert set(ts) == set(JUDGE_TARGETS)
    for judge, prefix in ts.items():
        assert deploy.judge_stack_name(judge, "SmolVLA-Demo") == f"{prefix}-SmolVLA-Demo"


def test_judge_ami_regions_match_supported_regions():
    sys.path.insert(0, str(ROOT))
    import vlasim  # noqa: PLC0415
    m = re.search(r"JUDGE_AMI_MAPPING: Record<string, string> = \{(.*?)\};", JUDGE_TS, re.S)
    regions = set(re.findall(r"'([a-z]{2}-[a-z]+-\d)':\s*'ami-[0-9a-f]+'", m.group(1)))
    assert regions == vlasim.SUPPORTED_REGIONS and "ap-northeast-2" in regions


# --- rendered UserData ----------------------------------------------------------------------------------

@pytest.mark.parametrize("judge", JUDGE_TARGETS)
def test_judge_userdata_renders(judge, tmp_path):
    pytest.importorskip("jinja2")
    pytest.importorskip("yaml")
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--judge-target", judge,
                           "--output-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (tmp_path / f"{judge}.sh").read_text()
    assert text.startswith("#!") and "{%" not in text and "{{" not in text
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(tmp_path / f"{judge}.sh")]).returncode == 0
    # official scoring entry point at the pinned revision; outputs next to the judged run
    assert "-m batch_utils.terminal_reward" in text and "--with-rationale" in text
    assert 'revision="031299e46e452fc0cedb4e0975eb9fa95fc37783"' in text
    assert 'batch_utils/requirements.txt' in text
    for key in ("judge.json", "judge.txt"):
        assert f'"${{S3_RUN}}/{key}"' in text
    assert "/flora-judge/userdata" in text and "Noncommercial" in text
    assert "cfn_signal 0" in text and "self_terminate" in text


def test_generate_needs_vla_or_judge():
    out = _cli("generate.py", "--dry-run")
    assert out.returncode == 2
    out = _cli("generate.py", "--vla", "smolvla", "--judge-target", "flora-judge", "--dry-run")
    assert out.returncode == 2 and "not allowed with" in out.stderr
