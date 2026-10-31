"""Evaluation protocol card (templates/_protocol-card.sh.j2): every target that writes summary.txt also writes
protocol.yaml next to it, and the rendered block produces a valid card under `set -euo pipefail`.

The block is cut out of a rendered userdata script and run with stub `aws`, `curl` and `nvidia-smi` on PATH,
so no AWS access or GPU is needed.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from targets import VLA_TARGETS  # noqa: E402

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import generate  # noqa: E402

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not on PATH")
MARK = "# Evaluation protocol card"
CARD_KEYS = {"schema", "target", "tool_commit", "checkpoints", "code_pins", "suite", "condition", "seed",
             "start_states", "chunk_rule", "step_cap", "instruction", "k", "n", "instance_type", "gpu", "region",
             "written_utc"}


def _render(vla, out_dir, extra=()):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", vla, "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return (out_dir / f"{vla}.sh").read_text()


def _card_block(text):
    start = text.index(MARK)
    end = text.index("protocol.yaml --region", start)
    end = text.index("\n    fi", end) + len("\n    fi")
    return text[start:end]


@pytest.mark.parametrize("vla", sorted(VLA_TARGETS))
def test_every_summary_writer_writes_a_card(vla, tmp_path):
    extra = ("--libero-suite", "10") if vla == "openvla-oft" else ()
    text = _render(vla, tmp_path, extra)
    writes_summary = bool(re.search(r"summary\.txt", text))
    if not writes_summary:
        assert MARK not in text
        return
    assert MARK in text, f"{vla}: writes summary.txt but no protocol card"
    assert text.count("/protocol.yaml --region") >= 1


def _stubs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "aws").write_text('#!/bin/sh\necho "$@" >> "$STUB_LOG"\n')
    (bin_dir / "curl").write_text('#!/bin/sh\ncase "$*" in *instance-type*) printf g5.xlarge;; *) printf tok;; esac\n')
    (bin_dir / "nvidia-smi").write_text('#!/bin/sh\necho "NVIDIA A10G"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir


def _run_card(tmp_path, summary_text, vla="openvla-oft", extra=("--libero-suite", "10")):
    block = _card_block(_render(vla, tmp_path, extra))
    summ = tmp_path / "task0_summary.txt"
    summ.write_text(summary_text)
    block = block.replace('"/tmp/oft_task${TASK_IDX}_summary.txt"', f'"{summ}"')
    script = "set -euo pipefail\n" + block + "\necho CARD_OK\n"
    env = {"PATH": f"{_stubs(tmp_path)}:/usr/bin:/bin", "STUB_LOG": str(tmp_path / "aws.log"), "TASK_IDX": "0",
           "REGION": "ap-northeast-2", "S3_TASK_PREFIX": "s3://b/run/task-0", "HOME": str(tmp_path)}
    out = subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "CARD_OK" in out.stdout, out.stderr
    card = yaml.safe_load((tmp_path / "task0_summary.protocol.yaml").read_text())
    return card, (tmp_path / "aws.log").read_text()


def test_card_from_a_full_summary(tmp_path):
    card, log = _run_card(tmp_path, "suite: libero_10\nsuccess_rate: 0.96\nn_episodes: 50\nn_success: 48\n"
                                    "condition: S3b\nreplan: 1\nmax_episode_steps: 520\nseed: 7\n")
    assert set(card) == CARD_KEYS
    assert card["schema"] == "rfm-simulation/protocol-card/v1" and card["target"] == "openvla-oft"
    assert (card["k"], card["n"], card["condition"], card["seed"]) == (48, 50, "S3b", "7")
    assert card["chunk_rule"] == "replan=1" and card["step_cap"] == "max_episode_steps=520"
    assert card["start_states"] == "unknown" and card["instruction"] == "unknown"
    assert (card["instance_type"], card["gpu"], card["region"]) == ("g5.xlarge", "1x NVIDIA A10G", "ap-northeast-2")
    assert {"repo": "moojink/openvla-7b-oft-finetuned-libero-10",
            "revision": "95220f9a3421a7ff12d4218e73d09ade830fa9a3", "suite": "10"} in card["checkpoints"]
    assert "s3://b/run/task-0/protocol.yaml" in log


def test_card_never_aborts_on_missing_keys(tmp_path):
    card, _ = _run_card(tmp_path, "success_rate: 0.5\n")
    assert card["k"] == "unknown" and card["seed"] == "unknown" and card["chunk_rule"] == "unknown"


def test_card_quotes_values(tmp_path):
    card, _ = _run_card(tmp_path, "condition: it's: a test\nn_episodes: 2\nn_success: 1\n")
    assert card["condition"] == "it's: a test"


def test_protocol_static_skips_unsafe_values():
    cfg = {"model": {"id": "x", "hf_repo": "org/model", "hf_model_revision": "abc123",
                     "build": {"repo_commit": "deadbeef", "note_version": "has space"}}}
    st = generate.protocol_static(cfg, "x")
    assert st["checkpoints"] == [dict(repo="org/model", revision="abc123", suite="")]
    assert st["code_pins"] == {"build.repo_commit": "deadbeef"}
    assert generate.protocol_static(cfg, "bad'name")["target"] == "unknown"


def test_no_aws_env_leaks_into_card(tmp_path):
    """The card lists only what the run recorded; it never copies credentials or the account id."""
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    card, _ = _run_card(tmp_path, "n_episodes: 1\nn_success: 1\n")
    assert not any(re.search(r"\b\d{12}\b", str(v)) for v in card.values())


def test_card_reads_cli_flags_recorded_in_the_summary(tmp_path):
    """LeRobot targets record the eval command, not a seed key: the card takes --seed / --policy.n_action_steps."""
    card, _ = _run_card(tmp_path, "suite: libero_spatial\nn_episodes: 10\nn_success: 8\ncondition: S0\n"
                                  "stack_eval_args: --policy.path=lerobot/smolvla_libero --eval.n_episodes=5 "
                                  "--seed=1000 --policy.n_action_steps=10 --env.task_ids=[0,1]\n")
    assert card["seed"] == "1000"
    assert card["chunk_rule"] == "policy.n_action_steps=10"
    assert card["step_cap"] == "unknown"
