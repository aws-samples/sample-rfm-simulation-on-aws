"""Offline tests for the RLinf RL post-training target (rlinf-oft-libero).

- assets/rlinf/rlinf_report.py: eval-log parsing, Wilson intervals, phase records, summary.txt lines,
  results.json / summary.txt report, training-curve rows.
- generate.py: smoke / full / --rl-steps / --rl-eval-run renders (bash -n, shellcheck -S error when present),
  pins and stage order in the script, input validation.
- results.py + leaderboard.py: the two phases read back as a baseline row and an `[rl-grpo-<N>steps]` row.
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assets" / "rlinf"))
import rlinf_report as rr  # noqa: E402

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")
import generate  # noqa: E402
import leaderboard as lb  # noqa: E402

VLA = "rlinf-oft-libero"
BASH = shutil.which("bash")
SHELLCHECK = shutil.which("shellcheck") or (str(Path.home() / ".local/bin/shellcheck")
                                            if (Path.home() / ".local/bin/shellcheck").exists() else None)


# --- report helpers ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text,rate", [
    ("x\neval/success_once: 0.52\n", 0.52),
    ("{'eval/success_once': 0.875, 'eval/return': 3.1}", 0.875),
    ("eval/success_once=tensor(0.4100)", 0.41),
    ("| eval/success_once | 0.5 |", 0.5),
    ("{'eval/success_at_end': array(0.35, dtype=float32), 'eval/success_once': array(0.5, dtype=float32)}", 0.5),  # v0.4
    ("eval/success_once: 0.1\n...\neval/success_once: 0.9\n", 0.9),   # last value wins
    ("env/success_once: 0.3\n", None),                                 # training metric is not the eval
    ("eval/success_once: 7.0\n", None),                                # not a rate
    ("", None),
])
def test_parse_success(text, rate):
    assert rr.parse_success(text) == rate


def test_phase_record_and_summary_lines():
    rec = rr.phase_record("rl", 0.5, 100, "s3://b/r/rl/checkpoints/global_step_5", "rl-grpo-5steps",
                          {"rl_steps": 5, "rl_profile": "smoke"})
    assert (rec["n_success"], rec["n_episodes"]) == (50, 100)
    lo, hi = rr.wilson(50, 100)
    assert rec["ci95_lo"] == lo and rec["ci95_hi"] == hi and lo < 0.5 < hi
    text = rr.summary_lines(rec, {"model": VLA, "suite": "libero_goal", "stack_mujoco": "3.3.2"})
    kv = dict(line.split(": ", 1) for line in text.splitlines())
    assert kv["condition"] == "rl-grpo-5steps" and kv["n_episodes"] == "100" and kv["n_success"] == "50"
    assert kv["success_rate"] == "0.5000" and kv["model"] == VLA and kv["rl_steps"] == "5"
    none = rr.phase_record("baseline", None, 100, "sft", "baseline")
    assert none["n_episodes"] is None and "success_rate: unknown" in rr.summary_lines(none, {})


def test_report_delta_and_overlap():
    base = rr.phase_record("baseline", 0.49, 500, "sft", "baseline")
    rl = rr.phase_record("rl", 0.94, 500, "rl", "rl-grpo-100steps", {"rl_steps": 100})
    data, text = rr.report(base, rl, {"suite": "libero_goal", "rl_profile": "full", "published_sft": 0.494,
                                      "published_rl": 0.9396})
    assert data["delta_pp"] == 45.0 and [r["phase"] for r in data["rows"]] == ["baseline", "rl"]
    assert "intervals: disjoint" in text and "+45.0 points" in text and "93.96%" in text
    _, small = rr.report(rr.phase_record("baseline", 0.5, 10, "s", "baseline"),
                         rr.phase_record("rl", 0.6, 10, "r", "rl-grpo-5steps"))
    assert "overlap" in small


def test_cli_phase_and_report(tmp_path):
    log = tmp_path / "eval.log"
    log.write_text("starting\n{'eval/success_once': 0.25}\n")
    common = json.dumps({"model": VLA, "suite": "libero_goal"})
    for phase, cond in (("baseline", "baseline"), ("rl", "rl-grpo-5steps")):
        assert rr.main(["phase", str(tmp_path / f"{phase}.json"), "--phase", phase, "--log", str(log), "--n", "100",
                        "--checkpoint", "x", "--condition", cond, "--common", common,
                        "--summary", str(tmp_path / f"{phase}_summary.txt")]) == 0
    assert rr.main(["report", str(tmp_path / "baseline.json"), str(tmp_path / "rl.json"), str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["delta_pp"] == 0.0 and res["rows"][1]["n_success"] == 25
    missing = tmp_path / "none.log"
    assert rr.main(["phase", str(tmp_path / "m.json"), "--phase", "rl", "--log", str(missing), "--n", "8",
                    "--checkpoint", "x", "--condition", "rl-grpo-5steps"]) == 1


def test_curve_rows():
    scalars = {"env/success_once": [(1, 0.2), (2, 0.4)], "rollout/rewards": [(1, 1.0), (2, 2.0)],
               "env/custom": [(2, 9.0)], "train/other": [(1, 0.0)]}
    keep, rows = rr.curve_rows(scalars)
    assert keep == ["env/success_once", "rollout/rewards", "env/custom"]
    assert rows == [{"step": 1, "env/success_once": 0.2, "rollout/rewards": 1.0},
                    {"step": 2, "env/success_once": 0.4, "rollout/rewards": 2.0, "env/custom": 9.0}]


# --- rendering --------------------------------------------------------------------------------------

def _render(tmp_path, *extra):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", VLA, "--output-dir", str(tmp_path),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return proc, tmp_path / f"{VLA}.sh"


def _check_script(path):
    text = path.read_text()
    assert BASH and subprocess.run([BASH, "-n", str(path)], capture_output=True).returncode == 0
    if SHELLCHECK:
        sc = subprocess.run([SHELLCHECK, "-S", "error", str(path)], capture_output=True, text=True)
        assert sc.returncode == 0, sc.stdout
    assert "{%" not in text and "{{ " not in text
    return text


CFG = yaml.safe_load((ROOT / "models" / f"{VLA}.yaml").read_text())


@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_render_profiles(tmp_path, profile):
    proc, path = _render(tmp_path, "--rl-profile", profile)
    assert proc.returncode == 0, proc.stderr
    text = _check_script(path)
    m = CFG["model"]
    prof = CFG["profiles"][profile]
    assert f'IMAGE="{m["docker_image"]}@{m["docker_digest"]}"' in text
    assert m["rlinf_commit"] in text and m["suites"]["goal"]["hf_model_revision"] in text
    assert f"RL_STEPS={prof['steps']}" in text and f"EVAL_NUM_ENVS={prof['eval_num_envs']}" in text
    assert f"MAX_WALL_HOURS={prof['max_wall_hours']}" in text
    assert f"EARLY_SIGNAL={'true' if prof['early_signal'] else 'false'}" in text
    for o in prof["overrides"]:
        assert o in text
    assert f"-ge {prof['instance']['min_gpus']} ]" in text
    # Stage order: verify (imports + render + hydra compose) before the 15 GB download, baseline before training.
    order = ["VERIFY OK", "--cfg job", "snapshot_download", "run_eval baseline task-0", "train_embodied_agent.py "
             "--config-path /workspace/RLinf/examples/embodiment/config/ --config-name libero_goal_grpo_openvlaoft ${TRAIN_OVERRIDES}\"",
             "run_eval rl task-1", "rlinf_report.py\" report"]
    pos = [text.find(s) for s in order]
    assert all(p > 0 for p in pos) and pos == sorted(pos), dict(zip(order, pos))
    assert 'mujoco==3.3.2' in text and "PYTHONFAULTHANDLER=1" in text
    # The report script is embedded and decodes to the asset.
    assert "REPORTEOF" in text
    # The RL row is labelled with the step that was evaluated (cap / --rl-eval-run), set before the RL eval.
    assert text.find('RL_CONDITION="rl-grpo-${EVAL_STEP}steps"') < text.find("run_eval rl task-1")


def test_render_steps_and_eval_run(tmp_path):
    proc, path = _render(tmp_path, "--rl-steps", "12", "--rl-eval-run", "2026-10-31-0100")
    assert proc.returncode == 0, proc.stderr
    text = _check_script(path)
    assert "RL_STEPS=12" in text and 'EVAL_RUN="2026-10-31-0100"' in text and "--rl-steps 12" in text
    assert "--include '*model_state_dict/*'" in text and 'CKPT_PROFILE' in text


@pytest.mark.parametrize("extra,msg", [
    (["--rl-eval-run", "last"], "YYYY-MM-DD-HHMM"),
    (["--rl-steps", "0"], ">= 1"),
    (["--rl-profile", "huge"], "invalid choice"),
])
def test_render_rejects(tmp_path, extra, msg):
    proc, _ = _render(tmp_path, *extra)
    assert proc.returncode != 0 and msg in proc.stderr


def test_rl_flags_only_for_rlinf(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", "smolvla", "--rl-profile", "full",
                           "--output-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0 and "apply to rlinf-oft-libero only" in proc.stderr


def test_bad_override_rejected(capsys):
    cfg = json.loads(json.dumps(CFG))
    cfg["profiles"]["smoke"]["overrides"] = ["env.train.total_num_envs=8; rm -rf /"]
    with pytest.raises(SystemExit):
        generate.generate_rlinf(cfg, True, VLA, profile="smoke")
    assert "hydra overrides" in capsys.readouterr().err


def test_profiles_instance_floors():
    """smoke = one GPU, full = eight; full signals early because the run does not fit the 12 h WaitCondition."""
    smoke, full = CFG["profiles"]["smoke"], CFG["profiles"]["full"]
    assert smoke["instance"]["min_gpus"] == 1 and full["instance"]["min_gpus"] == 8
    assert full["early_signal"] is True and smoke["early_signal"] is False
    for prof in (smoke, full):
        minutes = int(prof["instance"]["creationpolicy_timeout"].strip("PTM"))
        assert minutes <= 720 and prof["max_wall_hours"] * 60 >= minutes
    assert CFG["instance"]["preferred"] == smoke["instance"]["preferred"]  # vlasim doctor reads the default


# --- results / leaderboard --------------------------------------------------------------------------

def test_phases_become_baseline_and_rl_rows(monkeypatch):
    run = "2026-10-31-0130"
    common = {"model": VLA, "suite": "libero_goal", "stack_rlinf_commit": "2348ece3", "instance_type": "g6e.8xlarge"}
    base = rr.phase_record("baseline", 0.48, 100, "sft", "baseline")
    rl = rr.phase_record("rl", 0.62, 100, "rl", "rl-grpo-5steps", {"rl_steps": 5})
    t0 = lb.run_started(run)
    objs = {f"{run}/task-0/summary.txt": (rr.summary_lines(base, common), t0 + dt.timedelta(minutes=50)),
            f"{run}/task-1/summary.txt": (rr.summary_lines(rl, common), t0 + dt.timedelta(minutes=150))}
    bucket_name = "vla-sim-results-rlinf-oft-libero-ap-northeast-2-123456789012"

    class Fake:
        def prefixes(self, region, bucket, prefix):
            return sorted({prefix + k[len(prefix):].split("/", 1)[0] + "/" for k in objs if k.startswith(prefix)})

        def objects(self, region, bucket, prefix):
            return [dict(Key=k, LastModified=v[1]) for k, v in sorted(objs.items()) if k.startswith(prefix)]

        def text(self, region, bucket, key, head=None):
            v = objs.get(key)
            return None if v is None else v[0]

    class Client:
        def get_caller_identity(self):
            return {"Account": "123456789012"}

        def list_buckets(self):
            return {"Buckets": [{"Name": bucket_name}]}

    class Boto:
        def client(self, *a, **k):
            return Client()

    runs = lb.collect_runs(argparse.Namespace(region=None, vla=None, since=None), Boto(), s3=Fake(), workers=1)
    cells = lb.build(lb.select_cells(runs, targets=[VLA, "openvla-oft"]))
    got = {(c["target"], c["benchmark"]): (c["k"], c["n"]) for c in cells["cells"]}
    assert got == {(VLA, "LIBERO-Goal"): (48, 100), (f"{VLA} [rl-grpo-5steps]", "LIBERO-Goal"): (62, 100)}
    assert not cells["stack_differences"]  # same stack fingerprint for both phases
    assert lb.RL_CONDITION_RE.match("rl-grpo-100steps") and not lb.RL_CONDITION_RE.match("S0")


def test_doctor_reads_the_profile_instance_list():
    import vlasim
    assert vlasim._model_preferred(VLA) == CFG["profiles"]["smoke"]["instance"]["preferred"]
    full = vlasim._model_preferred(VLA, "full")
    assert full == CFG["profiles"]["full"]["instance"]["preferred"]
    assert min(vlasim._instance_vcpu(t) for t in full) == 96  # p4de.24xlarge: an 8-GPU quota is needed
    assert vlasim._peek_core_args(["--vla", VLA, "--rl-profile", "full"])[3] == "full"
    assert vlasim._model_preferred("xvla", "full") == vlasim._model_preferred("xvla")  # no profiles block


def test_hydra_config_path_is_absolute():
    """Hydra resolves a relative --config-path against the script's directory, so
    examples/embodiment/config/ became examples/embodiment/examples/embodiment/config."""
    import re

    text = (Path(__file__).resolve().parents[1] / "templates" / "rlinf-userdata.sh.j2").read_text()
    paths = re.findall(r"--config-path\s+(\S+)", text)
    assert paths and all(p.startswith("/") for p in paths), paths


def test_container_exports_embodied_path():
    """The train/eval configs put file://${oc.env:EMBODIED_PATH}/config/ in hydra.searchpath
    (run_embodiment.sh exports it); without it the compose step fails with a KeyError."""
    text = (ROOT / "templates" / "rlinf-userdata.sh.j2").read_text()
    assert "-e EMBODIED_PATH=/workspace/RLinf/examples/embodiment" in text


def test_train_overrides_only_touch_existing_rollout_keys():
    """rollout.model in the GRPO train configs has no is_lora/lora_rank (hydra rejects
    the override); the rollout worker merges actor.model into it instead."""
    import yaml

    cfg = yaml.safe_load((ROOT / "models" / "rlinf-oft-libero.yaml").read_text())
    found = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "overrides":
                    found.extend(v or [])
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(cfg)
    assert found and not [o for o in found if o.startswith("rollout.model.") and "lora" in o]


def test_gpu_guards_around_ray_runs():
    """A daemon-reload revoked GPU access mid-run (NVML Unknown Error); RLinf then put the rollout
    worker on NO_ACCEL, it died in __init__ and the eval driver waited ~2 h on a dead actor."""
    text = (ROOT / "templates" / "rlinf-userdata.sh.j2").read_text()
    assert "native.cgroupdriver=cgroupfs" in text
    assert text.count('gpu_check "') == 2 and text.count("watch_dead_actor \"${WORK}/logs/") == 2
