"""vlaeval: allenai/vla-evaluation-harness (vla-eval) run unmodified.

Covers the plan validation in generate.vlaeval_plan, the rendered UserData (pins, image by digest, the eval config
it writes, task sets, HF token gate), the results adapter assets/vlaeval/vlaeval_results.py against real aggregate
JSON files of the 2026-11-08 run (tests/fixtures/vlaeval/, copied as written plus a final newline), the per-task loop run through bash +
jq with a stub `vla-eval` / `aws`, and the summary.txt fields results.py reads.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "assets" / "vlaeval"))

yaml = pytest.importorskip("yaml")
pytest.importorskip("jinja2")

import vlaeval_results  # noqa: E402
from test_results import BUCKET, FakeS3  # noqa: E402

import generate  # noqa: E402
import results  # noqa: E402
import seeds  # noqa: E402

BASH = shutil.which("bash")
JQ = shutil.which("jq")
SETSID = shutil.which("setsid")
FIX = ROOT / "tests" / "fixtures" / "vlaeval"
SP50S4 = FIX / "sp50s4_LIBEROBenchmark_libero_spatial_aggregate.json"   # 4-shard run: 48/50, task 4 ep 0 and 2 failed
SMOKE = FIX / "smoke_LIBEROBenchmark_libero_spatial_aggregate.json"     # 1 episode, failure_reason timeout
DIGEST = "sha256:bc9930a7e42f970bdde7f3aaddaab0c223bd8e7e922e387cd9eb48aa89ccf2f1"


def _model():
    return yaml.safe_load((ROOT / "models" / "vlaeval.yaml").read_text())


def _render(out_dir, *extra, ok=True):
    proc = subprocess.run([sys.executable, str(ROOT / "generate.py"), "--vla", "vlaeval", "--output-dir", str(out_dir),
                           *extra], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if ok:
        assert proc.returncode == 0, f"{extra}:\n{proc.stdout}\n{proc.stderr}"
    return proc


def _tasks_json(text):
    line = next(ln for ln in text.splitlines() if ln.startswith("TASKS_JSON='"))
    return json.loads(line[len("TASKS_JSON='"):-1].replace("'\\''", "'"))


# --- plan validation -----------------------------------------------------------------------------------------

def test_default_plan_is_the_measured_recipe():
    benches, servers, tasks = generate.vlaeval_plan(_model())
    assert [b["image"] for b in benches] == [f"ghcr.io/allenai/vla-evaluation-harness/libero@{DIGEST}"]
    assert servers == [dict(name="lerobot-pi05-libero", config="configs/model_servers/lerobot/pi05_libero.yaml",
                            hf_repo="lerobot/pi05_libero_finetuned_v044",
                            hf_model_revision="7152f13e34c6c115d808aea0e5d08aca74eac14b",
                            gated_repos=["google/paligemma-3b-pt-224"])]
    (t,) = tasks
    # = raw/pack/sp50.yaml of the measured run, with the image by digest instead of :latest
    assert t["eval_config"] == {
        "server": {"url": "ws://localhost:8000", "timeout": 900},
        "docker": {"image": f"ghcr.io/allenai/vla-evaluation-harness/libero@{DIGEST}", "user": "host"},
        "output_dir": "/home/ubuntu/vlaeval-eval/task-0",
        "benchmarks": [{"benchmark": "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark",
                        "subname": "libero_spatial", "episodes_per_task": 5,
                        "params": {"suite": "libero_spatial", "seed": 7, "num_steps_wait": 10},
                        "recording": {"record_video": True}}]}
    assert (t["suite"], t["action_type"], t["condition"]) == ("libero_spatial", "other", "S0")
    m = _model()["model"]
    assert (m["harness_version"], m["harness_commit"]) == ("0.9.0", "8798464b2e3317f49b259dca8861aea04f75bb59")


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c["vlaeval"]["benchmarks"]["libero"].update(image_digest="latest"), "image_digest"),
    (lambda c: c["vlaeval"]["benchmarks"]["libero"].update(image="ghcr.io/x/libero:latest"), ".image"),
    (lambda c: c["vlaeval"]["model_servers"]["lerobot-pi05-libero"].update(hf_model_revision="main"),
     "hf_model_revision"),
    (lambda c: c["vlaeval"]["model_servers"]["lerobot-pi05-libero"].update(config="../../etc/passwd.yaml"), ".config"),
    (lambda c: c["model"].update(harness_commit="v0.9.0"), "harness_commit"),
    (lambda c: c["tasks"][0].update(model_server="nope"), "model_server"),
    (lambda c: c["tasks"][0].update(benchmark="nope"), "benchmark"),
    (lambda c: c["tasks"][0].update(episodes_per_task=0), "episodes_per_task"),
    (lambda c: c["tasks"][0].update(params={"seed": 1}), "params"),
    (lambda c: c["tasks"][0].update(params={"x": "a b"}), "params.x"),
    (lambda c: c["tasks"][0].update(typo=1), "unknown key"),
    (lambda c: c["tasks"][0].update(suite="libero spatial"), "suite"),
    (lambda c: c["vlaeval"]["run"].update(docker_user="root; rm"), "docker_user"),
])
def test_plan_rejects_bad_settings(mutate, msg):
    cfg = copy.deepcopy(_model())
    mutate(cfg)
    with pytest.raises(ValueError, match=msg):
        generate.vlaeval_plan(cfg)


def test_a_new_pair_is_one_entry():
    cfg = copy.deepcopy(_model())
    cfg["vlaeval"]["model_servers"]["other"] = {"config": "configs/model_servers/lerobot/smolvla_libero.yaml"}
    cfg["tasks"].append(dict(cfg["tasks"][0], model_server="other", suite="libero_goal"))
    _, servers, tasks = generate.vlaeval_plan(cfg)
    assert [s["name"] for s in servers] == ["lerobot-pi05-libero", "other"]
    assert tasks[1]["eval_config"]["output_dir"].endswith("/task-1") and tasks[1]["hf_repo"] == ""


def test_seed_flag_reaches_the_benchmark_params():
    cfg = seeds.apply(_model(), "vlaeval", seeds.resolve("123"))
    _, _, (t,) = generate.vlaeval_plan(cfg)
    assert t["eval_config"]["benchmarks"][0]["params"]["seed"] == 123
    assert cfg["seed_plan"]["status"] == "applied" and cfg["seed_plan"]["policy"] == "no"


# --- rendered UserData ----------------------------------------------------------------------------------------

@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("task_set", ["", "probe", "object", "goal", "long"])
def test_render_and_bash_n(task_set, tmp_path):
    _render(tmp_path, *(["--task-set", task_set] if task_set else []))
    script = tmp_path / "vlaeval.sh"
    assert subprocess.run([BASH, "-n", str(script)]).returncode == 0
    (t,) = _tasks_json(script.read_text())
    suite = {"": "libero_spatial", "probe": "libero_spatial", "object": "libero_object", "goal": "libero_goal",
             "long": "libero_10"}[task_set]
    assert t["suite"] == suite == t["eval_config"]["benchmarks"][0]["params"]["suite"]
    if task_set == "probe":
        assert t["eval_config"]["benchmarks"][0]["max_tasks"] == 1


def test_render_pins_and_runtime_install(tmp_path):
    _render(tmp_path)
    text = (tmp_path / "vlaeval.sh").read_text()
    assert 'uv pip install -q --python "${VENV}/bin/python" "vla-eval==0.9.0"' in text
    assert '( cd "${WORK}" && uv pip install' in text            # outside the checkout (exclude-newer pin)
    assert "git clone -q --branch v0.9.0 https://github.com/allenai/vla-evaluation-harness.git" in text
    assert '!= "8798464b2e3317f49b259dca8861aea04f75bb59" ]' not in text   # checked with = and die
    assert '[ "${HARNESS_COMMIT}" = "8798464b2e3317f49b259dca8861aea04f75bb59" ] || die' in text
    assert "uv venv -q --python 3.11" in text
    assert f'docker pull -q "ghcr.io/allenai/vla-evaluation-harness/libero@{DIGEST}"' in text
    assert ':latest' not in text
    assert 'snapshot_download("lerobot/pi05_libero_finetuned_v044", revision="7152f13e34c6c115d808aea0e5d08aca74eac14b")' in text
    assert '"huggingface_hub==2.4.0"' in text
    assert '_args+=(--arg "${_ckpt_arg}=${_ckpt}")' in text
    assert 'vla-eval" run --config "${EVAL_CONFIG}" --yes' in text
    # the adapter is embedded byte for byte
    import base64
    blob = text.split("<<'B64EOF'\n", 1)[1].split("B64EOF", 1)[0]
    assert base64.b64decode(blob) == (ROOT / "assets" / "vlaeval" / "vlaeval_results.py").read_bytes()


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("code,ok", [("200", True), ("403", False), ("401", False), ("404", True)])
def test_gated_repo_gate(tmp_path, code, ok):
    _render(tmp_path / "r")
    text = (tmp_path / "r" / "vlaeval.sh").read_text()
    start = text.index('HF_TOKEN=$(aws ssm get-parameter')
    end = text.index("export HF_TOKEN\n", start)
    script = (f'set -euo pipefail\nREGION=us-east-1\nlog() {{ echo "$*"; }}\ndie() {{ echo "DIE $*"; exit 9; }}\n'
              f'aws() {{ echo hf_x; }}\ncurl() {{ echo {code}; }}\n' + text[start:end])
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
    assert (proc.returncode == 0) == ok, proc.stdout + proc.stderr
    if not ok:
        assert "google/paligemma-3b-pt-224" in proc.stdout
    empty = script.replace("aws() { echo hf_x; }", "aws() { echo None; }")
    proc = subprocess.run([BASH, "-c", empty], capture_output=True, text=True)
    assert proc.returncode == 9 and "SSM /vla-simulator/hf-token is empty" in proc.stdout


# --- results adapter on the real aggregate JSON -----------------------------------------------------------------

def test_adapter_counts_match_the_harness():
    agg = vlaeval_results.load(SP50S4)
    assert vlaeval_results.counts(agg) == (48, 50, 0)
    assert abs(48 / 50 - agg["mean_success"]) < 1e-9
    fails = [(e["task_idx"], e["episode_id"]) for e in vlaeval_results.episodes(agg) if not e["success"]]
    assert fails == [(4, 0), (4, 2)]   # = the two *_fail.mp4 of the run
    agg = vlaeval_results.load(SMOKE)
    assert vlaeval_results.counts(agg) == (0, 1, 1)


def test_adapter_writes_summary_and_gallery_videos(tmp_path):
    out = tmp_path / "out"
    vids = out / "episodes" / "LIBEROBenchmark_libero_spatial"
    vids.mkdir(parents=True)
    shutil.copy(SP50S4, out / SP50S4.name.split("_", 1)[1])
    names = (FIX / "sp50s4_episode_videos.txt").read_text().split()
    for n in names:
        (vids / n).write_bytes(b"mp4")
    summary, vdir = tmp_path / "summary.txt", tmp_path / "videos"
    rc = vlaeval_results.main(["--output-dir", str(out), "--summary", str(summary), "--video-dir", str(vdir),
                               "--suite", "libero_spatial", "--meta", "condition=S0", "--meta", "action_type=pick-place",
                               "--meta", "seed=999", "--meta", "stack_vla_eval_version=0.9.0"])
    assert rc == 0
    kv = dict(results.KV_RE.findall(summary.read_text()))
    assert (kv["suite"], kv["env_name"], kv["n_success"], kv["n_episodes"], kv["success_rate"]) == \
        ("libero_spatial", "libero_spatial", "48", "50", "0.960")
    assert kv["seed"] == "7"                 # the aggregate's value wins over --meta
    assert kv["condition"] == "S0" and kv["action_type"] == "pick-place" and kv["n_videos"] == "50"
    assert kv["stack_vla_eval_harness_version"] == "0.9.0" and kv["stack_vla_eval_model_server"] == "LeRobotModelServer"
    assert "benchmark" not in kv             # results.py reads `benchmark:` as a LIBERO-Plus row
    got = sorted(p.name for p in vdir.glob("*.mp4"))
    assert len(got) == 50 and "libero_spatial_t04--episode_0-failure.mp4" in got
    assert "libero_spatial_t00--episode_0-success.mp4" in got
    assert sum(n.endswith("-failure.mp4") for n in got) == 2
    cap = (vdir / "libero_spatial_t04--episode_0-failure.txt").read_text()
    assert "task 4, episode 0" in cap and "failure" in cap


def test_adapter_marks_an_errored_episode_as_failure(tmp_path):
    out = tmp_path / "out"
    (out / "episodes" / "LIBEROBenchmark_libero_spatial").mkdir(parents=True)
    shutil.copy(SMOKE, out / "LIBEROBenchmark_libero_spatial_aggregate.json")
    (out / "episodes" / "LIBEROBenchmark_libero_spatial" / "task0000_ep0000_error.mp4").write_bytes(b"x")
    rc = vlaeval_results.main(["--output-dir", str(out), "--summary", str(tmp_path / "s.txt"),
                               "--video-dir", str(tmp_path / "v"), "--suite", "libero_spatial"])
    assert rc == 0
    kv = dict(results.KV_RE.findall((tmp_path / "s.txt").read_text()))
    assert (kv["n_success"], kv["n_episodes"], kv["n_errors"]) == ("0", "1", "1")
    assert [p.name for p in (tmp_path / "v").glob("*.mp4")] == ["libero_spatial_t00--episode_0-failure.mp4"]
    assert "episode error: timeout" in (tmp_path / "v" / "libero_spatial_t00--episode_0-failure.txt").read_text()


def test_adapter_without_aggregate_fails(tmp_path):
    (tmp_path / "out").mkdir()
    assert vlaeval_results.main(["--output-dir", str(tmp_path / "out"), "--summary", str(tmp_path / "s"),
                                 "--video-dir", str(tmp_path / "v"), "--suite", "x"]) == 2
    assert not (tmp_path / "s").exists()


def test_adapter_rejects_bad_meta_keys(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(SMOKE, out / "a_aggregate.json")
    assert vlaeval_results.main(["--output-dir", str(out), "--summary", str(tmp_path / "s"),
                                 "--video-dir", str(tmp_path / "v"), "--suite", "x", "--meta", "Bad Key=1"]) == 2


# --- the per-task loop through bash + jq with a stub vla-eval ------------------------------------------------------

STUB_VLAEVAL = """#!/bin/bash
# stub of `vla-eval`: serve = a long-lived process; run = write a real aggregate + episode videos into output_dir
echo "vla-eval $*" >> "$CALLS"
if [ "$1" = serve ]; then exec sleep 300; fi
cfg="$3"
dest="$EVAL_BASE_DIR/$(basename "$(jq -r .output_dir "$cfg")")"
mkdir -p "$dest/episodes/LIBEROBenchmark_libero_spatial"
cp "$FIXTURE" "$dest/LIBEROBenchmark_libero_spatial_aggregate.json"
: > "$dest/recording-x.sqlite"
for n in $(cat "$VIDEO_LIST"); do echo x > "$dest/episodes/LIBEROBenchmark_libero_spatial/$n"; done
exit "${STUB_RC:-0}"
"""


def _loop_script(text, tmp_path):
    start = text.index('CURRENT_SERVER=""\nstart_server()')
    end = text.index('[ -z "${FAILED}" ] || die', start)
    work = tmp_path / "work"
    (work / "venv" / "bin").mkdir(parents=True)
    stub = work / "venv" / "bin" / "vla-eval"
    stub.write_text(STUB_VLAEVAL)
    stub.chmod(0o755)
    cfg = work / "harness" / "configs" / "model_servers" / "lerobot" / "pi05_libero.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("extends: _base.yaml\nargs:\n  chunk_size: 10  # LIBERO protocol\n")
    shutil.copy(ROOT / "assets" / "vlaeval" / "vlaeval_results.py", work / "vlaeval_results.py")
    (tmp_path / "ckpt.path").write_text("/hf/snapshots/7152f13e\n")
    head = f"""set -euo pipefail
WORK={work}; VENV={work}/venv; HARNESS_DIR={work}/harness
export EVAL_BASE_DIR={tmp_path}/eval; VIDEO_BASE_DIR={tmp_path}/videos; RUN_ID=R; S3_BUCKET=b; REGION=r
export CALLS={tmp_path}/calls FIXTURE={SP50S4} VIDEO_LIST={FIX / 'sp50s4_episode_videos.txt'}
LOG={tmp_path}/userdata.log
VLAEVAL_VERSION=0.9.0; HARNESS_COMMIT=8798464b2e3317f49b259dca8861aea04f75bb59; DRIVER_VERSION=595.112.07
mkdir -p "$EVAL_BASE_DIR"
log() {{ echo "LOG $*"; }}
die() {{ echo "DIE $*"; exit 9; }}
_trap_handler() {{ echo TRAP; }}
SERVER_PID=""
stop_server() {{ [ -z "$SERVER_PID" ] || kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }}
aws() {{ echo "aws $*" >> "$CALLS"; }}
curl() {{ return 0; }}
"""
    body = text[start:end].replace("/tmp/vlaeval_ckpt_${SERVER_NAME}.path", str(tmp_path / "ckpt.path"))
    body = body.replace("/tmp/vlaeval_", f"{tmp_path}/vlaeval_")
    return head + body + 'echo "FAILED=[${FAILED}]"\necho "SUMMARY_ALL=${SUMMARY_ALL}"\n'


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
@pytest.mark.parametrize("stub_rc", [0, 1])
def test_task_loop_with_stub_harness(tmp_path, stub_rc):
    _render(tmp_path / "r")
    script = _loop_script((tmp_path / "r" / "vlaeval.sh").read_text(), tmp_path)
    env = {"PATH": os.environ["PATH"], "STUB_RC": str(stub_rc), "HOME": str(tmp_path)}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (tmp_path / "calls").read_text()
    assert "vla-eval serve --config configs/model_servers/lerobot/pi05_libero.yaml --arg checkpoint=/hf/snapshots/7152f13e" in calls
    run_cfg = json.loads((tmp_path / "work" / "eval-task-0.json").read_text())
    assert run_cfg["docker"]["image"].endswith(DIGEST) and run_cfg["server"]["timeout"] == 900
    kv = dict(results.KV_RE.findall((tmp_path / "vlaeval_task0_summary.txt").read_text()))
    assert (kv["n_success"], kv["n_episodes"], kv["n_action_steps"], kv["eval_exit"]) == ("48", "50", "10", str(stub_rc))
    assert kv["stack_vla_eval_benchmark_image"].endswith(DIGEST)
    assert kv["checkpoint"] == "lerobot/pi05_libero_finetuned_v044@7152f13e"
    assert kv["model"] == "lerobot-pi05-libero" and kv["condition"] == "S0" and kv["action_type"] == "other"
    # 50 renamed videos + captions uploaded under videos/; the harness's own mp4s are excluded from the raw copy
    assert calls.count("/R/task-0/videos/libero_spatial_t") == 100
    assert 'cp --recursive' in calls and '--exclude *.mp4' in calls
    assert "s3://b/R/task-0/summary.txt" in calls and "s3://b/R/task-0/protocol.yaml" in calls
    card = yaml.safe_load((tmp_path / "vlaeval_task0_summary.protocol.yaml").read_text())
    assert (card["k"], card["n"], card["seed"], card["chunk_rule"]) == (48, 50, "7", "n_action_steps=10")
    assert card["checkpoints"] == [{"repo": "lerobot/pi05_libero_finetuned_v044",
                                    "revision": "7152f13e34c6c115d808aea0e5d08aca74eac14b"}]
    assert ("FAILED=[ task-0]" in proc.stdout) == (stub_rc != 0)
    assert "48/50" in proc.stdout


# --- results.py reads the summary -------------------------------------------------------------------------------

def test_results_reads_the_adapter_summary(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(SP50S4, out / "LIBEROBenchmark_libero_spatial_aggregate.json")
    vlaeval_results.main(["--output-dir", str(out), "--summary", str(tmp_path / "s.txt"), "--video-dir",
                          str(tmp_path / "v"), "--suite", "libero_spatial", "--meta", "condition=S0",
                          "--meta", "action_type=pick-place"])
    s3 = FakeS3({BUCKET: {"R/task-0/summary.txt": (tmp_path / "s.txt").read_text()}})
    row = results.task_row(s3, "us-east-1", BUCKET, "R", "R/task-0/")
    assert (row["suite"], row["condition"], row["k"], row["n"], row["action_type"]) == \
        ("libero_spatial", "S0", 48, 50, "pick-place")
    assert row["stack"]["stack_vla_eval_harness_version"] == "0.9.0"
    assert "benchmark" not in row
