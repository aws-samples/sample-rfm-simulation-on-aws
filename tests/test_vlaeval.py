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
# The harness's own X-VLA x CALVIN ABC->D reproduction (1000 sequences, avg len 4.301), as published in its repo at
# v0.9.0: docs/reproductions/data/xvla-calvin/merged.json (git LFS object sha256 0d7d6a86..., 389,704 bytes, fetched
# from media.githubusercontent.com) plus a final newline. Written by an older harness (0.0.4.dev22) through the same
# ResultCollector._aggregate_metrics that v0.9.0 uses (metric_keys success + completed_subtasks, mean_<key>).
CALVIN = FIX / "xvla_calvin_merged_CALVINBenchmark_aggregate.json"
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
    (lambda c: c["vlaeval"]["run"].update(port=8001), "port must be 8000"),  # audit A67bdcd36a1 F1
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
    # audit A67bdcd36a1 F3: recordings of another benchmark in the same output dir are not paired with this one
    other = out / "episodes" / "MetaWorldBenchmark_mt50"  # sorts after the LIBERO dir
    other.mkdir()
    for n in names:
        (other / n.replace("_failure", "_success")).write_bytes(b"other")
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
    assert all((vdir / n).read_bytes() == b"mp4" for n in got)
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
    for rel, yml in (("xvla/calvin.yaml", "args:\n  chunk_size: 20\n"), ("rtc/rtc.yaml", "args:\n  method: rtc\n")):
        (cfg.parent.parent / rel).parent.mkdir(parents=True, exist_ok=True)
        (cfg.parent.parent / rel).write_text(yml)
    shutil.copy(ROOT / "assets" / "vlaeval" / "vlaeval_results.py", work / "vlaeval_results.py")
    (tmp_path / "ckpt_lerobot-pi05-libero.path").write_text("/hf/snapshots/7152f13e\n")
    (tmp_path / "ckpt_xvla-calvin.path").write_text("/hf/snapshots/d76710ee\n")
    head = f"""set -euo pipefail
WORK={work}; VENV={work}/venv; HARNESS_DIR={work}/harness
export EVAL_BASE_DIR={tmp_path}/eval; VIDEO_BASE_DIR={tmp_path}/videos; RUN_ID=R; S3_BUCKET=b; REGION=r
export CALLS={tmp_path}/calls FIXTURE={SP50S4} VIDEO_LIST={FIX / 'sp50s4_episode_videos.txt'}
LOG={tmp_path}/userdata.log
VLAEVAL_VERSION=0.9.0; HARNESS_COMMIT=8798464b2e3317f49b259dca8861aea04f75bb59; DRIVER_VERSION=595.112.07
TORCH_CONSTRAINT=none
mkdir -p "$EVAL_BASE_DIR"
log() {{ echo "LOG $*"; }}
die() {{ echo "DIE $*"; exit 9; }}
_trap_handler() {{ echo TRAP; }}
SERVER_PID=""
stop_server() {{ [ -z "$SERVER_PID" ] || kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }}
aws() {{ echo "aws $*" >> "$CALLS"; }}
curl() {{ return 0; }}
"""
    body = text[start:end].replace("/tmp/vlaeval_ckpt_${SERVER_NAME}.path", f"{tmp_path}/ckpt_${{SERVER_NAME}}.path")
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
    assert kv["stack_torch_constraint"] == "none"


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


def test_every_stack_name_starts_with_its_target_id():
    # `vlasim results --vla <id>` keeps the buckets whose stack name starts with <id>; the first vlaeval stack
    # was named VLA-Eval-Demo and its results were not found (measured 2026-11-09)
    import destroy

    assert destroy.STACK_NAMES["vlaeval"] == "VLAEval-Demo"
    bad = {k: s for k, s in destroy.STACK_NAMES.items() if not s.lower().startswith(k)}
    assert not bad


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
def test_task_loop_fails_when_the_result_does_not_reach_s3(tmp_path):
    # audit A67bdcd36a1 F2: the instance is terminated after the run, so a summary.txt that was not uploaded is a
    # lost result, not a success
    _render(tmp_path / "r")
    script = _loop_script((tmp_path / "r" / "vlaeval.sh").read_text(), tmp_path)
    script = script.replace('aws() {{ echo "aws $*" >> "$CALLS"; }}'.replace("{{", "{").replace("}}", "}"),
                            'aws() { echo "aws $*" >> "$CALLS"; case "$*" in *summary.txt*) return 1;; esac; }\n'
                            'sleep() { :; }')
    assert "summary.txt*) return 1" in script
    env = {"PATH": os.environ["PATH"], "STUB_RC": "0", "HOME": str(tmp_path)}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAILED=[ task-0]" in proc.stdout and "upload incomplete (summary.txt 0/1" in proc.stdout
    assert (tmp_path / "calls").read_text().count("/R/task-0/summary.txt") == 3  # retried


# --- batch A: LIBERO-Mem, CALVIN, Kinetix (per-benchmark suite_param / seed_param, docker_env, expect_videos) ------

def _set(name):
    import task_sets
    return task_sets.apply(_model(), name, "vlaeval")


def test_batch_a_sets_render_the_harness_configs():
    benches, servers, tasks = generate.vlaeval_plan(_set("batch-a"))
    assert [b["name"] for b in benches] == ["libero-mem", "calvin", "kinetix"]
    assert all(b["image"].startswith("ghcr.io/allenai/vla-evaluation-harness/") and "@sha256:" in b["image"]
               for b in benches)
    assert [s["name"] for s in servers] == ["lerobot-pi05-libero", "xvla-calvin", "rtc-kinetix"]
    assert servers[1]["hf_repo"] == "2toINF/X-VLA-Calvin-ABC_D" and servers[1]["gated_repos"] == []
    assert servers[2]["hf_repo"] == "" and servers[2]["config"] == "configs/model_servers/rtc/rtc.yaml"
    mem, calvin, kin = tasks
    assert [t["eval_config"]["output_dir"][-6:] for t in tasks] == ["task-0", "task-1", "task-2"]
    # LIBERO-Mem takes suite + seed like LIBERO
    assert mem["eval_config"]["benchmarks"][0]["params"] == {"suite": "libero_mem", "seed": 7, "num_steps_wait": 10}
    assert "env" not in mem["eval_config"]["docker"] and "expect_videos" not in mem
    # CALVIN: no suite (CALVINBenchmark(**params) would raise), the image's EGL env, = configs/benchmarks/calvin/eval.yaml
    c = calvin["eval_config"]
    assert c["benchmarks"][0]["params"] == {"seed": 0, "dataset_path": "/data/calvin/dataset/validation",
                                            "num_sequences": 5}
    assert c["benchmarks"][0]["subname"] == "calvin_abc_d" and c["benchmarks"][0]["episodes_per_task"] == 1
    assert c["docker"]["env"] == ["NVIDIA_DRIVER_CAPABILITIES=all"]
    assert (calvin["checkpoint_arg"], calvin["hf_model_revision"]) == \
        ("model_path", "d76710ee314ee1fa8506f421664c989b40bae415")
    # Kinetix: no suite, a path param, JAX preallocation off in the container, no videos expected
    k = kin["eval_config"]
    assert k["benchmarks"][0]["params"] == {"seed": 0, "observation_type": "symbolic", "max_episode_steps": 256,
                                            "rtc_worlds_dir": "/app/rtc/worlds", "action_noise_std": 0.1}
    assert k["docker"]["env"] == ["XLA_PYTHON_CLIENT_PREALLOCATE=false"] and k["benchmarks"][0]["max_tasks"] == 4
    assert kin["expect_videos"] is False and kin["checkpoint_arg"] == "" and kin["hf_repo"] == ""
    assert "PRNGKey" in kin["init_states"] and "validation set" in calvin["init_states"]
    # each single set is the same entry as in batch-a
    for name, t in (("libero-mem", mem), ("calvin", calvin), ("kinetix", kin)):
        (single,) = generate.vlaeval_plan(_set(name))[2]
        assert single["eval_config"]["benchmarks"] == t["eval_config"]["benchmarks"]
        assert single["eval_config"]["output_dir"].endswith("/task-0")


def test_default_tasks_json_has_no_new_keys():
    # the default LIBERO-Spatial entry renders the same tasks JSON as before suite_param / docker_env / expect_videos
    (t,) = generate.vlaeval_plan(_model())[2]
    assert "expect_videos" not in t and "init_states" not in t and "env" not in t["eval_config"]["docker"]


def test_seed_param_null_injects_nothing():
    cfg = _set("kinetix")
    cfg["vlaeval"]["benchmarks"]["kinetix"]["seed_param"] = None
    (t,) = generate.vlaeval_plan(cfg)[2]
    assert "seed" not in t["eval_config"]["benchmarks"][0]["params"]
    cfg["vlaeval"]["benchmarks"]["kinetix"]["seed_param"] = "base_seed"   # another arg name gets the seed
    (t,) = generate.vlaeval_plan(cfg)[2]
    assert t["eval_config"]["benchmarks"][0]["params"]["base_seed"] == 0


def _bench(c, name="kinetix"):
    return c["vlaeval"]["benchmarks"][name]


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: _bench(c).update(suite_param="Suite-X"), "suite_param"),
    (lambda c: _bench(c).update(seed_param="seed; rm"), "seed_param"),
    (lambda c: _bench(c).update(suite_param="seed"), "same arg"),
    (lambda c: _bench(c).update(docker_env="A=1"), "docker_env must be a list"),
    (lambda c: _bench(c).update(docker_env=["lower=1"]), "docker_env"),
    (lambda c: _bench(c).update(docker_env=["A=b c"]), "docker_env"),
    (lambda c: _bench(c).update(docker_env=["A=$(id)"]), "docker_env"),
    (lambda c: _bench(c).update(docker_env=["A='x'"]), "docker_env"),
    (lambda c: _bench(c).update(docker_env=["A=1", "A=2"]), "set twice"),
    (lambda c: _bench(c).update(docker_env=[1]), "docker_env"),
    (lambda c: _bench(c).update(expect_videos="no"), "expect_videos"),
    (lambda c: _bench(c).update(init_states="it's"), "init_states"),
    (lambda c: c["tasks"][0]["params"].update(seed=1), "params"),                 # injected by the entry
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir="/app/rtc worlds"), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir="/app;id"), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir="/app/$HOME"), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir="/app/it's"), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir="/" + "a" * 256), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update(tasks=[["a"], "b"]), "params.tasks"),      # lists are one level
    (lambda c: c["tasks"][0]["params"].update(tasks=[]), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks=["a b"]), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks=["a", None]), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks=[{"a": 1}]), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks=["$(id)"]), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks=["a"] * 65), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(tasks={"a": 1}), "params.tasks"),
    (lambda c: c["tasks"][0]["params"].update(rtc_worlds_dir=None), "params.rtc_worlds_dir"),
    (lambda c: c["tasks"][0]["params"].update({"Bad": 1}), "params.Bad"),
])
def test_plan_rejects_bad_benchmark_settings(mutate, msg):
    cfg = copy.deepcopy(_set("kinetix"))
    mutate(cfg)
    with pytest.raises(ValueError, match=msg):
        generate.vlaeval_plan(cfg)


def test_param_values_may_be_paths_and_import_strings():
    cfg = copy.deepcopy(_set("kinetix"))
    long_path = "/assets/benchmarks/" + "x" * 230
    cfg["tasks"][0]["params"].update(
        rtc_worlds_dir=long_path, eval_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:Dummy")
    (t,) = generate.vlaeval_plan(cfg)[2]
    p = t["eval_config"]["benchmarks"][0]["params"]
    assert p["rtc_worlds_dir"] == long_path and len(long_path) <= 256 and p["eval_config_cls"].endswith(":Dummy")
    # a suite-less benchmark may name `suite` in its params (nothing else injects it)
    cfg["tasks"][0]["params"]["suite"] = "x"
    assert generate.vlaeval_plan(cfg)[2][0]["eval_config"]["benchmarks"][0]["params"]["suite"] == "x"


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("task_set", ["libero-mem", "calvin", "kinetix", "batch-a"])
def test_batch_a_render_and_bash_n(task_set, tmp_path):
    _render(tmp_path, "--task-set", task_set)
    text = (tmp_path / "vlaeval.sh").read_text()
    assert subprocess.run([BASH, "-n", str(tmp_path / "vlaeval.sh")]).returncode == 0
    tasks = _tasks_json(text)
    assert tasks == json.loads(json.dumps(generate.vlaeval_plan(_set(task_set))[2]))   # survives the bash quoting
    assert ":latest" not in text
    for t in tasks:
        assert f'docker pull -q "{t["image"]}"' in text
    if task_set in ("calvin", "batch-a"):
        assert 'snapshot_download("2toINF/X-VLA-Calvin-ABC_D", revision="d76710ee314ee1fa8506f421664c989b40bae415")' \
            in text
    if task_set == "kinetix":
        assert 'snapshot_download("' not in text and "gated google" not in text   # RTC fetches its own checkpoints


# --- adapter: CALVIN average chain length --------------------------------------------------------------------------

def test_adapter_reports_calvin_avg_len_and_chain_rates(tmp_path):
    out = tmp_path / "out"
    (out / "episodes" / "CALVINBenchmark_calvin_abc_d").mkdir(parents=True)
    shutil.copy(CALVIN, out / "CALVINBenchmark_calvin_abc_d_aggregate.json")
    (out / "episodes" / "CALVINBenchmark_calvin_abc_d" / "task0000_ep0000_success.mp4").write_bytes(b"x")
    summary = tmp_path / "s.txt"
    rc = vlaeval_results.main(["--output-dir", str(out), "--summary", str(summary), "--video-dir",
                               str(tmp_path / "v"), "--suite", "calvin_abc_d",
                               "--init-states", "sequences drawn from the validation set with params.seed"])
    assert rc == 0
    kv = dict(results.KV_RE.findall(summary.read_text()))
    # = docs/reproductions/xvla.md of the harness: 1/5 95.6 % ... 5/5 73.9 %, avg len 4.30
    assert (kv["n_success"], kv["n_episodes"], kv["n_tasks"]) == ("739", "1000", "1000")
    assert kv["calvin_avg_len"] == "4.301" and kv["harness_mean_completed_subtasks"] == "4.301"
    assert [kv[f"calvin_chain_{i}_of_5"] for i in range(1, 6)] == ["0.956", "0.918", "0.871", "0.817", "0.739"]
    assert kv["calvin_n_sequences"] == "1000" and kv["init_states"].startswith("sequences drawn")
    assert kv["vla_eval_benchmark"] == "CALVINBenchmark" and kv["n_videos"] == "1"


def test_adapter_adds_nothing_for_libero(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    shutil.copy(SP50S4, out / "LIBEROBenchmark_libero_spatial_aggregate.json")
    vlaeval_results.main(["--output-dir", str(out), "--summary", str(tmp_path / "s.txt"), "--video-dir",
                          str(tmp_path / "v"), "--suite", "libero_spatial", "--init-states", ""])
    text = (tmp_path / "s.txt").read_text()
    assert "calvin_" not in text and "harness_mean_completed" not in text
    assert "init_states: fixed (benchmark init-state files, episode i = init state i)" in text


def test_adapter_calvin_lines_need_the_metric():
    agg = vlaeval_results.load(CALVIN)
    assert vlaeval_results.calvin_lines(dict(agg, metric_keys={"success": "mean"})) == []
    assert vlaeval_results.metric_lines(dict(agg, metric_keys={"success": "mean"})) == []
    no_eps = dict(agg, tasks=[{"task": "seq", "episodes": [{"metrics": {"success": False}}]}])
    assert vlaeval_results.calvin_lines(no_eps) == []


# --- the task loop over three servers ---------------------------------------------------------------------------

STUB_MULTI = """#!/bin/bash
# stub of `vla-eval` for several benchmarks: run = the fixture aggregate under the entry's own file name; videos only
# where the benchmark records frames (not Kinetix symbolic) and not for $NO_VIDEO_BENCH
echo "vla-eval $*" >> "$CALLS"
if [ "$1" = serve ]; then
    case "$3" in *"${STUB_SERVE_FAIL:-none}"*) echo "RuntimeError: NVIDIA driver too old"; exit 1 ;; esac
    exec sleep 300
fi
cfg="$3"
dest="$EVAL_BASE_DIR/$(basename "$(jq -r .output_dir "$cfg")")"
b=$(jq -r '.benchmarks[0].benchmark | split(":")[1]' "$cfg")_$(jq -r '.benchmarks[0].subname' "$cfg")
mkdir -p "$dest/episodes/$b"
cp "$FIXTURE" "$dest/${b}_aggregate.json"
case "$b" in Kinetix*|"$NO_VIDEO_BENCH") ;; *) echo x > "$dest/episodes/$b/task0000_ep0000_success.mp4" ;; esac
exit 0
"""


def _batch_loop(tmp_path, curl_body):
    _render(tmp_path / "r", "--task-set", "batch-a")
    script = _loop_script((tmp_path / "r" / "vlaeval.sh").read_text(), tmp_path)
    (tmp_path / "work" / "venv" / "bin" / "vla-eval").write_text(STUB_MULTI)
    script = script.replace("curl() { return 0; }", "curl() { " + curl_body + " }")
    assert curl_body in script
    return script


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
@pytest.mark.parametrize("no_video", ["", "LIBEROMemBenchmark_libero_mem"])
def test_batch_a_loop_switches_servers(tmp_path, no_video):
    # health = the current server process is alive, so a stopped server stops answering
    script = _batch_loop(tmp_path, '[ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null;')
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "NO_VIDEO_BENCH": no_video}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    serves = [ln for ln in (tmp_path / "calls").read_text().splitlines() if ln.startswith("vla-eval serve")]
    assert serves == [
        "vla-eval serve --config configs/model_servers/lerobot/pi05_libero.yaml --arg checkpoint=/hf/snapshots/7152f13e",
        "vla-eval serve --config configs/model_servers/xvla/calvin.yaml --arg model_path=/hf/snapshots/d76710ee",
        "vla-eval serve --config configs/model_servers/rtc/rtc.yaml"]      # RTC: no checkpoint arg
    cfgs = [json.loads((tmp_path / "work" / f"eval-task-{i}.json").read_text()) for i in range(3)]
    assert [c["docker"].get("env") for c in cfgs] == \
        [None, ["NVIDIA_DRIVER_CAPABILITIES=all"], ["XLA_PYTHON_CLIENT_PREALLOCATE=false"]]
    kv = dict(results.KV_RE.findall((tmp_path / "vlaeval_task2_summary.txt").read_text()))
    assert kv["init_states"].startswith("episode i = PRNGKey") and kv["checkpoint"] == "from server config@"
    assert "no episode videos expected" in proc.stdout
    # Kinetix without videos is not a failure; LIBERO-Mem without videos still is
    assert ("FAILED=[ task-0]" if no_video else "FAILED=[]") in proc.stdout


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
def test_batch_a_loop_stops_when_the_old_server_keeps_answering(tmp_path):
    script = _batch_loop(tmp_path, "return 0;")      # something still answers /health after the stop
    script = script.replace("sleep 2\n", ":\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "NO_VIDEO_BENCH": ""}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 9 and "DIE model server lerobot-pi05-libero still answers on :8000" in proc.stdout
    serves = [ln for ln in (tmp_path / "calls").read_text().splitlines() if ln.startswith("vla-eval serve")]
    assert len(serves) == 1    # the CALVIN server was never started against a stale health check


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
def test_batch_a_loop_continues_past_a_server_that_does_not_start(tmp_path):
    # measured 2026-11-09 batch A: the X-VLA server died at model load and the whole run stopped before Kinetix
    # health = alive and not the failing server (start_server's local _config is visible here: bash scoping);
    # a process-alive check alone races the stub's exit
    script = _batch_loop(tmp_path, '[ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null'
                                   ' && local _c="${_config:-}" && [ "${_c}" = "${_c#*xvla/}" ];')
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "NO_VIDEO_BENCH": "", "STUB_SERVE_FAIL": "xvla/"}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DIE" not in proc.stdout
    calls = (tmp_path / "calls").read_text().splitlines()
    assert [ln.split("--config ")[1].split()[0] for ln in calls if ln.startswith("vla-eval serve")] == [
        "configs/model_servers/lerobot/pi05_libero.yaml", "configs/model_servers/xvla/calvin.yaml",
        "configs/model_servers/rtc/rtc.yaml"]
    assert sum(ln.startswith("vla-eval run") for ln in calls) == 2      # CALVIN never reached `run`
    assert "FAILED=[ task-1]" in proc.stdout and "model server did not start" in proc.stdout
    assert any("s3://b/R/task-1/server.log" in ln for ln in calls)
    assert (tmp_path / "vlaeval_task2_summary.txt").exists()            # Kinetix still ran and reported


def _constraint_block(text):
    start = text.index('TORCH_CONSTRAINT="none"\n')
    end = text.index("\nfi\n", start) + 4
    return text[start:end]


HEADER_OPEN = '# /// script\n# dependencies = [\n#     "torch>=2.2",\n# ]\n#\n# [tool.uv]\n# exclude-newer = "x"\n# ///\n'
HEADER_CU128 = '# /// script\n# dependencies = [\n#     "torch>=2.7,<2.12",\n# ]\n#\n# [tool.uv]\n# exclude-newer = "x"\n# ///\n'
HEADER_OWN = '# /// script\n# [tool.uv]\n# constraint-dependencies = ["numpy<2"]\n# ///\n'
# stub of `uv lock --script`: an open torch pin locks CUDA 13 wheels (as torch 2.15 on PyPI), a capped one cu128
STUB_UV = """uv() {
    echo "uv $*" >> "$CALLS"
    local f="${@: -1}"
    if grep -q '"torch>=2.2",' "$f" && ! grep -q 'torch<2.11' "$f"; then
        printf '[[package]]\\nname = "nvidia-cudnn-cu13"\\n' > "$f.lock"
    else
        printf '[[package]]\\nname = "torch"\\nversion = "2.11.0+cu128"\\n' > "$f.lock"
    fi
}
"""


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("driver,expect", [("570.133.20", "torch<2.11 (xvla.py)"), ("595.112.07", "none"),
                                           ("", "none")])
def test_torch_constraint_only_where_the_lock_needs_cuda13(tmp_path, driver, expect):
    # torch >= 2.11 PyPI wheels need CUDA 13 (driver >= 580); xvla.py leaves torch open (measured 2026-11-09),
    # lerobot.py locks 2.11.0+cu128 and must stay as published
    _render(tmp_path / "r", "--task-set", "batch-a")
    block = _constraint_block((tmp_path / "r" / "vlaeval.sh").read_text())
    h = tmp_path / "h"
    ms = h / "src" / "vla_eval" / "model_servers"
    ms.mkdir(parents=True)
    (ms / "xvla.py").write_text(HEADER_OPEN)
    (ms / "lerobot.py").write_text(HEADER_CU128)
    (ms / "rtc.py").write_text(HEADER_OWN)
    (ms / "unused.py").write_text(HEADER_OPEN)
    cfg = h / "configs" / "model_servers"
    for rel, body in (("lerobot/_base.yaml", 'script: "src/vla_eval/model_servers/lerobot.py"\n'),
                      ("lerobot/pi05_libero.yaml", "extends: _base.yaml\n"),
                      ("xvla/_base.yaml", 'script: "src/vla_eval/model_servers/xvla.py"\n'),
                      ("xvla/calvin.yaml", "extends: _base.yaml\n"),
                      ("rtc/rtc.yaml", 'script: "src/vla_eval/model_servers/rtc.py"\n')):
        (cfg / rel).parent.mkdir(parents=True, exist_ok=True)
        (cfg / rel).write_text(body)
    script = (f'set -euo pipefail\nHARNESS_DIR={h}; VENV={Path(sys.executable).parent.parent}; '
              f'DRIVER_VERSION="{driver}"; CALLS={tmp_path}/calls\nlog() {{ :; }}\ndie() {{ echo "DIE $*"; exit 9; }}\n'
              + STUB_UV + block + 'echo "TC1=${TORCH_CONSTRAINT}"\n' + block + 'echo "TC=${TORCH_CONSTRAINT}"\n')
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"TC1={expect}" in proc.stdout and f"TC={expect}" in proc.stdout   # a re-run still records it
    xvla = (ms / "xvla.py").read_text()
    if expect == "none":
        assert xvla == HEADER_OPEN and not (tmp_path / "calls").exists()
    else:
        assert xvla.count('# constraint-dependencies = ["torch<2.11"]') == 1      # a re-run does not stack it
        assert xvla.index("# [tool.uv]") < xvla.index("constraint-dependencies") < xvla.index("exclude-newer")
        locked = (tmp_path / "calls").read_text()
        assert "lerobot.py" in locked and "unused.py" not in locked and "rtc.py" not in locked
    assert (ms / "lerobot.py").read_text() == HEADER_CU128          # resolves cu128 on its own
    assert (ms / "rtc.py").read_text() == HEADER_OWN and (ms / "unused.py").read_text() == HEADER_OPEN
    assert not list(ms.glob("*.lock"))                               # `uv run` resolves from the header as before


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_torch_constraint_stops_when_a_server_config_does_not_resolve(tmp_path):
    _render(tmp_path / "r", "--task-set", "batch-a")
    block = _constraint_block((tmp_path / "r" / "vlaeval.sh").read_text())
    cfg = tmp_path / "h" / "configs" / "model_servers" / "xvla"
    cfg.mkdir(parents=True)
    (cfg / "calvin.yaml").write_text("extends: _base.yaml\n")      # _base.yaml missing
    script = (f'set -euo pipefail\nHARNESS_DIR={tmp_path}/h; VENV={Path(sys.executable).parent.parent}; '
              f'DRIVER_VERSION="570.133.20"; CALLS={tmp_path}/calls\nlog() {{ :; }}\ndie() {{ echo "DIE $*"; exit 9; }}\n'
              + STUB_UV + block + 'echo "TC=${TORCH_CONSTRAINT}"\n')
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 9 and "DIE could not resolve the model-server scripts" in proc.stdout
    assert not (tmp_path / "calls").exists()


# --- batch B: RoboCerebra, MolmoSpaces x MolmoBot, ManiSkill2 x DB-CogACT ----------------------------------------
# (per-server checkpoint_marker / hf_allow_patterns / checkpoint_pass / hf_offline_repos, list params, render)

MOLMO_SHA = "cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9"
QWEN_SHA = "cdbee75f17c01a7cc42f958dc650907174af0554"
COGACT_SHA = "d4af8d4e8a07ef4c9f982f52d446d14f2039e81c"


def test_batch_b_sets_render_the_harness_configs():
    benches, servers, tasks = generate.vlaeval_plan(_set("batch-b"))
    assert [b["name"] for b in benches] == ["robocerebra", "molmospaces", "maniskill2"]
    assert benches[0]["image"] == ("ghcr.io/allenai/vla-evaluation-harness/robocerebra@sha256:"
                                   "89ad3218a46e1b3d2e4926ef788a9d3bce312bbe900c37e75a27b93e1be80c34")
    # the allenai molmospaces package is private: the harness's own pull fallback (cli/_docker.py), by digest
    assert benches[1]["image"] == ("ghcr.io/worv-ai/vla-evaluation-harness-public/molmospaces@sha256:"
                                   "ca13e61b0b67031298c4e1347c33e0b853abc35caaad4169961a3b14ea07e141")
    assert benches[2]["image"] == ("ghcr.io/allenai/vla-evaluation-harness/maniskill2@sha256:"
                                   "3c4d35caae55a3eec6daa608fcb095354bbbde8fd905b6d714d40bb9d8192191")
    assert [s["name"] for s in servers] == ["lerobot-pi05-libero", "molmobot-droid", "db-cogact-maniskill2"]
    lerobot, molmo, cogact = servers
    assert set(lerobot) == {"name", "config", "hf_repo", "hf_model_revision", "gated_repos"}   # no new keys
    assert molmo == dict(name="molmobot-droid", config="configs/model_servers/molmobot/droid.yaml",
                         hf_repo="allenai/MolmoBot-DROID", hf_model_revision=MOLMO_SHA, gated_repos=[],
                         checkpoint_marker="model.pt", checkpoint_pass="repo_id",
                         hf_offline_repos=[dict(repo="Qwen/Qwen3-4B-Instruct-2507", revision=QWEN_SHA,
                                                allow_patterns=["*.json", "*.txt"])])
    assert cogact == dict(name="db-cogact-maniskill2", config="configs/model_servers/db_cogact/maniskill2.yaml",
                          hf_repo="Dexmal/maniskill2-db-cogact", hf_model_revision=COGACT_SHA, gated_repos=[])
    cer, molmo_t, ms2 = tasks
    assert [t["eval_config"]["output_dir"][-6:] for t in tasks] == ["task-0", "task-1", "task-2"]
    # RoboCerebra: seed but no suite (= configs/benchmarks/robocerebra/eval.yaml params), a list param
    b = cer["eval_config"]["benchmarks"][0]
    assert b["params"] == {"seed": 7, "robocerebra_root": "/workspace/RoboCerebra_Bench", "task_types": ["Ideal"]}
    assert (b["max_tasks"], b["episodes_per_task"]) == (2, 2) and "runtime" not in cer["eval_config"]["docker"]
    assert cer["model_server"] == "lerobot-pi05-libero" and "checkpoint_pass" not in cer
    # MolmoSpaces: neither suite nor seed, the long path and module:Class params of pick_and_place.yaml
    b = molmo_t["eval_config"]["benchmarks"][0]
    assert b["params"] == {
        "benchmark_dir": "/assets/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/"
                         "FrankaPickandPlaceHardBench_20260212_200ep_json_benchmark",
        "eval_config_cls": "molmo_spaces.evaluation.configs.evaluation_configs:DummyPickPlaceEvalConfig",
        "task_horizon": 600, "send_wrist_image": True, "send_state": True}
    assert (b["max_tasks"], b["max_steps"], b["episodes_per_task"]) == (4, 600, 1)
    assert molmo_t["checkpoint_pass"] == "repo_id" and molmo_t["checkpoint_arg"] == "hf_repo"
    # ManiSkill2: no suite / seed, the task subset and resolution as lists (= configs/benchmarks/maniskill2/eval.yaml)
    b = ms2["eval_config"]["benchmarks"][0]
    assert b["params"] == {"tasks": ["PickCube-v0", "StackCube-v0"], "max_episode_steps": 400,
                           "render_resolution": [256, 256], "enabled_cameras": ["base_camera"]}
    assert (b["episodes_per_task"], b["max_steps"]) == (3, 400) and "max_tasks" not in b
    assert (ms2["checkpoint_arg"], ms2["hf_model_revision"]) == ("model_path", COGACT_SHA)
    for t in tasks:
        assert "render" not in t["eval_config"] and "env" not in t["eval_config"]["docker"]
        assert t.get("expect_videos", True) is True and t["init_states"]
    # each single set is the same entry as in batch-b
    for name, t in (("robocerebra", cer), ("molmospaces", molmo_t), ("maniskill2", ms2)):
        (single,) = generate.vlaeval_plan(_set(name))[2]
        assert single["eval_config"]["benchmarks"] == t["eval_config"]["benchmarks"]
        assert single["eval_config"]["output_dir"].endswith("/task-0")


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
@pytest.mark.parametrize("task_set", ["robocerebra", "molmospaces", "maniskill2", "batch-b"])
def test_batch_b_render_and_bash_n(task_set, tmp_path):
    _render(tmp_path, "--task-set", task_set)
    text = (tmp_path / "vlaeval.sh").read_text()
    assert subprocess.run([BASH, "-n", str(tmp_path / "vlaeval.sh")]).returncode == 0
    tasks = _tasks_json(text)
    assert tasks == json.loads(json.dumps(generate.vlaeval_plan(_set(task_set))[2]))
    assert ":latest" not in text
    for t in tasks:
        assert f'docker pull -q "{t["image"]}"' in text
    molmo = task_set in ("molmospaces", "batch-b")
    assert (f'_p = snapshot_download("allenai/MolmoBot-DROID", revision="{MOLMO_SHA}")' in text) == molmo
    assert ('[ -f "${_path}/model.pt" ] || die "allenai/MolmoBot-DROID snapshot ${_path} has no model.pt"'
            in text) == molmo
    assert (f'joinpath("refs", "main").write_text("{MOLMO_SHA}")' in text) == molmo
    assert (f'snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="{QWEN_SHA}", '
            'allow_patterns=["*.json", "*.txt"])' in text) == molmo
    assert ("export HF_HUB_OFFLINE=1" in text) == molmo and ('"${CKPT_PASS}"; then' in text) == molmo
    if task_set in ("maniskill2", "batch-b"):
        assert f'print(snapshot_download("Dexmal/maniskill2-db-cogact", revision="{COGACT_SHA}"))' in text
        assert '[ -f "${_path}/config.json" ] || die "Dexmal/maniskill2-db-cogact snapshot' in text
    if task_set == "robocerebra":   # the default server, nothing new rendered
        assert "refs" not in text and "allow_patterns" not in text and "CKPT_PASS" not in text


def test_default_render_has_no_batch_b_lines(tmp_path):
    # the new lines render only for a server that sets the keys; the LIBERO and batch A renders of every target
    # were also diffed by hand against the pre-batch-B renders (identical but for the tool_commit line)
    for extra in ([], ["--task-set", "batch-a"]):
        _render(tmp_path, *extra)
        text = (tmp_path / "vlaeval.sh").read_text()
        for needle in ("HF_HUB_OFFLINE", "refs", "allow_patterns", "CKPT_PASS", "_pass=", "offline repo"):
            assert needle not in text, (extra, needle)
        assert text.count('has no config.json"') == (2 if extra else 1)


def _server(c, name="molmobot-droid"):
    return c["vlaeval"]["model_servers"][name]


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: _server(c).update(checkpoint_marker="../model.pt"), "checkpoint_marker"),
    (lambda c: _server(c).update(checkpoint_marker="/abs/model.pt"), "checkpoint_marker"),
    (lambda c: _server(c).update(checkpoint_marker="a b"), "checkpoint_marker"),
    (lambda c: _server(c).update(checkpoint_pass="url"), "checkpoint_pass must be"),
    (lambda c: _server(c).update(hf_allow_patterns=["*.json"]), "whole snapshot"),     # repo_id: no partial
    (lambda c: _server(c).update(checkpoint_pass="path"), "only with checkpoint_pass repo_id"),
    (lambda c: _server(c).update(hf_offline_repos=[]), "non-empty list"),
    (lambda c: _server(c).update(hf_offline_repos={"repo": "a/b"}), "non-empty list"),
    (lambda c: _server(c)["hf_offline_repos"][0].update(revision="main"), r"hf_offline_repos\[0\].revision"),
    (lambda c: _server(c)["hf_offline_repos"][0].update(repo="nope"), r"hf_offline_repos\[0\].repo"),
    (lambda c: _server(c)["hf_offline_repos"][0].update(typo=1), r"hf_offline_repos\[0\] must be a mapping"),
    (lambda c: _server(c)["hf_offline_repos"][0].update(allow_patterns=["a b"]), "allow_patterns"),
    (lambda c: _server(c)["hf_offline_repos"][0].update(allow_patterns="*.json"), "allow_patterns"),
    (lambda c: _server(c, "db-cogact-maniskill2").update(hf_allow_patterns=[]), "hf_allow_patterns"),
    (lambda c: _server(c, "db-cogact-maniskill2").update(hf_allow_patterns=["../x"]), "hf_allow_patterns"),
    (lambda c: _server(c, "db-cogact-maniskill2").update(hf_allow_patterns=["a'b"]), "hf_allow_patterns"),
    (lambda c: _server(c, "db-cogact-maniskill2").update(checkpoint_pass="repo_id", hf_repo=None,
                                                         hf_model_revision=None), "needs hf_repo"),
    (lambda c: _bench(c, "maniskill2").update(render="egl"), "render must be gpu or cpu"),
    (lambda c: _server(c)["hf_offline_repos"].append(dict(repo="allenai/MolmoBot-DROID", revision="0" * 40)),
     "shared refs/main"),
])
def test_plan_rejects_bad_checkpoint_settings(mutate, msg):
    cfg = copy.deepcopy(_set("batch-b"))
    mutate(cfg)
    with pytest.raises(ValueError, match=msg):
        generate.vlaeval_plan(cfg)


def test_marker_and_patterns_need_a_checkpoint():
    cfg = copy.deepcopy(_set("kinetix"))
    _server(cfg, "rtc-kinetix")["checkpoint_marker"] = "x.pkl"
    with pytest.raises(ValueError, match="need hf_repo"):
        generate.vlaeval_plan(cfg)


def test_list_params_reach_the_run_config():
    cfg = copy.deepcopy(_set("maniskill2"))
    cfg["tasks"][0]["params"]["tasks"] = ["PickCube-v0"] * 64          # at the limit
    (t,) = generate.vlaeval_plan(cfg)[2]
    assert t["eval_config"]["benchmarks"][0]["params"]["tasks"] == ["PickCube-v0"] * 64


@pytest.mark.parametrize("value", ["cpu", "gpu"])
def test_render_key_is_the_run_level_render(value):
    cfg = copy.deepcopy(_set("batch-b"))
    _bench(cfg, "maniskill2")["render"] = value
    cer, _, ms2 = generate.vlaeval_plan(cfg)[2]
    assert ms2["eval_config"]["render"] == value and "render" not in cer["eval_config"]


def test_allow_patterns_and_marker_reach_the_download():
    cfg = copy.deepcopy(_set("calvin"))
    _server(cfg, "xvla-calvin").update(hf_allow_patterns=["*.json", "*.safetensors"], checkpoint_marker="model.safetensors")
    text = generate.generate_vlaeval(cfg, dry_run=True)
    assert ('snapshot_download("2toINF/X-VLA-Calvin-ABC_D", revision="d76710ee314ee1fa8506f421664c989b40bae415", '
            'allow_patterns=["*.json", "*.safetensors"]))') in text
    assert '[ -f "${_path}/model.safetensors" ] || die' in text and 'has no config.json' not in text
    assert "HF_HUB_OFFLINE" not in text and "refs" not in text


STUB_HF = """import os, pathlib
def snapshot_download(repo, revision=None, allow_patterns=None):
    snap = pathlib.Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    if os.environ.get("STUB_MARKER") == "1":
        (snap / "model.pt").write_text("x")
    with open(os.environ["CALLS"], "a") as f:
        f.write(f"snapshot_download {repo} {revision} {allow_patterns}\\n")
    return str(snap)
"""


def _download_block(tmp_path, marker):
    _render(tmp_path / "r", "--task-set", "molmospaces")
    text = (tmp_path / "r" / "vlaeval.sh").read_text()
    start = text.index("export HF_XET_CLIENT_READ_TIMEOUT")
    end = text.index("# Model server: `vla-eval serve`", start)
    stub = tmp_path / "stub"
    (stub / "huggingface_hub").mkdir(parents=True, exist_ok=True)
    (stub / "huggingface_hub" / "__init__.py").write_text(STUB_HF)
    body = text[start:end].replace("/tmp/vlaeval_", f"{tmp_path}/vlaeval_")
    script = (f'set -euo pipefail\nexport HF_HOME={tmp_path}/hf CALLS={tmp_path}/calls PYTHONPATH={stub} '
              f'STUB_MARKER={marker}\nlog() {{ echo "LOG $*"; }}\ndie() {{ echo "DIE $*"; exit 9; }}\n'
              f'uv() {{ while [ "$1" != python ]; do shift; done; shift; "{sys.executable}" "$@"; }}\n'
              # `timeout` would exec the real uv binary (not the function above), which CI does not have
              f'timeout() {{ while [ "$1" != uv ]; do shift; done; "$@"; }}\n' + body)
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_repo_id_download_points_main_at_the_pin(tmp_path):
    # [6/8] for MolmoBot through bash, `uv run` replaced by this Python with a stub huggingface_hub: the checkpoint
    # and the Qwen3 tokenizer are fetched at their pins and refs/main of each names the pin, which is what the
    # server's offline snapshot_download(<id>) / AutoTokenizer.from_pretrained(<id>) resolve (checked by hand on
    # 2026-11-09 with huggingface_hub 1.33.0 + transformers 5.22.1, the versions molmobot.py locks to)
    proc = _download_block(tmp_path, marker=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hub = tmp_path / "hf" / "hub"
    assert (hub / "models--allenai--MolmoBot-DROID" / "refs" / "main").read_text() == MOLMO_SHA
    assert (hub / "models--Qwen--Qwen3-4B-Instruct-2507" / "refs" / "main").read_text() == QWEN_SHA
    assert (tmp_path / "calls").read_text().splitlines() == [
        f"snapshot_download allenai/MolmoBot-DROID {MOLMO_SHA} None",
        f"snapshot_download Qwen/Qwen3-4B-Instruct-2507 {QWEN_SHA} ['*.json', '*.txt']"]
    path = (tmp_path / "vlaeval_ckpt_molmobot-droid.path").read_text().strip()
    assert path.endswith(f"models--allenai--MolmoBot-DROID/snapshots/{MOLMO_SHA}") and "DIE" not in proc.stdout


@pytest.mark.skipif(BASH is None, reason="bash not on PATH")
def test_repo_id_download_stops_without_the_marker(tmp_path):
    proc = _download_block(tmp_path, marker=0)
    assert proc.returncode == 9 and "DIE allenai/MolmoBot-DROID snapshot" in proc.stdout and "has no model.pt" in proc.stdout
    assert "Qwen" not in (tmp_path / "calls").read_text()               # stopped before the next download


STUB_B = STUB_MULTI.replace('if [ "$1" = serve ]; then\n',
                            'if [ "$1" = serve ]; then\n    echo "serve-env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}" >> "$CALLS"\n')


@pytest.mark.skipif(None in (BASH, JQ, SETSID), reason="bash / jq / setsid not on PATH")
def test_batch_b_loop_passes_the_repo_id_offline(tmp_path):
    _render(tmp_path / "r", "--task-set", "batch-b")
    script = _loop_script((tmp_path / "r" / "vlaeval.sh").read_text(), tmp_path)
    assert STUB_B != STUB_MULTI
    (tmp_path / "work" / "venv" / "bin" / "vla-eval").write_text(STUB_B)
    ms = tmp_path / "work" / "harness" / "configs" / "model_servers"
    for rel in ("molmobot/droid.yaml", "db_cogact/maniskill2.yaml"):
        (ms / rel).parent.mkdir(parents=True, exist_ok=True)
        (ms / rel).write_text("extends: _base.yaml\nargs:\n  chunk_size: 10\n")
    (tmp_path / "ckpt_molmobot-droid.path").write_text(f"/hf/snapshots/{MOLMO_SHA[:8]}\n")
    (tmp_path / "ckpt_db-cogact-maniskill2.path").write_text(f"/hf/snapshots/{COGACT_SHA[:8]}\n")
    script = script.replace("curl() { return 0; }",
                            'curl() { [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; }')
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "NO_VIDEO_BENCH": ""}
    proc = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [ln for ln in (tmp_path / "calls").read_text().splitlines()
             if ln.startswith(("serve-env", "vla-eval serve"))]
    assert calls == [
        "vla-eval serve --config configs/model_servers/lerobot/pi05_libero.yaml --arg checkpoint=/hf/snapshots/7152f13e",
        "serve-env HF_HUB_OFFLINE=unset",
        "vla-eval serve --config configs/model_servers/molmobot/droid.yaml --arg hf_repo=allenai/MolmoBot-DROID",
        "serve-env HF_HUB_OFFLINE=1",
        "vla-eval serve --config configs/model_servers/db_cogact/maniskill2.yaml --arg model_path=/hf/snapshots/d4af8d4e",
        "serve-env HF_HUB_OFFLINE=unset"]
    assert "FAILED=[]" in proc.stdout and "HF_HUB_OFFLINE=1: allenai/MolmoBot-DROID" in proc.stdout
    cfgs = [json.loads((tmp_path / "work" / f"eval-task-{i}.json").read_text()) for i in range(3)]
    assert cfgs[0]["benchmarks"][0]["params"]["task_types"] == ["Ideal"]
    assert cfgs[2]["benchmarks"][0]["params"]["render_resolution"] == [256, 256]
    kv = dict(results.KV_RE.findall((tmp_path / "vlaeval_task1_summary.txt").read_text()))
    assert kv["checkpoint"] == f"allenai/MolmoBot-DROID@{MOLMO_SHA[:8]}" and kv["init_states"].startswith("task i")
    card = yaml.safe_load((tmp_path / "vlaeval_task1_summary.protocol.yaml").read_text())
    assert {"repo": "Qwen/Qwen3-4B-Instruct-2507", "revision": QWEN_SHA} in card["checkpoints"]


def test_plan_rejects_two_repo_id_pins_of_one_repo_across_servers():
    cfg = copy.deepcopy(_set("batch-b"))
    servers = cfg["vlaeval"]["model_servers"]
    servers["molmobot-droid-b"] = copy.deepcopy(servers["molmobot-droid"])
    servers["molmobot-droid-b"]["hf_model_revision"] = "0" * 40
    task = next(t for t in cfg["tasks"] if t["model_server"] == "molmobot-droid")
    cfg["tasks"].append(dict(task, model_server="molmobot-droid-b"))
    with pytest.raises(ValueError, match="shared refs/main"):
        generate.vlaeval_plan(cfg)
    # same pin twice is fine: refs/main gets the same commit
    servers["molmobot-droid-b"]["hf_model_revision"] = servers["molmobot-droid"]["hf_model_revision"]
    generate.vlaeval_plan(cfg)
