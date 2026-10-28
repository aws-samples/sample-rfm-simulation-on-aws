#!/usr/bin/env python3
"""flora_judge.py: map a finished run's rollout videos to episodes, and turn Flora scores into judge.json.

Runs on the flora-judge instance (stdlib only; generate.py embeds it into the UserData) and is
imported by the unit tests. It never runs the model itself: scoring is done by NVIDIA's own
``batch_utils.terminal_reward`` from the Flora-Rewarder repository, downloaded at run time.

    python3 flora_judge.py prepare --run-dir DIR --jobs jobs.jsonl --manifest manifest.json \
        [--libero-task-map libero_suite_task_map.py]
    python3 flora_judge.py report --manifest manifest.json --scores terminal.jsonl \
        --json judge.json --txt judge.txt [--meta key=value ...]
    python3 flora_judge.py captions --manifest manifest.json --report judge.json --run-dir DIR

DIR is a local copy of s3://<results-bucket>/<run_id>/ (task-N/videos/*.mp4 + *.txt sidecars,
task-N/eval_info.json, task-N/summary.txt). Per episode it finds:

  task instruction  sidecar "Task:" line > "--task=" in the file name (OpenVLA-OFT style) >
                    LIBERO task map for LeRobot videos (<suite>_<id>--eval_episode_N.mp4)
  sim success       "success=True|False" in the file name > sidecar "result:" line >
                    LeRobot eval_info.json per_task[].metrics.successes (matched via video_paths)

An episode without an instruction is not scored (Flora needs one); an episode without a sim
success is scored but left out of the agreement figures.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# Flora outcome score: 1 = complete failure ... 5 = complete success (model card). Scores at or
# above this threshold count as "Flora says success" when compared with the simulator's flag.
# 4 = "mostly or fully done": a 3 is the model's midpoint and is treated as failure.
SUCCESS_THRESHOLD = 4
SCHEMA_VERSION = 1

_SUCCESS_IN_NAME = re.compile(r"success[=_-](true|false)", re.I)
_TASK_IN_NAME = re.compile(r"--task=(.+)$")
_SIDECAR_TASK = re.compile(r"^Task:\s*(.+?)\s*$", re.M)
_SIDECAR_RESULT = re.compile(r"^result:\s*(true|false)\b", re.M | re.I)
_LEROBOT_GROUP = re.compile(r"^(?P<suite>[a-z][a-z0-9]*(?:_[a-z0-9]+)*?)_(?P<task_id>\d+)$")
_SCORE_TEXT = re.compile(r"Score:\s*([1-5])\s*$")
_TASK_DIR = re.compile(r"^task-(\d+)$")


# ───────────────────────────── instructions ─────────────────────────────


def libero_languages(task_map_source: str) -> dict:
    """{suite: [language, ...]} from LIBERO's libero_suite_task_map.py, parsed without executing it.

    Same rule as LIBERO's grab_language_from_filename (benchmark/__init__.py) with task order 0,
    the order lerobot-eval's LiberoEnv uses for --env.task_ids.
    """
    tree = ast.parse(task_map_source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map"
                                                for t in node.targets):
            task_map = ast.literal_eval(node.value)
            break
    else:
        raise ValueError("libero_task_map not found")
    out = {}
    for suite, names in task_map.items():
        out[suite] = [_libero_language(n) for n in names]
    return out


def _libero_language(name: str) -> str:
    if name[:1].isupper():  # LIBERO-90/-10 names start with the scene, e.g. KITCHEN_SCENE3_turn_on_...
        cut = name.find("SCENE") + (8 if "SCENE10" in name else 7)
        return " ".join(name[cut:].split("_"))
    return " ".join(name.split("_"))


def _humanize(task: str) -> str:
    task = task.strip().strip('"').strip()
    if " " not in task and ("_" in task or "/" in task):
        task = task.rsplit("/", 1)[-1].replace("_", " ")
    return task


def sidecar_task(text: str):
    """Instruction from a video's .txt sidecar ("Task: ..."); openarm writes 'Task: NAME — "instr"'."""
    if not text:
        return None
    m = _SIDECAR_TASK.search(text)
    if not m:
        return None
    value = m.group(1)
    quoted = re.search(r'"([^"]+)"', value)
    return _humanize(quoted.group(1) if quoted else value) or None


def sidecar_result(text: str):
    if not text:
        return None
    m = _SIDECAR_RESULT.search(text)
    return None if not m else m.group(1).lower() == "true"


# ───────────────────────────── episodes ─────────────────────────────────


def split_lerobot_name(filename: str):
    """'libero_spatial_0--eval_episode_3.mp4' -> ('libero_spatial_0', 'eval_episode_3.mp4');
    a flat 'eval_episode_3.mp4' (runs before the per-task prefix) -> (None, 'eval_episode_3.mp4')."""
    if "--" in filename and filename.split("--", 1)[1].startswith("eval_episode_"):
        group, base = filename.split("--", 1)
        return group, base
    return None, filename


def lerobot_episodes(eval_info: dict) -> list:
    """[(group_dir, video_basename, success, suite, task_id)] from a LeRobot eval_info.json.

    video_paths[i] is the i-th rendered episode of the task; episodes are rendered in order, so it
    pairs with successes[i] (lerobot_eval.py renders the first max_episodes_rendered episodes)."""
    out = []
    for entry in (eval_info or {}).get("per_task") or []:
        metrics = entry.get("metrics") or {}
        successes = metrics.get("successes") or []
        for i, path in enumerate(metrics.get("video_paths") or []):
            p = Path(str(path))
            success = bool(successes[i]) if i < len(successes) else None
            out.append((p.parent.name, p.name, success, entry.get("task_group"), entry.get("task_id")))
    return out


def episode_info(filename: str, sidecar: str = "", eval_info: dict = None, languages: dict = None) -> dict:
    """Task instruction and sim success of one video (see the module docstring for the order)."""
    languages = languages or {}
    base = filename[:-4] if filename.endswith(".mp4") else filename
    info = {"instruction": None, "instruction_source": "none", "sim_success": None, "sim_success_source": "none"}

    m = _SUCCESS_IN_NAME.search(base)
    if m:
        info.update(sim_success=m.group(1).lower() == "true", sim_success_source="filename")
    else:
        r = sidecar_result(sidecar)
        if r is not None:
            info.update(sim_success=r, sim_success_source="sidecar")

    task = sidecar_task(sidecar)
    if task:
        info.update(instruction=task, instruction_source="sidecar")
    else:
        m = _TASK_IN_NAME.search(base)
        if m:
            info.update(instruction=m.group(1).replace("_", " ").strip(), instruction_source="filename")

    group, video = split_lerobot_name(filename)
    candidates = [e for e in lerobot_episodes(eval_info) if e[1] == video and (group is None or e[0] == group)]
    if len(candidates) == 1:  # a flat name shared by several tasks is ambiguous: leave it unknown
        _, _, success, suite, task_id = candidates[0]
        if info["sim_success"] is None and success is not None:
            info.update(sim_success=success, sim_success_source="eval_info")
        if info["instruction"] is None:
            suite, task_id = _suite_and_id(candidates[0][0], suite, task_id)
            langs = languages.get(suite or "")
            if langs and task_id is not None and 0 <= task_id < len(langs):
                info.update(instruction=langs[task_id], instruction_source="libero_task_map")
    return info


def _suite_and_id(group_dir: str, suite, task_id):
    if suite is not None and task_id is not None:
        return suite, int(task_id)
    m = _LEROBOT_GROUP.match(group_dir or "")
    return (m.group("suite"), int(m.group("task_id"))) if m else (suite, task_id)


def build_episodes(run_dir: Path, languages: dict = None) -> list:
    """Every task-N/videos/*.mp4 under run_dir, in (task, name) order, with its manifest fields."""
    run_dir = Path(run_dir)
    episodes = []
    tasks = sorted((int(m.group(1)), d) for d in run_dir.iterdir() if d.is_dir() and (m := _TASK_DIR.match(d.name)))
    for task_idx, task_dir in tasks:
        eval_info = _read_json(task_dir / "eval_info.json")
        for video in sorted((task_dir / "videos").glob("*.mp4")):
            sidecar = video.with_suffix(".txt")
            text = sidecar.read_text(errors="replace") if sidecar.is_file() else ""
            ep = {
                "id": f"ep{len(episodes) + 1:04d}",
                "task_idx": task_idx,
                "video_key": f"{task_dir.name}/videos/{video.name}",
                "local_path": str(video),
                **episode_info(video.name, text, eval_info, languages),
            }
            episodes.append(ep)
    return episodes


def jobs_for(episodes: list) -> list:
    """batch_utils job lines: one video per episode (the rollout mp4 is already a camera mosaic)."""
    return [{"id": e["id"], "task": e["instruction"], "videos": [e["local_path"]]} for e in episodes if e["instruction"]]


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ───────────────────────────── scores + report ──────────────────────────


def parse_score(record: dict):
    """Integer score 1..5 from one terminal_reward output line, else None.

    Uses the script's parsed ``score``; falls back to the raw ``output`` text ("... Score: N")."""
    score = record.get("score")
    if isinstance(score, bool):
        score = None
    if isinstance(score, int) and 1 <= score <= 5:
        return score
    if isinstance(score, str) and score.strip() in {"1", "2", "3", "4", "5"}:
        return int(score.strip())
    m = _SCORE_TEXT.search(str(record.get("output") or "").strip())
    return int(m.group(1)) if m else None


def read_scores(lines) -> dict:
    out = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("id"):
            out[str(rec["id"])] = rec
    return out


def flora_success(score, threshold: int = SUCCESS_THRESHOLD):
    return None if score is None else score >= threshold


def summarize(episodes: list, threshold: int = SUCCESS_THRESHOLD) -> dict:
    """Agreement between the simulator's success flag and Flora (score >= threshold)."""
    scored = [e for e in episodes if e.get("flora_score") is not None]
    compared = [e for e in scored if e.get("sim_success") is not None]
    confusion = {"sim_success_flora_success": 0, "sim_success_flora_failure": 0,
                 "sim_failure_flora_success": 0, "sim_failure_flora_failure": 0}
    for e in compared:
        key = ("sim_success" if e["sim_success"] else "sim_failure") + \
              ("_flora_success" if e["flora_score"] >= threshold else "_flora_failure")
        confusion[key] += 1
    agree = confusion["sim_success_flora_success"] + confusion["sim_failure_flora_failure"]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    hist = {str(s): sum(1 for e in scored if e["flora_score"] == s) for s in range(1, 6)}
    return {
        "n_episodes": len(episodes),
        "n_scored": len(scored),
        "n_compared": len(compared),
        "n_unscored": len(episodes) - len(scored),
        "agreement": round(agree / len(compared), 4) if compared else None,
        "n_agree": agree,
        "confusion": confusion,
        "mean_score": mean([e["flora_score"] for e in scored]),
        "mean_score_sim_success": mean([e["flora_score"] for e in compared if e["sim_success"]]),
        "mean_score_sim_failure": mean([e["flora_score"] for e in compared if not e["sim_success"]]),
        "sim_success_rate": round(sum(bool(e["sim_success"]) for e in compared) / len(compared), 4) if compared else None,
        "flora_success_rate": round(sum(e["flora_score"] >= threshold for e in scored) / len(scored), 4) if scored else None,
        "score_histogram": hist,
    }


def build_report(manifest: dict, scores: dict, meta: dict = None, threshold: int = SUCCESS_THRESHOLD,
                 dropped: dict = None) -> dict:
    """judge.json: per-episode fields + summary. ``dropped`` = {episode id: reason} for videos the
    instance could not decode (left out of the Flora batch)."""
    dropped = dropped or {}
    episodes = []
    for e in manifest.get("episodes", []):
        rec = scores.get(e["id"]) or {}
        score = parse_score(rec) if rec else None
        reason = None
        if not e.get("instruction"):
            reason = "no task instruction found"
        elif e["id"] in dropped:
            reason = str(dropped[e["id"]])
        elif score is None:
            reason = "no parseable Flora score"
        episodes.append({
            "id": e["id"],
            "task_idx": e.get("task_idx"),
            "video_key": e["video_key"],
            "instruction": e.get("instruction"),
            "instruction_source": e.get("instruction_source"),
            "sim_success": e.get("sim_success"),
            "sim_success_source": e.get("sim_success_source"),
            "flora_score": score,
            "flora_success": flora_success(score, threshold),
            "flora_expected_score": rec.get("expected_score"),
            "rationale": rec.get("rationale"),
            "skipped_reason": reason,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "judge": "flora-rewarder-outcome",
        "success_threshold": threshold,
        "threshold_note": f"flora_success = flora_score >= {threshold} (1 = complete failure, 5 = complete success)",
        **(meta or {}),
        "summary": summarize(episodes, threshold),
        "episodes": episodes,
    }


REPORT_KEYS = ("schema_version", "success_threshold", "summary", "episodes")
EPISODE_KEYS = ("id", "video_key", "instruction", "sim_success", "flora_score", "flora_success", "rationale")


def validate_report(report: dict) -> list:
    """Schema problems of a judge.json (empty list = valid)."""
    problems = [f"missing key {k}" for k in REPORT_KEYS if k not in report]
    for i, e in enumerate(report.get("episodes", [])):
        problems += [f"episodes[{i}]: missing {k}" for k in EPISODE_KEYS if k not in e]
        s = e.get("flora_score")
        if s is not None and (not isinstance(s, int) or not 1 <= s <= 5):
            problems.append(f"episodes[{i}]: flora_score {s!r} outside 1..5")
        if s is not None and e.get("flora_success") != (s >= report.get("success_threshold", SUCCESS_THRESHOLD)):
            problems.append(f"episodes[{i}]: flora_success does not match the threshold")
    return problems


def render_txt(report: dict) -> str:
    s = report["summary"]
    th = report.get("success_threshold", SUCCESS_THRESHOLD)

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.1f}%" if pct else f"{v}"

    c = s["confusion"]
    lines = [
        f"Flora-Rewarder outcome judge — {report.get('target', '?')} run {report.get('run_id', '?')}",
        f"model: {report.get('model', '?')} @ {str(report.get('model_revision', '?'))[:12]}"
        f" ({report.get('license', 'see model card')})",
        f"instance: {report.get('instance_type', '?')}  tensor_parallel: {report.get('tensor_parallel', '?')}"
        f"  wall: {report.get('wall_time_s', '?')} s",
        f"flora_success = score >= {th} (1 = complete failure, 5 = complete success)",
        "",
        f"episodes: {s['n_episodes']}  scored: {s['n_scored']}  compared with sim: {s['n_compared']}",
        f"agreement with sim success: {fmt(s['agreement'], True)} ({s['n_agree']}/{s['n_compared']})",
        f"sim success rate: {fmt(s['sim_success_rate'], True)}  flora success rate: {fmt(s['flora_success_rate'], True)}",
        f"mean score: all {fmt(s['mean_score'])}  sim-success {fmt(s['mean_score_sim_success'])}"
        f"  sim-failure {fmt(s['mean_score_sim_failure'])}",
        "confusion (rows sim, cols flora):",
        f"  sim success: flora success {c['sim_success_flora_success']}  flora failure {c['sim_success_flora_failure']}",
        f"  sim failure: flora success {c['sim_failure_flora_success']}  flora failure {c['sim_failure_flora_failure']}",
        "score histogram: " + "  ".join(f"{k}:{v}" for k, v in s["score_histogram"].items()),
        "",
        "id      sim    score  video / rationale",
    ]
    for e in report["episodes"]:
        sim = "-" if e["sim_success"] is None else ("ok" if e["sim_success"] else "fail")
        score = "-" if e["flora_score"] is None else str(e["flora_score"])
        lines.append(f"{e['id']}  {sim:<5}  {score:<5}  {e['video_key']}")
        if e.get("rationale"):
            lines.append(f"{'':21}{e['rationale']}")
        if e.get("skipped_reason"):
            lines.append(f"{'':21}(not scored: {e['skipped_reason']})")
    return "\n".join(lines) + "\n"


# ───────────────────────────── gallery captions ─────────────────────────

CAPTION_MARK = "--- Flora judge ---"


def annotate_caption(text: str, episode: dict, report: dict) -> str:
    """Sidecar text with a Flora block appended (any earlier Flora block is replaced)."""
    text = (text or "").split("\n" + CAPTION_MARK, 1)[0].rstrip("\n")
    if text.startswith(CAPTION_MARK):
        text = ""
    block = [CAPTION_MARK, f"Flora score: {episode['flora_score']}/5"
             f" ({'success' if episode['flora_success'] else 'failure'} at >= {report.get('success_threshold')})"]
    if episode.get("rationale"):
        block.append(f"Rationale: {episode['rationale']}")
    block.append(f"Model: {report.get('model', '')}@{str(report.get('model_revision', ''))[:8]}")
    return (text + "\n\n" if text else "") + "\n".join(block) + "\n"


# ───────────────────────────── CLI ──────────────────────────────────────


def _meta(pairs) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not k:
            continue
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        elif re.fullmatch(r"-?\d+\.\d+", v):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="manifest.json + batch_utils jobs.jsonl from a local run copy")
    pr.add_argument("--run-dir", type=Path, required=True)
    pr.add_argument("--jobs", type=Path, required=True)
    pr.add_argument("--manifest", type=Path, required=True)
    pr.add_argument("--libero-task-map", type=Path, help="LIBERO libero_suite_task_map.py (optional)")
    rp = sub.add_parser("report", help="judge.json + judge.txt from the manifest and terminal_reward output")
    rp.add_argument("--manifest", type=Path, required=True)
    rp.add_argument("--scores", type=Path, required=True)
    rp.add_argument("--json", type=Path, required=True)
    rp.add_argument("--txt", type=Path, required=True)
    rp.add_argument("--threshold", type=int, default=SUCCESS_THRESHOLD)
    rp.add_argument("--dropped", type=Path, help="JSON {episode id: reason} of videos left out of the batch")
    rp.add_argument("--meta", nargs="*", default=[], help="key=value pairs copied into judge.json")
    cp = sub.add_parser("captions", help="write annotated sidecars (<video>.txt) for scored episodes")
    cp.add_argument("--manifest", type=Path, required=True)
    cp.add_argument("--report", type=Path, required=True)
    cp.add_argument("--out-dir", type=Path, required=True, help="annotated sidecars, same relative paths")
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        languages = {}
        if a.libero_task_map and a.libero_task_map.is_file():
            try:
                languages = libero_languages(a.libero_task_map.read_text())
            except (ValueError, SyntaxError) as e:
                print(f"[flora-judge] LIBERO task map unreadable ({e}); LeRobot videos need a sidecar Task", file=sys.stderr)
        episodes = build_episodes(a.run_dir, languages)
        jobs = jobs_for(episodes)
        a.manifest.write_text(json.dumps({"episodes": episodes}, indent=1))
        a.jobs.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
        print(f"episodes={len(episodes)} jobs={len(jobs)} "
              f"with_sim_success={sum(e['sim_success'] is not None for e in episodes)}")
        return 0 if jobs else 3

    if a.cmd == "report":
        manifest = json.loads(a.manifest.read_text())
        scores = read_scores(a.scores.read_text().splitlines()) if a.scores.is_file() else {}
        dropped = _read_json(a.dropped) if a.dropped else None
        report = build_report(manifest, scores, _meta(a.meta), a.threshold, dropped)
        problems = validate_report(report)
        if problems:
            print("[flora-judge] judge.json schema problems: " + "; ".join(problems), file=sys.stderr)
            return 4
        a.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
        a.txt.write_text(render_txt(report))
        s = report["summary"]
        print(f"scored={s['n_scored']}/{s['n_episodes']} compared={s['n_compared']} agreement={s['agreement']}")
        return 0 if s["n_scored"] else 5

    manifest = {e["id"]: e for e in json.loads(a.manifest.read_text())["episodes"]}
    report = json.loads(a.report.read_text())
    n = 0
    for e in report["episodes"]:
        if e.get("flora_score") is None:
            continue
        local = Path(manifest[e["id"]]["local_path"]).with_suffix(".txt")
        text = local.read_text(errors="replace") if local.is_file() else ""
        out = a.out_dir / Path(e["video_key"]).with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(annotate_caption(text, e, report))
        n += 1
    print(f"captions={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
