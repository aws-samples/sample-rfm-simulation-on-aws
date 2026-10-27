"""GalleryPublisher: copy run videos into the video history gallery and rebuild its index.

Triggered by ObjectCreated events (suffix .mp4 and .txt) on a target's results bucket.

  results:  s3://<results-bucket>/<run_id>/<path>/<name>.mp4   (+ optional <name>.txt caption)
  gallery:  s3://<gallery-bucket>/g/<id>/<target>/<run_id>/<path>/<name>.mp4

Then g/<id>/index.html and g/<id>/manifest.json are rebuilt from a listing of g/<id>/, so the page
shows every target published under the same notification e-mail, grouped by target, newest run
first. The builder functions are pure (no AWS calls) so tests and destroy.py can reuse them.

Environment: GALLERY_BUCKET, GALLERY_PREFIX (g/<id>/), GALLERY_TARGET (e.g. smolvla).
"""

import html
import json
import os
import re
import urllib.parse
from datetime import datetime, timezone

VIDEO_EXT = ".mp4"
CAPTION_EXT = ".txt"
INDEX = "index.html"
MANIFEST = "manifest.json"
CAPTION_MAX_BYTES = 4096
REBUILD_ATTEMPTS = 3

_s3 = None


def _client():
    global _s3
    if _s3 is None:
        import boto3  # noqa: PLC0415 - imported lazily so the builders import without boto3

        _s3 = boto3.client("s3")
    return _s3


# ───────────────────────────── pure helpers ─────────────────────────────


def gallery_key(prefix: str, target: str, source_key: str):
    """Results key <run_id>/<rest> -> gallery key <prefix><target>/<run_id>/<rest>; None if not a run file."""
    parts = source_key.split("/")
    if len(parts) < 2 or any(p in ("", ".", "..") for p in parts):
        return None
    return f"{prefix}{target}/{source_key}"


def build_manifest(objects, prefix: str, captions=None) -> dict:
    """Group a listing of the gallery prefix into {targets: [{name, runs: [{run_id, videos}]}]}.

    objects:  iterable of {"Key", "LastModified" (datetime or ISO str), "Size"}
    captions: {gallery caption key: text}
    Targets are sorted by name, runs newest first (time in the run id, else latest object time), videos by path.
    """
    captions = captions or {}
    keys = set()
    runs: dict = {}
    for obj in objects:
        key = obj["Key"]
        if not key.startswith(prefix):
            continue
        keys.add(key)
        rel = key[len(prefix):]
        parts = rel.split("/")
        if len(parts) < 3 or not rel.endswith(VIDEO_EXT):
            continue
        target, run_id = parts[0], parts[1]
        modified = _iso(obj.get("LastModified"))
        run = runs.setdefault((target, run_id), {"run_id": run_id, "updated": modified, "videos": []})
        run["updated"] = max(run["updated"], modified)
        caption_key = key[: -len(VIDEO_EXT)] + CAPTION_EXT
        run["videos"].append({
            "path": rel,
            "name": "/".join(parts[2:]),
            "size": int(obj.get("Size", 0)),
            "caption": captions.get(caption_key, ""),
        })
    for run in runs.values():
        run["videos"].sort(key=lambda v: v["path"])
    targets = []
    for name in sorted({t for t, _ in runs}):
        target_runs = [r for (t, _), r in runs.items() if t == name]
        target_runs.sort(key=lambda r: (_run_time(r), r["run_id"]), reverse=True)
        targets.append({"name": name, "runs": target_runs})
    return {"targets": targets}


_RUN_ID_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})")


def _run_time(run: dict) -> str:
    """Start time of a run from its id (YYYY-MM-DD-HHMM...), else its latest upload time.

    Uses the id first so runs copied later (backfill) still sort by when they ran.
    """
    m = _RUN_ID_TIME.match(run["run_id"])
    return f"{m.group(1)}T{m.group(2)}:{m.group(3)}" if m else run["updated"]


def caption_keys(objects, prefix: str):
    """Caption objects (.txt next to a .mp4) in a listing of the gallery prefix."""
    keys = {o["Key"] for o in objects if o["Key"].startswith(prefix)}
    return sorted(k for k in keys if k.endswith(CAPTION_EXT) and k[: -len(CAPTION_EXT)] + VIDEO_EXT in keys)


def build_index_html(manifest: dict, generated: str = "") -> str:
    """Self-contained HTML page (no external JS/CSS); every value from S3 is escaped."""
    e = html.escape
    targets = manifest.get("targets", [])
    out = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="robots" content="noindex">',
        "<title>RFM video history</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:1.5rem;color:#222}"
        "nav a{margin-right:1rem}section{margin-top:2rem}h3{font-size:1rem;margin:1.2rem 0 .4rem}"
        ".grid{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0;width:320px}"
        "video{width:320px;background:#000}figcaption{font-size:.8rem;white-space:pre-wrap;"
        "word-break:break-word}.meta{color:#666;font-size:.85rem}</style>",
        "</head><body>",
        "<h1>RFM video history</h1>",
    ]
    total_runs = sum(len(t["runs"]) for t in targets)
    total_videos = sum(len(r["videos"]) for t in targets for r in t["runs"])
    out.append(f'<p class="meta">{len(targets)} target(s), {total_runs} run(s), {total_videos} video(s)'
               + (f", updated {e(generated)}" if generated else "") + "</p>")
    if not targets:
        out.append("<p>No videos yet.</p>")
    else:
        out.append("<nav>" + "".join(
            f'<a href="#{e(_anchor(t["name"]))}">{e(t["name"])} ({len(t["runs"])})</a>' for t in targets) + "</nav>")
    for t in targets:
        out.append(f'<section id="{e(_anchor(t["name"]))}"><h2>{e(t["name"])}</h2>')
        for run in t["runs"]:
            out.append(f'<h3>Run {e(run["run_id"])} <span class="meta">{e(run["updated"])}, '
                       f'{len(run["videos"])} video(s)</span></h3><div class="grid">')
            for v in run["videos"]:
                src = e(urllib.parse.quote(v["path"]))
                cap = f"<figcaption>{e(v['name'])}" + (f"\n{e(v['caption'])}" if v["caption"] else "") + "</figcaption>"
                out.append(f'<figure><video controls preload="metadata" src="{src}"></video>{cap}</figure>')
            out.append("</div>")
        out.append("</section>")
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def _anchor(name: str) -> str:
    return "t-" + "".join(c if c.isalnum() or c in "-_" else "-" for c in name)


def _iso(value) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value or "")


# ───────────────────────────── S3 side ─────────────────────────────────


def list_prefix(bucket: str, prefix: str):
    s3 = _client()
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def rebuild_index(bucket: str, prefix: str) -> dict:
    """Rebuild <prefix>index.html + manifest.json; re-list and retry if new objects arrived meanwhile."""
    s3 = _client()
    old_captions = _load_manifest_captions(bucket, prefix)
    manifest = {"targets": []}
    for _ in range(REBUILD_ATTEMPTS):
        objects = list_prefix(bucket, prefix)
        captions = {}
        for key in caption_keys(objects, prefix):
            if key in old_captions:
                captions[key] = old_captions[key]
            else:
                captions[key] = _read_caption(bucket, key)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest = build_manifest(objects, prefix, captions)
        manifest["generated"] = generated
        manifest["captions"] = captions
        s3.put_object(Bucket=bucket, Key=prefix + MANIFEST, Body=json.dumps(manifest, indent=1).encode(),
                      ContentType="application/json", CacheControl="no-cache")
        s3.put_object(Bucket=bucket, Key=prefix + INDEX, Body=build_index_html(manifest, generated).encode(),
                      ContentType="text/html; charset=utf-8", CacheControl="no-cache")
        after = {o["Key"] for o in list_prefix(bucket, prefix)} - {prefix + INDEX, prefix + MANIFEST}
        if after == {o["Key"] for o in objects} - {prefix + INDEX, prefix + MANIFEST}:
            break
    return manifest


def _load_manifest_captions(bucket: str, prefix: str) -> dict:
    try:
        body = _client().get_object(Bucket=bucket, Key=prefix + MANIFEST)["Body"].read()
        return dict(json.loads(body).get("captions", {}))
    except Exception:  # noqa: BLE001 - first run (no manifest yet) or unreadable: start over
        return {}


def _read_caption(bucket: str, key: str) -> str:
    try:
        body = _client().get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{CAPTION_MAX_BYTES - 1}")["Body"].read()
        return body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a missing caption only drops the text, not the video
        return ""


def _exists(bucket: str, key: str) -> bool:
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - 404, or 403 without s3:ListBucket on the source bucket
        return False


def _copy(src_bucket: str, src_key: str, bucket: str, dest: str) -> None:
    content_type = "video/mp4" if src_key.endswith(VIDEO_EXT) else "text/plain; charset=utf-8"
    _client().copy_object(CopySource={"Bucket": src_bucket, "Key": src_key}, Bucket=bucket, Key=dest,
                          MetadataDirective="REPLACE", ContentType=content_type,
                          CacheControl="public, max-age=86400")
    print(f"copied s3://{src_bucket}/{src_key} -> s3://{bucket}/{dest}")


def handler(event, context):
    bucket = os.environ["GALLERY_BUCKET"]
    prefix = os.environ["GALLERY_PREFIX"]
    target = os.environ["GALLERY_TARGET"]
    copied = 0
    for record in event.get("Records", []):
        src_bucket = record["s3"]["bucket"]["name"]
        src_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        dest = gallery_key(prefix, target, src_key)
        if dest is None:
            continue
        pairs = []
        if src_key.endswith(CAPTION_EXT):
            # Only captions that sit next to a video (summary.txt and logs are not published).
            if _exists(src_bucket, src_key[: -len(CAPTION_EXT)] + VIDEO_EXT):
                pairs.append((src_key, dest))
        elif src_key.endswith(VIDEO_EXT):
            pairs.append((src_key, dest))
            # The caption may have been uploaded before the video: pick it up now.
            caption = src_key[: -len(VIDEO_EXT)] + CAPTION_EXT
            if _exists(src_bucket, caption):
                pairs.append((caption, dest[: -len(VIDEO_EXT)] + CAPTION_EXT))
        for key, dest_key in pairs:
            _copy(src_bucket, key, bucket, dest_key)
            copied += 1
    if copied:
        manifest = rebuild_index(bucket, prefix)
        print(f"index rebuilt: {sum(len(t['runs']) for t in manifest['targets'])} run(s)")
    return {"copied": copied}
