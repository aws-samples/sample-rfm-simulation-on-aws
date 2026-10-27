"""Video history gallery: id/password helpers, the GalleryPublisher index builder and Lambda handler
(against a fake S3), and the destroy.py data choice. No AWS calls."""

import hashlib
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import gallery  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gallery_publisher", ROOT / "cdk" / "lambda" / "gallery_publisher" / "index.py")
pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)

GID = "0123456789abcdef0123456789abcdef"
PREFIX = f"g/{GID}/"


# ───────────────────────────── id / password ─────────────────────────────


def test_email_hash_is_stable_case_insensitive_and_salted():
    h = gallery.email_hash("salt-a", "User@Example.com")
    assert re.fullmatch(r"[0-9a-f]{32}", h)
    assert h == gallery.email_hash("salt-a", "  user@example.COM ")
    assert h != gallery.email_hash("salt-b", "user@example.com")
    assert h != gallery.email_hash("salt-a", "other@example.com")


def test_new_password_is_random_url_safe():
    pws = {gallery.new_password() for _ in range(50)}
    assert len(pws) == 50
    for pw in pws:
        assert len(pw) == 20 and re.fullmatch(r"[A-Za-z0-9_-]+", pw)


def test_password_digest_matches_cloudfront_function_vector():
    # Same vector as cdk/test/gallery.test.ts (runs cdk/functions/gallery-auth.js).
    assert gallery.password_digest(GID, "s3cret-Pass_1") == \
        "7e51c6871c3dd77b9379e976ba2c4205ef41cdd638e18ac635fad0c4627d21bb"
    assert gallery.password_digest(GID, "x") == hashlib.sha256((GID + "x").encode()).hexdigest()


def test_gallery_url_and_params():
    assert gallery.gallery_url("d1.cloudfront.net", GID) == f"https://d1.cloudfront.net/g/{GID}/"
    assert gallery.password_param(GID) == f"/vla-simulator/gallery/{GID}/password"
    assert gallery.SALT_PARAM.startswith("/vla-simulator/")


# ───────────────────────────── destroy choice ─────────────────────────────


def _no_prompt(_):
    raise AssertionError("must not prompt")


@pytest.mark.parametrize("keep,delete,tty,expected", [
    (True, False, True, gallery.KEEP),
    (False, True, True, gallery.DELETE_ALL),
    (False, True, False, gallery.DELETE_ALL),
    (True, False, False, gallery.KEEP),
    (False, False, False, gallery.KEEP),  # no TTY: safe default
])
def test_destroy_choice_flags(keep, delete, tty, expected):
    assert gallery.resolve_destroy_choice(keep, delete, tty, ask=_no_prompt) == expected


@pytest.mark.parametrize("answers,expected", [
    ([""], gallery.KEEP),
    (["1"], gallery.KEEP),
    (["2"], gallery.DELETE_ALL),
    (["x", "3", "2"], gallery.DELETE_ALL),
])
def test_destroy_choice_prompt(answers, expected, capsys):
    it = iter(answers)
    assert gallery.resolve_destroy_choice(False, False, True, ask=lambda _: next(it)) == expected


def test_destroy_choice_conflict():
    with pytest.raises(ValueError):
        gallery.resolve_destroy_choice(True, True, True)


# ───────────────────────────── index builder ─────────────────────────────


def _obj(key, minute, size=10):
    return {"Key": PREFIX + key, "LastModified": datetime(2026, 10, 27, 1, minute, tzinfo=timezone.utc), "Size": size}


def test_gallery_key():
    assert pub.gallery_key(PREFIX, "smolvla", "20261027-0100/task-0/videos/ep0.mp4") == \
        f"{PREFIX}smolvla/20261027-0100/task-0/videos/ep0.mp4"
    assert pub.gallery_key(PREFIX, "smolvla", "ep0.mp4") is None  # not under a run id
    assert pub.gallery_key(PREFIX, "smolvla", "run/../x.mp4") is None
    assert pub.gallery_key(PREFIX, "smolvla", "run//x.mp4") is None


def test_manifest_groups_by_target_newest_run_first():
    objects = [
        _obj("smolvla/run-a/task-0/videos/ep1.mp4", 1),
        _obj("smolvla/run-a/task-0/videos/ep0.mp4", 2),
        _obj("smolvla/run-b/task-0/videos/ep0.mp4", 30),
        _obj("smolvla/run-b/task-0/videos/ep0.txt", 30),
        _obj("xvla/run-c/task-1/videos/ep0.mp4", 5),
        _obj("index.html", 59),
        _obj("manifest.json", 59),
        _obj("smolvla/run-b/summary.txt", 31),  # no video next to it: not a caption
        {"Key": "g/ffffffffffffffffffffffffffffffff/pi/r/x.mp4", "LastModified": "2026-10-27T00:00:00Z"},
    ]
    caps = pub.caption_keys(objects, PREFIX)
    assert caps == [PREFIX + "smolvla/run-b/task-0/videos/ep0.txt"]
    m = pub.build_manifest(objects, PREFIX, {caps[0]: "put the bowl on the plate"})
    assert [t["name"] for t in m["targets"]] == ["smolvla", "xvla"]
    smol = m["targets"][0]
    assert [r["run_id"] for r in smol["runs"]] == ["run-b", "run-a"]  # newest first
    assert [v["name"] for v in smol["runs"][1]["videos"]] == ["task-0/videos/ep0.mp4", "task-0/videos/ep1.mp4"]
    assert smol["runs"][0]["videos"][0]["caption"] == "put the bowl on the plate"
    assert smol["runs"][0]["updated"] == "2026-10-27T01:30:00Z"
    assert len(m["targets"][1]["runs"]) == 1


def test_index_html_escapes_everything():
    objects = [
        _obj('evil<script>/run&1/"q"/a b.mp4', 1),
        _obj('evil<script>/run&1/"q"/a b.txt', 1),
    ]
    caption_key = PREFIX + 'evil<script>/run&1/"q"/a b.txt'
    page = pub.build_index_html(pub.build_manifest(objects, PREFIX, {caption_key: "<img src=x onerror=alert(1)>"}),
                                generated="<now>")
    assert "<script>" not in page and "<img" not in page and "<now>" not in page
    assert "evil&lt;script&gt;" in page and "run&amp;1" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    # the video src is percent-encoded and relative to g/<id>/
    assert 'src="evil%3Cscript%3E/run%261/%22q%22/a%20b.mp4"' in page
    assert "http" not in page.split("<body>")[1]  # no external resources


def test_index_html_empty_and_counts():
    assert "No videos yet." in pub.build_index_html({"targets": []})
    page = pub.build_index_html(pub.build_manifest([_obj("pi/r1/v.mp4", 1), _obj("pi/r2/v.mp4", 2)], PREFIX))
    assert "1 target(s), 2 run(s), 2 video(s)" in page
    assert page.index("Run r2") < page.index("Run r1")


# ───────────────────────────── handler (fake S3) ─────────────────────────────


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class _Paginator:
    def __init__(self, s3):
        self.s3 = s3

    def paginate(self, Bucket, Prefix):
        keys = sorted(k for (b, k) in self.s3.objects if b == Bucket and k.startswith(Prefix))
        yield {"Contents": [{"Key": k, "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc), "Size": 1}
                            for k in keys]}


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_paginator(self, _name):
        return _Paginator(self)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)

    def get_object(self, Bucket, Key, Range=None):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body

    def copy_object(self, CopySource, Bucket, Key, **_):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]


def _event(bucket, key):
    return {"Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]}


def test_handler_copies_video_caption_and_rebuilds(monkeypatch):
    fake = FakeS3({
        ("res", "r1/task-0/videos/ep+0.mp4"): b"mp4",
        ("res", "r1/task-0/videos/ep+0.txt"): "pick up the cup".encode(),
        ("res", "r1/summary.txt"): b"summary",
    })
    monkeypatch.setattr(pub, "_s3", fake)
    monkeypatch.setenv("GALLERY_BUCKET", "gal")
    monkeypatch.setenv("GALLERY_PREFIX", PREFIX)
    monkeypatch.setenv("GALLERY_TARGET", "smolvla")

    # S3 event keys are URL-encoded ('+' for space, %2B for '+').
    assert pub.handler(_event("res", "r1/task-0/videos/ep%2B0.mp4"), None) == {"copied": 2}
    assert ("gal", f"{PREFIX}smolvla/r1/task-0/videos/ep+0.mp4") in fake.objects
    assert ("gal", f"{PREFIX}smolvla/r1/task-0/videos/ep+0.txt") in fake.objects
    page = fake.objects[("gal", PREFIX + "index.html")].decode()
    assert "pick up the cup" in page and "Run r1" in page

    # summary.txt has no video next to it: ignored, no rebuild
    assert pub.handler(_event("res", "r1/summary.txt"), None) == {"copied": 0}
    assert not any(k.endswith("summary.txt") for (b, k) in fake.objects if b == "gal")


def test_manifest_orders_runs_by_run_id_time_not_copy_time():
    # A backfilled (older) run is copied after a newer one: the run id decides the order.
    objects = [
        _obj("smolvla/2026-10-27-0350/task-0/videos/ep0.mp4", 1),
        _obj("smolvla/2026-10-14-0645/task-0/videos/ep0.mp4", 40),
        _obj("smolvla/2026-10-22-1316/task-0/videos/ep0.mp4", 50),
    ]
    m = pub.build_manifest(objects, PREFIX)
    assert [r["run_id"] for r in m["targets"][0]["runs"]] == [
        "2026-10-27-0350", "2026-10-22-1316", "2026-10-14-0645"]


class _FakeS3:
    def __init__(self, buckets):
        self.buckets = buckets  # {bucket: set(keys)}
        self.copies = []

    def get_paginator(self, _name):
        fake = self

        class _P:
            def paginate(self, Bucket, Prefix=""):
                yield {"Contents": [{"Key": k} for k in sorted(fake.buckets[Bucket]) if k.startswith(Prefix)]}
        return _P()

    def copy_object(self, CopySource, Bucket, Key, **_kw):
        self.copies.append((CopySource["Key"], Key))
        self.buckets[Bucket].add(Key)


def test_backfill_copies_missing_videos_and_captions_once(monkeypatch):
    target = PREFIX + "smolvla/"
    s3 = _FakeS3({
        "results": {"r1/task-0/videos/ep0.mp4", "r1/task-0/videos/ep0.txt", "r1/summary.txt",
                    "r2/task-0/videos/ep0.mp4", "r1/task-0/eval.log"},
        "gallery": {target + "r2/task-0/videos/ep0.mp4"},
    })
    rebuilt = []

    class _B:
        def client(self, *_a, **_k):
            return s3
    monkeypatch.setattr(gallery, "_boto3", lambda: _B())
    monkeypatch.setattr(gallery, "rebuild_index", lambda b, p, r: rebuilt.append((b, p)))
    assert gallery.backfill("ap-northeast-2", "results", "gallery", target) == 2
    assert sorted(d for _, d in s3.copies) == [target + "r1/task-0/videos/ep0.mp4", target + "r1/task-0/videos/ep0.txt"]
    assert rebuilt == [("gallery", PREFIX)]
    assert gallery.backfill("ap-northeast-2", "results", "gallery", target) == 0  # idempotent
    assert len(rebuilt) == 1
