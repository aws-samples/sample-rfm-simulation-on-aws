"""Download one HOPE-Video scene and the low-resolution object meshes, checked against pinned MD5 sums.

HOPE-Video (NVIDIA; Lin et al., "Multi-view Fusion for Multi-level Robotic Scene Understanding", IROS 2021;
https://github.com/swtyree/hope-dataset) is licensed CC BY-NC-SA 4.0: research / non-commercial use
only. Nothing from it is stored in this repository; the instance downloads it at run time.

The files are hosted on Google Drive only (the upstream `setup.py` uses gdown). The Drive ids and MD5
sums below are copied from the upstream `setup.json` at commit 621d855f (2022-12-15). There is no
Hugging Face mirror of HOPE-Video (bop-benchmark/hope carries HOPE-Image in BOP format, not the video
sequences), so a Drive quota error ("Too many users have viewed or downloaded this file") is the main
failure mode: the script retries, then exits non-zero with that message.

usage: python hope_fetch.py --scene 0 --out /opt/r2s/hope
       -> <out>/scene_0000/{NNNN.json,NNNN_rgb.jpg,NNNN_depth.png,scene.ply}  and  <out>/meshes/eval/*.obj
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

SETUP_JSON_COMMIT = "621d855f58817f8edbb4367ee0efbb7786a59a66"
SCENES = {
    0: ("1PVYp08FShW_J0Z6lgPlsb8iUqruOg7sn", "02bbfcdd9f610d638ec26ddd607777fb"),
    1: ("1-UB4AZVFr7g2nICI6vUeYDKSc5Od0sAv", "6cf92484944f338f86335a2018e4cb9e"),
    2: ("15D6TXMRlHzL5IGjwvGObYJQ7fiTyPys_", "4d844a48ba961f7ced07a5af8549e050"),
    3: ("10Zcu0c7n9OjGHLdIU3IGlqakDFYs_e6S", "125c6e388101ee4d5ac1fd0740fef560"),
    4: ("1Gwwk77VdYvnbnaHQ6x5GPyEgIr3rZrSD", "6a7e42ac6f272c782f93ff23b0df3f4e"),
    5: ("12ZIA_31M1UyaYdP8p_1mxZyaa0hggHRt", "26a4dfeac6f1299481c449b21c2f4e28"),
    6: ("1x0e6wQI9gcMMqeZEzbVqCOqrOuj6WLbC", "0ab2e32f924a1477beebf8f4d947338b"),
    7: ("15HtOzfy8v2V5IXKU6s4H_6nR3TjagOA-", "ce5066e41b2c88f609c928ff5b010109"),
    8: ("1VAi4ScMSkTDY4Q6cuYje9OI_G6ko8kJi", "df522b2090e0ce53947d59072405376c"),
    9: ("1BZbQDV0Dzm4YeVASKPNGkm6R4EMIYkYB", "708a02b3f5c15c68a4b3978a0065e908"),
}
MESHES_EVAL = ("1Y6Jo7EVM0ANjcWmYOKdGy7LWpld1VWfc", "619ee0142016369554e26592d0368a78")


def md5sum(path, buf=1 << 20):
    h = hashlib.md5()  # nosec B324 - integrity check against the upstream setup.json, not security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(drive_id, md5, dest_zip, attempts=4, wait_s=60):
    """gdown the file (CLI, so the gdown version is whatever the venv pinned) until the MD5 matches."""
    if os.path.exists(dest_zip) and md5sum(dest_zip) == md5:
        return dest_zip
    for i in range(1, attempts + 1):
        rc = subprocess.run(  # nosec B603,B607 - fixed argv, drive id from the table above
            ["gdown", "--quiet", f"https://drive.google.com/uc?id={drive_id}", "-O", dest_zip]).returncode
        if rc == 0 and os.path.exists(dest_zip):
            got = md5sum(dest_zip)
            if got == md5:
                return dest_zip
            print(f"[hope_fetch] MD5 mismatch for {dest_zip}: {got} != {md5}", file=sys.stderr)
        print(f"[hope_fetch] attempt {i}/{attempts} failed (Google Drive quota or network); "
              f"retrying in {wait_s}s", file=sys.stderr)
        if i < attempts:
            time.sleep(wait_s)
    raise SystemExit(f"[hope_fetch] could not download {drive_id} with MD5 {md5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", type=int, default=0, choices=sorted(SCENES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-meshes", action="store_true", help="skip the GT meshes (no `gt` world)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scene_dir = os.path.join(a.out, "scene_%04d" % a.scene)
    z = fetch(*SCENES[a.scene], os.path.join(a.out, "hope_video_scene_%04d.zip" % a.scene))
    zipfile.ZipFile(z).extractall(scene_dir)
    os.remove(z)
    info = {"dataset": "HOPE-Video", "license": "CC BY-NC-SA 4.0", "scene": a.scene, "scene_dir": scene_dir,
            "setup_json_commit": SETUP_JSON_COMMIT, "md5": SCENES[a.scene][1]}
    if not a.no_meshes:
        mesh_dir = os.path.join(a.out, "meshes", "eval")
        zm = fetch(*MESHES_EVAL, os.path.join(a.out, "hope_meshes_eval.zip"))
        zipfile.ZipFile(zm).extractall(mesh_dir)
        os.remove(zm)
        info["mesh_dir"] = mesh_dir
    print(json.dumps(info))


if __name__ == "__main__":
    main()
