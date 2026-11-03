# Intro video

Source of the README intro: [`docs/showcase/intro/intro.mp4`](../../docs/showcase/intro/intro.mp4) (narrated, captions burned in)
and its silent looping preview [`docs/showcase/intro/intro-preview.webp`](../../docs/showcase/intro/intro-preview.webp).

- `scenes.json`: the scene list — narration text, the showcase clips each scene uses (path under `docs/showcase/`,
  playback speed, optional `crop` in source pixels to leave out a burned-in task caption), the deploy diagram nodes and
  the bar chart numbers (from [`docs/results.md`](../../docs/results.md)).
- `render.py`: draws every frame with Pillow (titles, diagram, chart, captions) around the decoded rollout clips and
  pipes them into ffmpeg (H.264 + AAC, animated WebP for the preview). No browser, no GUI.

Every moving picture is a real rollout recorded by `vlasim.py deploy`; nothing is a mock-up.

## Re-render

```bash
python -m venv .venv-video && . .venv-video/bin/activate
pip install numpy pillow boto3 imageio-ffmpeg matplotlib   # matplotlib only provides the DejaVu fonts
python tools/intro-video/render.py                          # writes both files under docs/showcase/intro/
```

- Narration is Amazon Polly (neural voice `Matthew`, `us-east-1`; the whole script is about 1,000 characters, a few
  cents). It needs AWS credentials with `polly:SynthesizeSpeech`. Audio and word timings are cached in
  `.temp/intro-video-cache/` (gitignored), so later renders make no call. The word timings set each scene's length,
  the moment each diagram node lights up and the caption lines.
- `--no-voice`: no AWS call; silent track, timings estimated from the word count.
- `--stills DIR --at 3,20,41`: write PNG frames at those seconds instead of encoding (to check the layout).
- `--only video|preview`, `--crf N` (default 27; the MP4 should stay under the 4 MB `check-added-large-files` limit of
  the pre-commit config), `--font / --font-bold / --font-mono` (any TTF instead of DejaVu).
- ffmpeg is taken from `$FFMPEG`, then `PATH`, then the `imageio-ffmpeg` wheel. It needs `libx264` and `libwebp`.

To change the video, edit `scenes.json` (text, clips, speeds) and render again; `tests/test_intro_video.py` checks that
every clip it names exists.
