# Video history gallery

Back to the [README](../README.md) · [docs index](README.md).

Every completion e-mail links to one page with **all the videos of every run deployed under your
notification e-mail**, grouped by target (model), newest run first. The page is served by
Amazon CloudFront from a private S3 bucket and is protected by a password. The `aws s3 sync`
download line stays in the e-mail as before.

```text
Video history (all your runs, by model; password: python vlasim.py gallery):
  https://d1234abcd.cloudfront.net/g/3f9c.../

Download results:
  aws s3 sync s3://vla-sim-results-smolvla-demo-us-east-1-<ACCOUNT>/<RUN_ID>/ ./smolvla-results/<RUN_ID>/ --region us-east-1
```

## Using it

```bash
python vlasim.py gallery                    # URL + password for the e-mail in simulator-config.yaml
python vlasim.py gallery --email a@b.com    # another e-mail / --region another region
python vlasim.py gallery --reset-password   # issue a new password (old one stops working in seconds)
```

Open the URL; the browser asks for a user name and password. The user name is ignored, the
password is the one printed above. Videos play inline (`<video>`), with the episode caption
(task instruction) when the target writes one. After a [post-run judge](targets.md#post-run-judge-flora-rewarder)
run, each scored video also shows its Flora score (`Flora 4/5`); the judge appends it to the video's
caption and the target's `GalleryPublisher` republishes it (the target stack must still exist, and
must have been deployed with this version so the rewritten caption is read again).

## How it works

```
deploy.py --vla <target>
  ├─ RFM-Gallery stack (once per account, created when missing)
  │    S3 bucket rfm-gallery-<account>-<region> (private, RETAIN)
  │    CloudFront distribution ── OAC ──> bucket
  │    CloudFront Function (viewer-request) + KeyValueStore: Basic auth for /g/<id>/...
  ├─ SSM /vla-simulator/gallery/salt and /vla-simulator/gallery/<id>/password (SecureString)
  ├─ KeyValueStore entry <id> -> sha256(<id> + password)
  └─ target stack + GalleryPublisher Lambda (S3 notification on .mp4 / .txt in the results bucket)
       copies videos to g/<id>/<target>/<run_id>/... and rebuilds g/<id>/index.html + manifest.json
```

- `<id>` = first 32 hex chars of HMAC-SHA256(salt, lower-case e-mail). The salt is random per
  account and region, so the id cannot be derived from the e-mail without access to the account.
- The password is random (20 URL-safe characters) and only its salted SHA-256 digest is in the
  KeyValueStore. Wrong password and unknown id both answer `401`.
- The page is plain HTML (no external JavaScript or CSS); every name taken from S3 is escaped.
- Earlier runs are included: after each deploy, `deploy.py` copies videos that are already in the
  target's results bucket (kept by `destroy.py --keep-data`) into the gallery. Files already in the
  gallery are skipped. Runs are listed by the time in their run id, newest first.
- One gallery per region: targets deployed in another region have their own gallery URL there.

## Opting out

- One deploy: `python deploy.py --vla <target> --no-gallery` (or `vlasim deploy ... --no-gallery`).
- **Other regions**: CloudFront names (OAC, KeyValueStore) are account-global, so the gallery exists in one region per account. A deploy in another region stops before CDK with an error naming the existing KeyValueStore; deploy there with `--no-gallery`.
- Always: `deployment.gallery: false` in `simulator-config.yaml`.

Without the gallery the target stack is the same as before (no Lambda, no notification, no link
in the e-mail). Registering the password uses the CloudFront KeyValueStore API, which is signed
with SigV4A and needs the AWS CRT (`boto3[crt]` in `requirements.txt`); `vlasim doctor` warns
when it is missing.

After upgrading this repository, `python deploy.py --vla <target> --update-gallery` re-deploys the
shared RFM-Gallery stack (otherwise it is only created when missing).

## Destroy: keep or delete the data

`destroy.py` asks what to do with the data (a numbered prompt on a terminal, default **1**):

| Choice | Flag | Compute stack | Results bucket | Gallery |
|---|---|---|---|---|
| 1. Keep data (default) | `--keep-data` | deleted | kept | kept, page keeps working |
| 2. Delete all | `--delete-all` | deleted, plus its Lambda log groups | emptied + deleted | this target's videos removed, index rebuilt |

With `--delete-all`, when **no other target stack is left in the region**, the shared gallery goes
too: the RFM-Gallery stack (CloudFront distribution, function, KeyValueStore), its bucket (all
e-mails' histories in that region) and the `/vla-simulator/gallery/*` SSM parameters.
`--delete-all` asks you to type the stack name; add `--yes` to skip that (required without a
terminal). Without a terminal and without a flag, destroy keeps the data.

## Cost

CloudFront and S3 at the sample's volume are typically cents per month: S3 Standard storage for
the copied videos (the results bucket keeps its own copy), CloudFront data transfer when you watch
them, CloudFront Functions and KeyValueStore reads per request, and a few Lambda invocations per
run. The distribution itself has no fixed monthly charge. See [cost.md](cost.md) for the EC2 side.
