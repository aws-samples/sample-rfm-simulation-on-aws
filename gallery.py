"""Video history gallery helpers shared by deploy.py, destroy.py and vlasim.py.

The gallery is one CloudFront distribution + S3 bucket per account and region (stack RFM-Gallery,
cdk/lib/gallery-stack.ts). Each notification e-mail gets a folder g/<id>/ and a password:

  salt      SSM SecureString /vla-simulator/gallery/salt (random, created on first use)
  id        HMAC-SHA256(salt, lower(e-mail)) hex, first 32 chars
  password  SSM SecureString /vla-simulator/gallery/<id>/password (random, URL-safe)
  KVS       <id> -> sha256hex(<id> + password), checked by the CloudFront Function (Basic auth)

The pure helpers (email_hash, new_password, password_digest, gallery_url, resolve_destroy_choice)
need only the stdlib; the AWS helpers import boto3 lazily.
"""

import hashlib
import hmac
import secrets
import subprocess  # nosec B404 - hardcoded command lists only
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GALLERY_STACK_NAME = "RFM-Gallery"
GALLERY_APP = "npx ts-node --prefer-ts-exts bin/gallery.ts"
SSM_PREFIX = "/vla-simulator/gallery"
SALT_PARAM = f"{SSM_PREFIX}/salt"
HASH_LEN = 32
STABLE_STATUSES = ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE", "IMPORT_COMPLETE")

KEEP, DELETE_ALL = "keep", "delete-all"


# ───────────────────────────── pure helpers ─────────────────────────────


def email_hash(salt: str, email: str) -> str:
    """Gallery id of an e-mail: HMAC-SHA256(salt, lower(stripped e-mail)), first 32 hex chars."""
    return hmac.new(salt.encode(), email.strip().lower().encode(), hashlib.sha256).hexdigest()[:HASH_LEN]


def new_password() -> str:
    """Random 20-character URL-safe password."""
    return secrets.token_urlsafe(15)


def new_salt() -> str:
    return secrets.token_hex(32)


def password_digest(gallery_id: str, password: str) -> str:
    """Value stored in the KeyValueStore; must match cdk/functions/gallery-auth.js."""
    return hashlib.sha256((gallery_id + password).encode()).hexdigest()


def gallery_url(domain: str, gallery_id: str) -> str:
    return f"https://{domain}/g/{gallery_id}/"


def password_param(gallery_id: str) -> str:
    return f"{SSM_PREFIX}/{gallery_id}/password"


def resolve_destroy_choice(keep_data: bool, delete_all: bool, interactive: bool, ask=input) -> str:
    """destroy.py data choice: flags win; a TTY gets a numbered prompt (default keep); else keep."""
    if keep_data and delete_all:
        raise ValueError("--keep-data and --delete-all are mutually exclusive")
    if delete_all:
        return DELETE_ALL
    if keep_data or not interactive:
        return KEEP
    print("What should happen to the data of this stack?")
    print("  1) Keep data: delete the compute stack only; the results bucket and the video history")
    print("     gallery (S3 + CloudFront) stay available  [default]")
    print("  2) Delete all: also delete the results bucket and this target's gallery videos (and the")
    print("     whole gallery when no other target stack is left in the region)")
    while True:
        answer = ask("Choose 1 or 2 [1]: ").strip()
        if answer in ("", "1"):
            return KEEP
        if answer == "2":
            return DELETE_ALL
        print("Please answer 1 or 2.")


# ───────────────────────────── AWS helpers ─────────────────────────────


def _boto3():
    import boto3  # noqa: PLC0415 - lazy: the pure helpers above are used without boto3 (tests)

    return boto3


def get_or_create_salt(region: str, create: bool = True):
    from botocore.exceptions import ClientError  # noqa: PLC0415

    ssm = _boto3().client("ssm", region_name=region)
    try:
        return ssm.get_parameter(Name=SALT_PARAM, WithDecryption=True)["Parameter"]["Value"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "ParameterNotFound":
            raise
        if not create:
            return None
    salt = new_salt()
    try:
        ssm.put_parameter(Name=SALT_PARAM, Value=salt, Type="SecureString", Overwrite=False,
                          Description="RFM video history gallery: salt for the per-e-mail gallery id")
        return salt
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterAlreadyExists":  # created concurrently
            return ssm.get_parameter(Name=SALT_PARAM, WithDecryption=True)["Parameter"]["Value"]
        raise


def get_password(region: str, gallery_id: str):
    from botocore.exceptions import ClientError  # noqa: PLC0415

    try:
        return _boto3().client("ssm", region_name=region).get_parameter(
            Name=password_param(gallery_id), WithDecryption=True)["Parameter"]["Value"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterNotFound":
            return None
        raise


def put_password(region: str, gallery_id: str, password: str) -> None:
    _boto3().client("ssm", region_name=region).put_parameter(
        Name=password_param(gallery_id), Value=password, Type="SecureString", Overwrite=True,
        Description="RFM video history gallery password (python vlasim.py gallery)")


def stack_outputs(region: str, stack_name: str):
    """{OutputKey: OutputValue} of a stack in a stable state, or None when absent/unusable."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    cfn = _boto3().client("cloudformation", region_name=region)
    try:
        stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    except ClientError:
        return None
    if stack["StackStatus"] not in STABLE_STATUSES:
        return None
    return {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}


def _kvs_client():
    try:
        import awscrt  # noqa: F401, PLC0415 - the KeyValueStore API is signed with SigV4A (needs the CRT)
    except ImportError:
        print("[error] The video history gallery needs the AWS CRT for the CloudFront KeyValueStore API "
              "(SigV4A).\n  Run: pip install -r requirements.txt   (installs boto3[crt])\n"
              "  Or deploy without the gallery: --no-gallery", file=sys.stderr)
        sys.exit(1)
    return _boto3().client("cloudfront-keyvaluestore", region_name="us-east-1")


def kvs_put(kvs_arn: str, gallery_id: str, digest: str) -> bool:
    """Write <id> -> digest when it differs. Returns True if written."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    kvs = _kvs_client()
    try:
        if kvs.get_key(KvsARN=kvs_arn, Key=gallery_id)["Value"] == digest:
            return False
    except ClientError:  # key not there yet (ResourceNotFoundException): write it below
        pass
    etag = kvs.describe_key_value_store(KvsARN=kvs_arn)["ETag"]
    kvs.put_key(KvsARN=kvs_arn, Key=gallery_id, Value=digest, IfMatch=etag)
    return True


def deploy_gallery_stack(region: str) -> None:
    cmd = [
        "npx", "cdk", "deploy", GALLERY_STACK_NAME,
        "--app", GALLERY_APP,
        "-c", f"region={region}",
        "--require-approval", "never",
        "--output", "cdk.out-gallery",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR / "cdk"), check=True)  # nosec B603,B607 - hardcoded command list


def destroy_gallery_stack(region: str) -> None:
    cmd = [
        "npx", "cdk", "destroy", GALLERY_STACK_NAME,
        "--app", GALLERY_APP,
        "-c", f"region={region}",
        "--force",
        "--output", "cdk.out-gallery",
    ]
    subprocess.run(cmd, cwd=str(BASE_DIR / "cdk"), check=True)  # nosec B603,B607 - hardcoded command list


# Prefix of the KeyValueStore name CDK generates for RFM-Gallery/GalleryPasswords. CloudFront names are
# account-global, so a second RFM-Gallery in another region fails on AlreadyExists (OAC and KVS).
GALLERY_KVS_PREFIX = "RFMGalleryGalleryPasswords"


def other_region_gallery_kvs():
    """Name of an existing RFM-Gallery KeyValueStore (any region), or None. Errors -> None (do not block)."""
    try:
        cf = _boto3().client("cloudfront", region_name="us-east-1")
        items = cf.list_key_value_stores().get("KeyValueStoreList", {}).get("Items") or []
    except Exception:  # noqa: BLE001 - best-effort precheck; the CDK deploy reports the real error
        return None
    return next((i["Name"] for i in items if str(i.get("Name", "")).startswith(GALLERY_KVS_PREFIX)), None)


def ensure_gallery(region: str, email: str, update_stack: bool = False) -> dict:
    """Deploy RFM-Gallery when missing, make sure the e-mail has a password and a KVS entry.

    Returns {"bucket", "domain", "id", "url", "kvs_arn"}.
    """
    outputs = None if update_stack else stack_outputs(region, GALLERY_STACK_NAME)
    if outputs is None:
        if not update_stack and (kvs := other_region_gallery_kvs()):
            print(f"[error] {GALLERY_STACK_NAME} is not deployed in {region}, but its KeyValueStore {kvs} exists: "
                  f"the gallery lives in another region of this account, and CloudFront names are account-global, "
                  f"so a second one fails with AlreadyExists. Deploy with --no-gallery, or in the gallery's region.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[gallery] Deploying the shared video history stack {GALLERY_STACK_NAME} in {region} ...")
        deploy_gallery_stack(region)
        outputs = stack_outputs(region, GALLERY_STACK_NAME)
        if outputs is None:
            print(f"[error] {GALLERY_STACK_NAME} is not in a usable state after deploy.", file=sys.stderr)
            sys.exit(1)
    salt = get_or_create_salt(region)
    gallery_id = email_hash(salt, email)
    password = get_password(region, gallery_id)
    if password is None:
        password = new_password()
        put_password(region, gallery_id, password)
        print(f"[gallery] Created a password for {email} (show it with: python vlasim.py gallery)")
    if kvs_put(outputs["KvsArn"], gallery_id, password_digest(gallery_id, password)):
        print("[gallery] Password registered in the CloudFront KeyValueStore (edge propagation takes a few seconds).")
    # Seed an empty page so the URL works before the first video arrives (no 403 from S3).
    try:
        _boto3().client("s3", region_name=region).head_object(
            Bucket=outputs["GalleryBucketName"], Key=f"g/{gallery_id}/index.html")
    except Exception:  # noqa: BLE001 - missing (404) or unreadable: (re)build it
        rebuild_index(outputs["GalleryBucketName"], f"g/{gallery_id}/", region)
    domain = outputs["DistributionDomainName"]
    return {
        "bucket": outputs["GalleryBucketName"],
        "domain": domain,
        "id": gallery_id,
        "url": gallery_url(domain, gallery_id),
        "kvs_arn": outputs["KvsArn"],
    }


def reset_password(region: str, email: str) -> dict:
    outputs = stack_outputs(region, GALLERY_STACK_NAME)
    if outputs is None:
        raise LookupError(f"{GALLERY_STACK_NAME} is not deployed in {region}")
    salt = get_or_create_salt(region, create=False)
    if salt is None:
        raise LookupError(f"no gallery salt in {region} (SSM {SALT_PARAM})")
    gallery_id = email_hash(salt, email)
    password = new_password()
    put_password(region, gallery_id, password)
    kvs_put(outputs["KvsArn"], gallery_id, password_digest(gallery_id, password))
    return {"url": gallery_url(outputs["DistributionDomainName"], gallery_id), "password": password}


def lookup(region: str, email: str) -> dict:
    """URL + password for an e-mail (read-only)."""
    outputs = stack_outputs(region, GALLERY_STACK_NAME)
    if outputs is None:
        raise LookupError(f"{GALLERY_STACK_NAME} is not deployed in {region}")
    salt = get_or_create_salt(region, create=False)
    if salt is None:
        raise LookupError(f"no gallery salt in {region} (SSM {SALT_PARAM})")
    gallery_id = email_hash(salt, email)
    return {"url": gallery_url(outputs["DistributionDomainName"], gallery_id),
            "password": get_password(region, gallery_id), "id": gallery_id}


def delete_prefix(bucket: str, prefix: str, region: str) -> int:
    """Delete every object under prefix ("" = whole bucket). Returns the number deleted."""
    s3 = _boto3().client("s3", region_name=region)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
            n += len(keys)
    return n


def delete_bucket(bucket: str, region: str):
    """Empty and delete a bucket (non-versioned). Returns objects deleted; None when it does not exist."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    s3 = _boto3().client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        return None
    n = delete_prefix(bucket, "", region)
    s3.delete_bucket(Bucket=bucket)
    return n


def rebuild_index(bucket: str, gallery_prefix: str, region: str) -> None:
    """Rebuild g/<id>/index.html with the Lambda's own builder (after objects were removed)."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "gallery_publisher", BASE_DIR / "cdk" / "lambda" / "gallery_publisher" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._s3 = _boto3().client("s3", region_name=region)
    mod.rebuild_index(bucket, gallery_prefix)


def backfill(region: str, results_bucket: str, gallery_bucket: str, target_prefix: str) -> int:
    """Copy videos of earlier runs (already in the results bucket) into the gallery; return the count.

    target_prefix = g/<id>/<target>/ (the target stack's GalleryPrefix output). Existing gallery
    objects are skipped, so this is safe to repeat. The index is rebuilt once at the end.
    """
    s3 = _boto3().client("s3", region_name=region)
    have = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=gallery_bucket, Prefix=target_prefix):
        have.update(o["Key"] for o in page.get("Contents", []))
    src = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=results_bucket):
        src.update(o["Key"] for o in page.get("Contents", []))
    copied = 0
    for key in sorted(src):
        if key.endswith(".mp4"):
            pairs = [key]
            caption = key[: -len(".mp4")] + ".txt"
            if caption in src:
                pairs.append(caption)
        else:
            continue
        for k in pairs:
            parts = k.split("/")
            if len(parts) < 2 or any(p in ("", ".", "..") for p in parts):
                continue
            dest = f"{target_prefix}{k}"
            if dest in have:
                continue
            s3.copy_object(CopySource={"Bucket": results_bucket, "Key": k}, Bucket=gallery_bucket, Key=dest,
                           MetadataDirective="REPLACE",
                           ContentType="video/mp4" if k.endswith(".mp4") else "text/plain; charset=utf-8",
                           CacheControl="public, max-age=86400")
            copied += 1
    if copied:
        rebuild_index(gallery_bucket, "/".join(target_prefix.split("/")[:2]) + "/", region)
    return copied


def delete_ssm_params(region: str) -> int:
    """Delete the salt and every gallery password in the region."""
    ssm = _boto3().client("ssm", region_name=region)
    names = []
    for page in ssm.get_paginator("get_parameters_by_path").paginate(Path=SSM_PREFIX, Recursive=True):
        names.extend(p["Name"] for p in page.get("Parameters", []))
    for i in range(0, len(names), 10):
        ssm.delete_parameters(Names=names[i:i + 10])
    return len(names)
