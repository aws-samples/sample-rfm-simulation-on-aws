#!/usr/bin/env python3
"""
zmq_grpc_bridge.py — ZMQ ↔ gRPC bridge for GR00T bridge mode

The simulation EC2 instance does not load the model locally; it delegates inference to a
remote GR00T gRPC endpoint (behind an NLB).

  rollout_policy.py (ZMQ REQ :8000) → [bridge] → gRPC NLB:50051

ZMQ REP(:8000) : talks to the existing rollout_policy.py (no Isaac-GR00T code changes)
gRPC stub      : calls the remote endpoint NLB(:50051)

Message conversion flow:
  ZMQ recv  → MsgSerializer { endpoint envelope + obs dict (numpy __ndarray_class__) }
  gRPC send → InferRequest { image_data(JPEG), instruction, joint states }
  gRPC recv → InferResponse { action_chunks map<string, bytes> }
  ZMQ send  → MsgSerializer { [action_dict, {}] } — matches PolicyServer response format

Protocol reference: Isaac-GR00T gr00t/policy/server_client.py (MsgSerializer, PolicyServer, PolicyClient)

Usage:
  python zmq_grpc_bridge.py --endpoint NLB_HOST:50051 [--zmq-port 8000]

Extra dependencies (beyond the Isaac-GR00T venv defaults):
  pip install "grpcio>=1.80.0" Pillow
  (pyzmq, numpy and msgpack are already in the Isaac-GR00T venv)
"""

import argparse
import io
import logging
import sys

import msgpack
import numpy as np
import zmq
from PIL import Image

import grpc
import gr00t_pb2
import gr00t_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="[bridge] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Per-joint DOF — used to restore action_chunks bytes → numpy (H, DOF)
_DOF_MAP = {
    "left_arm":   7,
    "right_arm":  7,
    "left_hand":  6,
    "right_hand": 6,
    "waist":      3,
}


# ── MsgSerializer ─────────────────────────────────────────────────────────────
# Same serialization as Isaac-GR00T PolicyClient/PolicyServer (see server_client.py)
# numpy array: {"__ndarray_class__": True, "as_npy": <bytes>}

class MsgSerializer:
    @staticmethod
    def to_bytes(data) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False)

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"[bridge] Cannot encode type: {type(obj)}")


# ── Image encoding ─────────────────────────────────────────────────────────────

def _encode_image(image_np: np.ndarray) -> bytes:
    """numpy (H, W, C) uint8 → JPEG bytes"""
    buf = io.BytesIO()
    Image.fromarray(image_np).convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ── Obs → gRPC InferRequest ───────────────────────────────────────────────────

def _obs_to_infer_request(obs: dict) -> "gr00t_pb2.InferRequest":
    """
    Gymnasium VecEnv obs dict → gRPC InferRequest

    In the PolicyClient → PolicyServer (Gr00tSimPolicyWrapper) protocol
    obs uses the Gr00t sim flat format:
      {
        "video.<key>":  np.ndarray(n_envs, T, H, W, C) uint8,
        "state.<key>":  np.ndarray(n_envs, T, DOF),
        "annotation.human.task_description": np.ndarray or [[str]]
      }
    The nested format ({"video": {...}}) is also supported as a fallback.
    """
    # ── Video ───────────────────────────────────────────────────────────────
    # Flat format: try "video.<cam_key>" first
    video_dict = {k[len("video."):]: v for k, v in obs.items() if k.startswith("video.")}
    if not video_dict:
        # Nested format fallback
        video_dict = obs.get("video", {})
    if not video_dict:
        raise ValueError("obs missing 'video.*' key (checked flat and nested)")

    image = next(iter(video_dict.values()))
    while image.ndim > 3:
        image = image[0]  # drop batch dims in order: n_envs=0, T=0, ...
    image_bytes = _encode_image(np.asarray(image, dtype=np.uint8))

    # ── Language / instruction ────────────────────────────────────────────────
    # Flat format: "annotation.human.task_description" or "annotation.human.coarse_action"
    instruction = ""
    for lang_key in [
        "annotation.human.task_description",
        "annotation.human.coarse_action",
        "task",
    ]:
        if lang_key in obs:
            raw = obs[lang_key]
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            # Unwrap nested lists until we get a scalar
            while isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            instruction = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            break
    if not instruction:
        # Nested format fallback
        task = obs.get("language", {}).get("task", [[""]])
        if hasattr(task, "tolist"):
            task = task.tolist()
        row = task[0] if task else [""]
        if hasattr(row, "tolist"):
            row = row.tolist()
        raw_instr = row[0] if row else ""
        instruction = raw_instr.decode("utf-8") if isinstance(raw_instr, bytes) else str(raw_instr)

    # ── State: joint arrays → float32 lists ────────────────────────────────────
    # Flat format: "state.<joint_key>"
    state_flat = {k[len("state."):]: v for k, v in obs.items() if k.startswith("state.")}
    state_nested = obs.get("state", {})
    state = state_flat if state_flat else state_nested

    def _flat_state(key):
        arr = state.get(key)
        if arr is None:
            return []
        return np.asarray(arr, dtype=np.float32).flatten().tolist()

    return gr00t_pb2.InferRequest(
        image_data=image_bytes,
        instruction=instruction,
        left_arm=_flat_state("left_arm"),
        right_arm=_flat_state("right_arm"),
        left_hand=_flat_state("left_hand"),
        right_hand=_flat_state("right_hand"),
        waist=_flat_state("waist"),
    )


# ── gRPC InferResponse → action dict ─────────────────────────────────────────

def _infer_resp_to_action(resp: "gr00t_pb2.InferResponse") -> dict:
    """
    gRPC InferResponse.action_chunks → action dict with shaped numpy arrays

    InferResponse.action_chunks: map<string, bytes> (float32 tobytes())
    Restore: np.frombuffer → reshape (B=1, H, DOF)
    This must match the action_dict returned by PolicyServer (local Gr00tPolicy).
    """
    action_dict = {}
    for k, v_bytes in resp.action_chunks.items():
        arr = np.frombuffer(v_bytes, dtype=np.float32).copy()
        dof = _DOF_MAP.get(k)
        if dof and len(arr) % dof == 0:
            arr = arr.reshape(1, len(arr) // dof, dof)  # (B=1, H, DOF)
        # add the "action." prefix — same as the Gr00tSimPolicyWrapper output format
        # gr00t_policy.py: return {f"action.{key}": action[key] for key in action}, info
        action_dict[f"action.{k}"] = arr
    return action_dict


# ── Bridge main loop ──────────────────────────────────────────────────────────

def run(endpoint: str, zmq_port: int) -> None:
    # strip the grpc:// prefix
    host = endpoint[len("grpc://"):] if endpoint.startswith("grpc://") else endpoint

    # gRPC channel → NLB
    channel = grpc.insecure_channel(
        host,
        options=[
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
        ],
    )
    stub = gr00t_pb2_grpc.GR00TInferenceStub(channel)
    log.info("gRPC channel → %s", host)

    # ZMQ REP socket — drop-in replacement for PolicyServer
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{zmq_port}")
    log.info("ZMQ REP listening on :%d", zmq_port)

    req_count = 0
    try:
        while True:
            raw = sock.recv()

            # MsgSerializer decode — the format PolicyClient sends
            try:
                request = MsgSerializer.from_bytes(raw)
            except Exception as exc:
                log.error("Request decode failed: %s", exc)
                sock.send(MsgSerializer.to_bytes({"error": str(exc)}))
                continue

            endpoint_name = request.get("endpoint", "get_action")

            # ── non-inference endpoints ────────────────────────────────────────────
            if endpoint_name == "ping":
                sock.send(MsgSerializer.to_bytes({"status": "ok", "message": "Server is running"}))
                continue
            elif endpoint_name == "reset":
                sock.send(MsgSerializer.to_bytes({}))
                continue
            elif endpoint_name == "get_modality_config":
                sock.send(MsgSerializer.to_bytes({}))
                continue
            elif endpoint_name == "kill":
                sock.send(MsgSerializer.to_bytes({"status": "ok"}))
                log.info("Kill received — bridge exiting.")
                break
            elif endpoint_name != "get_action":
                log.warning("Unknown endpoint: %s", endpoint_name)
                sock.send(MsgSerializer.to_bytes({"error": f"Unknown endpoint: {endpoint_name}"}))
                continue

            # ── get_action ───────────────────────────────────────────────────
            data = request.get("data", {})
            obs = data.get("observation", {})

            # obs → gRPC InferRequest
            try:
                infer_req = _obs_to_infer_request(obs)
            except Exception as exc:
                log.error("obs → InferRequest failed: %s", exc)
                # return an empty response in PolicyServer format: [action_dict, info_dict]
                sock.send(MsgSerializer.to_bytes([{}, {}]))
                continue

            # call gRPC Infer
            try:
                infer_resp = stub.Infer(infer_req, timeout=10.0)
            except grpc.RpcError as exc:
                log.error("gRPC Infer failed: %s %s", exc.code(), exc.details())
                sock.send(MsgSerializer.to_bytes([{}, {}]))
                continue

            # InferResponse → action dict → MsgSerializer → ZMQ REP
            # return format: [action_dict, info_dict] (same as PolicyServer)
            # PolicyClient._get_action: return tuple(response) → (action_dict, {})
            action_dict = _infer_resp_to_action(infer_resp)
            sock.send(MsgSerializer.to_bytes([action_dict, {}]))

            req_count += 1
            if req_count % 10 == 0:
                log.info("Processed %d inference requests", req_count)

    except KeyboardInterrupt:
        log.info("Bridge stopped.")
    finally:
        sock.close()
        ctx.term()
        channel.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZMQ ↔ gRPC bridge — GR00T bridge (remote inference) mode"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="gRPC endpoint (e.g. NLB_HOST:50051 or grpc://NLB_HOST:50051)",
    )
    parser.add_argument(
        "--zmq-port",
        type=int,
        default=8000,
        help="ZMQ REP listen port (default: 8000 — same as PolicyServer)",
    )
    args = parser.parse_args()
    run(args.endpoint, args.zmq_port)


if __name__ == "__main__":
    main()
