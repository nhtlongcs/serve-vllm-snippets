"""Probe a vLLM-served model for the max images per chat request.

Boots vllm.entrypoints.openai.api_server with one of the configs in configs/,
waits for /v1/models, then sends chat-completion requests with an
exponentially growing image count, falling back to bisection once a request
fails. Reports the largest image count that succeeded.

Usage:
    python test_max_images.py configs/gemma4-31b.json --max-probe 32
"""

import argparse
import base64
import io
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Reuse config plumbing from main.py.
sys.path.insert(0, str(Path(__file__).parent))
from main import _augment_library_path, config_to_cli_args, load_config  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_image_b64(size: int, seed: int = 0) -> str:
    from PIL import Image
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    import signal
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _build_cmd(cfg: dict, host: str, port: int, allowed_path: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", host,
        "--port", str(port),
    ]
    cmd += config_to_cli_args(cfg)
    cmd += ["--allowed-local-media-path", allowed_path]
    cmd += cfg.get("server_extra_args", [])
    return cmd


def _wait_ready(base_url: str, model_name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
                if r.status == 200:
                    body = json.loads(r.read().decode())
                    ids = {m.get("id") for m in body.get("data", [])}
                    if model_name in ids or body.get("data"):
                        return True
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


def _post_chat(base_url: str, model_name: str, img_b64: str, n_images: int, timeout: float):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        for _ in range(n_images)
    ]
    content.append({"type": "text", "text": "Reply with the single word OK."})
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")[:1000]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:1000]
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return -1, repr(e)


def _classify_failure(status: int, body: str) -> str:
    txt = body.lower()
    if "out of memory" in txt or "cuda out of memory" in txt or "cuda oom" in txt:
        return "oom"
    if "maximum context" in txt or "longer than the maximum" in txt or "max_model_len" in txt or "exceeds" in txt and "length" in txt:
        return "context_overflow"
    if "limit_mm_per_prompt" in txt or "limit per prompt" in txt or "too many" in txt and "image" in txt:
        return "mm_limit"
    if status == -1:
        return "transport_error"
    return f"http_{status}"


def probe(base_url, model_name, img_b64, max_probe, timeout):
    """Doubling search + bisection."""
    best = 0
    failed_at = None
    last_status, last_body = 0, ""

    n = 1
    while n <= max_probe:
        print(f"  probing n={n}...", flush=True)
        status, body = _post_chat(base_url, model_name, img_b64, n, timeout)
        last_status, last_body = status, body
        if status == 200:
            best = n
            if n == max_probe:
                break
            n = min(n * 2, max_probe)
        else:
            failed_at = n
            print(f"    failed: {_classify_failure(status, body)} :: {body[:200]}", flush=True)
            break

    if failed_at is None or best >= failed_at - 1:
        return best, last_status, last_body

    lo, hi = best + 1, failed_at - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"  bisect n={mid} [{lo}..{hi}]...", flush=True)
        status, body = _post_chat(base_url, model_name, img_b64, mid, timeout)
        last_status, last_body = status, body
        if status == 200:
            best = mid
            lo = mid + 1
        else:
            print(f"    failed: {_classify_failure(status, body)}", flush=True)
            hi = mid - 1
    return best, last_status, last_body


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="Path to a config JSON in configs/")
    parser.add_argument("--max-probe", type=int, default=64, help="Upper bound on images per request to probe")
    parser.add_argument("--image-size", type=int, default=448, help="Side length of the random test image (px)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="Port for the local server (default: pick free)")
    parser.add_argument("--ready-timeout", type=float, default=1800, help="Seconds to wait for server readiness")
    parser.add_argument("--req-timeout", type=float, default=300, help="Per-request HTTP timeout")
    parser.add_argument("--out", help="Write JSON result to this file")
    parser.add_argument("--keep-server", action="store_true", help="Leave the server running after the probe (for debugging)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Make sure the server allows enough images per prompt to reach max_probe.
    existing = dict(cfg.get("limit_mm_per_prompt") or {})
    existing["image"] = max(args.max_probe, existing.get("image", 0))
    cfg["limit_mm_per_prompt"] = existing

    if "served_model_name" not in cfg:
        cfg["served_model_name"] = Path(cfg["model"].rstrip("/")).name
    model_name = cfg["served_model_name"]

    port = args.port or _free_port()
    base_url = f"http://{args.host}:{port}"
    cmd = _build_cmd(cfg, args.host, port, "/")

    print(f"Spawning server: {' '.join(cmd)}\n", flush=True)

    env = os.environ.copy()
    _augment_library_path(env)

    # Run in a fresh session so we can SIGTERM the whole tree: vllm's api_server
    # spawns EngineCore as a subprocess and a plain proc.terminate() only kills
    # the parent — the child reparents to init and keeps holding the GPU.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    result = {
        "config": args.config,
        "model": cfg["model"],
        "served_model_name": model_name,
        "image_size_px": args.image_size,
        "max_probe": args.max_probe,
        "limit_mm_per_prompt": cfg["limit_mm_per_prompt"],
        "max_model_len": cfg.get("max_model_len"),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization"),
        "max_images_ok": None,
        "last_status": None,
        "last_failure": None,
    }
    try:
        print(f"Waiting for {base_url}/v1/models (up to {args.ready_timeout}s)...", flush=True)
        if not _wait_ready(base_url, model_name, args.ready_timeout):
            print(f"Server did not become ready within {args.ready_timeout}s", flush=True)
            return 2
        print(f"Server ready. Generating random {args.image_size}x{args.image_size} test image...", flush=True)
        img_b64 = _make_image_b64(args.image_size)

        best, status, body = probe(base_url, model_name, img_b64, args.max_probe, args.req_timeout)
        result["max_images_ok"] = best
        result["last_status"] = status
        if status != 200:
            result["last_failure"] = {
                "category": _classify_failure(status, body),
                "snippet": body[:400],
            }
        print("\n=== RESULT ===")
        print(json.dumps(result, indent=2))
    finally:
        if not args.keep_server:
            _terminate_process_group(proc)
        else:
            print(f"Leaving server running (PID {proc.pid}, pgid {os.getpgid(proc.pid)}) on {base_url}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0 if result["max_images_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
