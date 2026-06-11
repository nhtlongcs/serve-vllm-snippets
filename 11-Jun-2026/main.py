import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

# Keys consumed by this launcher; everything else in a config file is treated
# as a vLLM CLI option (key -> --key, value JSON-encoded if non-scalar).
_LAUNCHER_KEYS = {"_notes", "server_extra_args"}
# Keys that map to CLI flags by convention (underscore -> hyphen).
_SCALAR_FLAG_KEYS = {
    "model",
    "served_model_name",
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
}
_BOOL_FLAG_KEYS = {"trust_remote_code", "enforce_eager", "enable_prefix_caching"}
_JSON_FLAG_KEYS = {"limit_mm_per_prompt", "mm_processor_kwargs", "media_io_kwargs"}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def config_to_cli_args(cfg: dict) -> list[str]:
    args: list[str] = []
    for key, value in cfg.items():
        if key in _LAUNCHER_KEYS:
            continue
        if key in _BOOL_FLAG_KEYS:
            if value:
                args.append(_flag(key))
            continue
        if key in _JSON_FLAG_KEYS:
            args += [_flag(key), json.dumps(value)]
            continue
        if key in _SCALAR_FLAG_KEYS or isinstance(value, (str, int, float)):
            args += [_flag(key), str(value)]
            continue
        # Fallback: JSON-encode anything we don't recognize.
        args += [_flag(key), json.dumps(value)]
    return args


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Host LLM using vLLM (OpenAI-compatible API)")
    parser.add_argument("--config", help="Path to a JSON config file (see configs/)")
    parser.add_argument("--model", help="Path to model or HuggingFace model ID")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, help="Maximum context length")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--served-model-name", help="Model name exposed via /v1 API")
    parser.add_argument("--allowed-local-media-path", default="/", help="Root directory vLLM may read local file:// images from")
    args, extra_args = parser.parse_known_args()

    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    # CLI overrides config.
    overrides = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "served_model_name": args.served_model_name,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if "model" not in cfg:
        parser.error("--model or --config (with a 'model' key) is required")

    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("tensor_parallel_size", 1)
    if "served_model_name" not in cfg:
        cfg["served_model_name"] = Path(cfg["model"].rstrip("/")).name

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
    ]
    cmd += config_to_cli_args(cfg)
    cmd += ["--allowed-local-media-path", args.allowed_local_media_path]
    cmd += cfg.get("server_extra_args", [])
    cmd += extra_args

    print(f"Starting vLLM server: {cfg['served_model_name']} on {args.host}:{args.port}")
    print(f"Model path: {cfg['model']}")
    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    _augment_library_path(env)
    subprocess.run(cmd, check=True, env=env)


def _augment_library_path(env: dict) -> None:
    """Wire CUDA + libstdc++ from the active conda env into the spawned process.

    - LIBRARY_PATH: build-time link path. flashinfer JITs against libcuda.so,
      which conda's cuda-toolkit drops into $CONDA_PREFIX/lib/stubs.
    - LD_LIBRARY_PATH: dlopen path. The .so flashinfer produces is linked
      against conda's libstdc++ (GLIBCXX_3.4.32+), which the host system may
      not provide; prepending $CONDA_PREFIX/lib makes dlopen find it.
    """
    conda_prefix = env.get("CONDA_PREFIX")
    link_candidates: list[str] = []
    runtime_candidates: list[str] = []
    if conda_prefix:
        link_candidates += [
            f"{conda_prefix}/lib/stubs",
            f"{conda_prefix}/targets/x86_64-linux/lib/stubs",
            f"{conda_prefix}/lib",
        ]
        runtime_candidates += [
            f"{conda_prefix}/lib",
            f"{conda_prefix}/lib/stubs",
        ]
    link_candidates.append("/usr/lib/x86_64-linux-gnu")

    link_paths = [d for d in link_candidates if os.path.isdir(d)]
    if link_paths:
        prev = env.get("LIBRARY_PATH", "")
        env["LIBRARY_PATH"] = ":".join(link_paths) + (":" + prev if prev else "")

    runtime_paths = [d for d in runtime_candidates if os.path.isdir(d)]
    if runtime_paths:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(runtime_paths) + (":" + prev if prev else "")


if __name__ == "__main__":
    main()
