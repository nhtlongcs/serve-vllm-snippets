import argparse
import os
import subprocess
import sys

DEFAULT_MODEL = "/home/support/llm/Qwen3.5-27B"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def main():
    parser = argparse.ArgumentParser(description="Host LLM using vLLM (OpenAI-compatible API)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to model or HuggingFace model ID")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum context length")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--served-model-name", default=None, help="Model name exposed via /v1 API (defaults to model path basename)")
    parser.add_argument("--allowed-local-media-path", default="/", help="Root directory vLLM may read local file:// images from")
    args, extra_args = parser.parse_known_args()

    served_name = args.served_model_name or args.model.rstrip("/").split("/")[-1]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--served-model-name", served_name,
    ]

    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]

    cmd += ["--allowed-local-media-path", args.allowed_local_media_path]

    cmd += extra_args

    print(f"Starting vLLM server: {served_name} on {args.host}:{args.port}")
    print(f"Model path: {args.model}")
    print(f"Command: {' '.join(cmd)}\n")

    env = os.environ.copy()
    # flashinfer JIT links against libcuda.so; miniforge3 puts it in lib/stubs, not lib64/stubs
    cuda_stubs = "/home/tnguyenho/miniforge3/lib/stubs:/usr/lib/x86_64-linux-gnu"
    env["LIBRARY_PATH"] = cuda_stubs + (":" + env["LIBRARY_PATH"] if "LIBRARY_PATH" in env else "")

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
