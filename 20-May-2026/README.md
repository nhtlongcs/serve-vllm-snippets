# run-llm

OpenAI-compatible LLM inference server powered by vLLM, managed with `uv`.

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.12 |
| Package manager | uv |
| GPU | NVIDIA (Ampere or newer recommended) |
| CUDA toolkit | 12.8 (miniforge3) |
| CUDA driver | ≥ 12.x (tested: 555.42.06 / CUDA 12.5) |

> **Note on CUDA versions:** The installed packages use `torch==2.11.0+cu128` (CUDA 12.8 build) and `vllm` from the cu129 nightly wheel server. Both link against `libcudart.so.12`, so any CUDA 12.x driver is sufficient. Do **not** install the default PyPI `torch` or `vllm` — their wheels are compiled for CUDA 13 and will fail on CUDA 12.x drivers.

---

## Installation

```bash
uv sync
```

This resolves all dependencies from the configured indices in `pyproject.toml` (PyTorch cu128 + vLLM cu129 nightly). No manual pip commands needed.

---

## Running the Server

```bash
uv run python main.py
```

Starts an OpenAI-compatible API server at `http://0.0.0.0:8000` serving `Qwen3.5-27B` by default.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `/home/support/llm/Qwen3.5-27B` | Model path or HuggingFace ID |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |
| `--tensor-parallel-size` | `1` | Number of GPUs |
| `--max-model-len` | *(model default)* | Max context length |
| `--dtype` | `bfloat16` | Tensor dtype |
| `--served-model-name` | *(basename of model path)* | Model name in API responses |
| `--allowed-local-media-path` | `/` | Root directory for `file://` image access |

Any unrecognized flags are forwarded directly to vLLM.

```bash
# Examples
uv run python main.py --port 8080
uv run python main.py --model /home/support/llm/Qwen2.5-14B-Instruct --max-model-len 32768
uv run python main.py --enable-prefix-caching   # extra vllm flag, passed through
```

---

## API Usage

The server is OpenAI API-compatible. Use any OpenAI SDK or plain HTTP.

### Text

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen3.5-27B",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
print(response.choices[0].message.content)
```

### Single Image

```python
response = client.chat.completions.create(
    model="Qwen3.5-27B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe this image."}
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
```

### Multiple Sequential Images

Multiple `image_url` blocks in a single message are supported:

```python
response = client.chat.completions.create(
    model="Qwen3.5-27B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "...image1..."}},
        {"type": "text", "text": "Image 1."},
        {"type": "image_url", "image_url": {"url": "...image2..."}},
        {"type": "text", "text": "Image 2. Compare both."}
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
```

### Thinking Mode

`Qwen3.5-27B` is a reasoning model and produces a thinking preamble by default. Disable it per-request:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

---

## Using Local Images (Server A / Server B Setup)

This applies when **vLLM runs on Server A** and your **client scripts run on Server B**.

Since vLLM and the image files are co-located on Server A, use `file://` paths. vLLM reads the file directly from disk — no image transfer or hosting required.

**Server A** must be started with `--allowed-local-media-path` pointing to the image root (default: `/`):

```bash
# Server A
uv run python main.py --allowed-local-media-path /home/tnguyenho/workspace/lsc-adl/images
```

**Server B** references images by their absolute path on Server A:

```python
# Server B
import openai
client = openai.OpenAI(base_url="http://<server-a-ip>:8000/v1", api_key="none")

img_path = "/home/tnguyenho/workspace/lsc-adl/images/images/origin/202003/08/20200308_122924_000.jpg"

response = client.chat.completions.create(
    model="Qwen3.5-27B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"file://{img_path}"}},
        {"type": "text", "text": "Describe this image."}
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
print(response.choices[0].message.content)
```

> The `file://` scheme is only resolved on the server side. Server B never needs to read or transfer the image file itself.

---

## Dependency Configuration Notes

> **Required for CUDA 12.x environments only.** If your driver supports CUDA 13+, a standard `pip install vllm` should work without this configuration.

The latest vLLM on PyPI is built against CUDA 13 (`libcudart.so.13`). On a CUDA 12.x system the import fails immediately. This project pins PyTorch and vLLM to CUDA 12.x-compatible builds:

- **PyTorch** → installed from the `cu128` index (same version, different CUDA build)
- **vLLM** → installed from the vLLM cu129 nightly wheel server (links `libcudart.so.12`)
- **torchvision / torchaudio** → overridden to match, preventing PyPI from selecting the cu130 builds

The `pyproject.toml` handles all of this. Running `uv sync` is sufficient.
