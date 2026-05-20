# serve-vllm-snippets

A collection of self-contained scripts and configurations for serving LLMs with [vLLM](https://github.com/vllm-project/vllm).

Each subdirectory is a dated snapshot — an independent, runnable setup with its own dependencies and README.

## Snippets

| Directory | Description |
|---|---|
| [20-May-2026](./20-May-2026/) | OpenAI-compatible vLLM server with CUDA 12.x support, managed via `uv` |

## Structure

```
serve-vllm-snippets/
└── <date>/          # self-contained snippet
    ├── main.py
    ├── pyproject.toml
    └── README.md
```

Each snippet has its own README with installation and usage instructions.
