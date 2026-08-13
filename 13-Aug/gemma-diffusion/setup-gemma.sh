#!/usr/bin/env bash
# Installs vllm nightly (cu129, dev500+) which includes DiffusionGemma support
# from PR vllm-project/vllm#45163 (merged 2026-06-12).
# Uses /home/tnguyenho/miniforge3 — that's where the vllm CLI lives.
set -euo pipefail

MINIFORGE=/home/tnguyenho/miniforge3
WHEEL="https://wheels.vllm.ai/b3f0a0a0df76dda92ec4b2c9335f77e84adad911/vllm-0.22.1rc1.dev500%2Bgb3f0a0a0d.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "==> Installing vllm nightly with DiffusionGemma support into miniforge3..."
"$MINIFORGE/bin/pip" install "$WHEEL"

echo "==> Verifying DiffusionGemmaForBlockDiffusion registration..."
LD_LIBRARY_PATH="$MINIFORGE/lib:${LD_LIBRARY_PATH:-}" \
"$MINIFORGE/bin/python3" - <<'EOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert 'DiffusionGemmaForBlockDiffusion' in archs, "DiffusionGemmaForBlockDiffusion NOT in supported archs!"
print("OK - DiffusionGemmaForBlockDiffusion is registered.")

from vllm.transformers_utils.config import get_config
cfg = get_config('/spinning/tnguyenho/llm/diffusiongemma-26B-A4B-it', trust_remote_code=False)
assert cfg.model_type == 'diffusion_gemma', f"Unexpected model_type: {cfg.model_type}"
assert cfg.canvas_length == 256, f"Unexpected canvas_length: {cfg.canvas_length}"
print(f"OK - Model config: {cfg.architectures}, canvas_length={cfg.canvas_length}")
print("Setup complete. Run ./serve_diffusion_gemma.sh to start the server.")
EOF
