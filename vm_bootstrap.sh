#!/bin/bash
# One-shot VM bootstrap (run ON the GPU VM). Expects either ~/fedcrag.bundle
# (scp'd git bundle) or an already-cloned ~/FedCRAG. Idempotent.
set -euo pipefail
cd ~
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y -q tmux

if [ ! -d FedCRAG ]; then
    git clone -b w3-campaign "$HOME/fedcrag.bundle" FedCRAG
fi
cd FedCRAG

[ -d .venv ] || uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest

.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE — driver missing? run nvidia-smi"
print("CUDA OK:", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB")
EOF

.venv/bin/python -m pytest tests/ -q
bash run_w3.sh smoke
echo "BOOTSTRAP COMPLETE"
