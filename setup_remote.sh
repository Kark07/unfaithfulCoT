#!/usr/bin/env bash
# Setup for the SHARED A6000 server.
#
# Design constraints, because the box has other users on it:
#   * nothing is installed globally -- everything lives in ./.venv
#   * no sudo, no system package manager, no touching anyone's dotfiles
#   * HuggingFace caches are redirected INTO this project directory, so model
#     downloads (several GB) do not fill the shared home quota or race against
#     another user's cache
#   * the GPU is shared: we check free VRAM and never assume device 0 is ours
#
# Usage:
#   bash setup_remote.sh          # one time
#   source env.sh                 # every new shell afterwards
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$PROJECT_DIR/.venv}"

# ---------------------------------------------------------------------------
# 0. pick a python without changing anything global
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for cand in python3.11 python3.10 python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi
[[ -z "$PYTHON" ]] && { echo "!! no python3 found; set PYTHON=/path/to/python"; exit 1; }
PYVER="$($PYTHON -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "=== python: $PYTHON ($PYVER) ==="
case "$PYVER" in
  3.10|3.11|3.12) ;;
  *) echo "  !! untested python $PYVER -- 3.10-3.12 recommended" ;;
esac

# ---------------------------------------------------------------------------
# 1. project-local caches (BEFORE any pip/HF call)
# ---------------------------------------------------------------------------
echo "=== 1. project-local caches ==="
cat > "$PROJECT_DIR/env.sh" <<EOF
# Source this in every new shell:  source env.sh
export PROJECT_DIR="$PROJECT_DIR"

# Keep every cache inside the project so we never touch the shared home quota.
export HF_HOME="\$PROJECT_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="\$PROJECT_DIR/hf_cache/hub"
export TRANSFORMERS_CACHE="\$PROJECT_DIR/hf_cache/transformers"
export HF_DATASETS_CACHE="\$PROJECT_DIR/hf_cache/datasets"
export TORCH_HOME="\$PROJECT_DIR/.cache/torch"
export PIP_CACHE_DIR="\$PROJECT_DIR/.cache/pip"
export TOKENIZERS_PARALLELISM=false

# The GPU is shared. Pin ours explicitly, e.g.  export CUDA_VISIBLE_DEVICES=0
# Run  nvidia-smi  first and pick a device nobody else is on.
: "\${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

source "\$PROJECT_DIR/.venv/bin/activate"
echo "env ready | HF_HOME=\$HF_HOME | CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
EOF
mkdir -p "$PROJECT_DIR/hf_cache" "$PROJECT_DIR/.cache" \
         "$PROJECT_DIR/results" "$PROJECT_DIR/logs"
# shellcheck disable=SC1091
export HF_HOME="$PROJECT_DIR/hf_cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
echo "  caches -> $PROJECT_DIR/{hf_cache,.cache}"

# ---------------------------------------------------------------------------
# 2. isolated virtualenv
# ---------------------------------------------------------------------------
echo "=== 2. virtualenv -> $VENV ==="
if [[ -d "$VENV" ]]; then
  echo "  exists, reusing (delete it to rebuild)"
else
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -qU pip wheel setuptools
echo "  pip: $(python -m pip --version | cut -d' ' -f2)  ($(which python))"

# ---------------------------------------------------------------------------
# 3. torch -- Blackwell (sm_120) needs cu128 wheels
# ---------------------------------------------------------------------------
echo "=== 3. torch (cu128 / Blackwell sm_120) ==="
pip install -q torch --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# 4. libraries
# ---------------------------------------------------------------------------
echo "=== 4. core libs ==="
pip install -q \
  "transformers>=4.46" "datasets>=3.0" "accelerate>=1.0" \
  einops pandas numpy scipy matplotlib seaborn tqdm

echo "=== 5. circuit tooling (needed from E2 on, NOT for E0) ==="
pip install -q transformer_lens || \
  echo "  !! transformer_lens failed -- E0 does not need it; fix before E2"

# ---------------------------------------------------------------------------
# 6. verification
# ---------------------------------------------------------------------------
echo "=== 6. verification ==="
python - <<'PY'
import os, sys, torch
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  !! NO GPU VISIBLE. Check CUDA_VISIBLE_DEVICES and the driver.")
    sys.exit(1)
i = torch.cuda.current_device()
cap = torch.cuda.get_device_capability(i)
free, total = torch.cuda.mem_get_info(i)
print(f"  device           {torch.cuda.get_device_name(i)} (idx {i})")
print(f"  capability       sm_{cap[0]}{cap[1]}")
print(f"  vram             {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
if cap[0] >= 12 and not torch.__version__.split('+')[0] >= "2.7":
    print("  !! Blackwell needs torch >= 2.7 built for cu128")
if free / 1e9 < 12:
    print(f"  !! only {free/1e9:.1f} GB free -- another user may be on this GPU.")
    print("     Run nvidia-smi, pick a free device, re-export CUDA_VISIBLE_DEVICES.")
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(f"  matmul smoke     ok ({(x @ x).sum().item():.1f})")
print(f"  HF_HOME          {os.environ.get('HF_HOME')}")
try:
    import transformer_lens
    print(f"  transformer_lens {transformer_lens.__version__}")
except Exception as e:
    print(f"  transformer_lens MISSING ({type(e).__name__}) -- fine for E0")
PY

echo
echo "=== done ==="
echo "  every new shell:   source env.sh"
echo "  check the GPU:     nvidia-smi          # pick a free device"
echo "  run the pilot:     python -u src/run_e0_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 30"
echo
echo "  data/processed/ is committed -- no need to rebuild datasets on this box."
