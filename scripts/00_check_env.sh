#!/usr/bin/env bash
set -u

echo "== DriveMind-VL Environment Check =="
echo "Current path: $(pwd)"
echo

echo "== Python =="
if command -v python >/dev/null 2>&1; then
  python --version
else
  echo "python: not found"
fi

if command -v pip >/dev/null 2>&1; then
  pip --version
else
  echo "pip: not found"
fi
echo

echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  nvidia-smi || true
else
  echo "nvidia-smi: not found"
fi
echo

echo "== PyTorch =="
python - <<'PY'
try:
    import torch
    print("torch installed: yes")
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"gpu[{idx}] name: {props.name}")
            print(f"gpu[{idx}] memory_gb: {props.total_memory / 1024**3:.2f}")
except Exception as exc:
    print("torch installed: no")
    print("hint: run bash scripts/01_install_env.sh")
    print("error:", exc)
PY
echo

echo "== Disk =="
df -h . || true
echo

echo "== Memory =="
if command -v free >/dev/null 2>&1; then
  free -h
else
  python - <<'PY'
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"total_gb: {mem.total / 1024**3:.2f}")
    print(f"available_gb: {mem.available / 1024**3:.2f}")
except Exception:
    print("memory info unavailable without free/psutil")
PY
fi

