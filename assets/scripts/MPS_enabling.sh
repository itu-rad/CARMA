#!/usr/bin/env bash
# enable_mps_all_gpus_minimal.sh
set -euo pipefail

require() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1"; exit 1; }; }
require nvidia-smi
require nvidia-cuda-mps-control

# 1) (Recommended) persistence mode for fast client startup
#    You can also use the nvidia-persistenced service on your distro.
sudo nvidia-smi -pm 1 >/dev/null

# 2) Set all GPUs to EXCLUSIVE_PROCESS (recommended with MPS)
GPU_N=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((GPU_N-1))); do
  sudo nvidia-smi -i "$i" -c EXCLUSIVE_PROCESS >/dev/null
done

# 3) Ensure default pipe/log dirs exist and perms are sane
#    (MPS defaults: pipe=/tmp/nvidia-mps, logs=/var/log/nvidia-mps)
sudo mkdir -p /tmp/nvidia-mps /var/log/nvidia-mps
sudo chmod 1777 /tmp/nvidia-mps
sudo chmod 755 /var/log/nvidia-mps

# 4) If a control daemon exists, shut it down cleanly
if pgrep -x nvidia-cuda-mps-control >/dev/null; then
  echo quit | nvidia-cuda-mps-control || true
  sleep 1
fi

# 5) Start one control daemon for the node (all GPUs visible)
#    Do NOT set CUDA_VISIBLE_DEVICES here if you want all GPUs usable.
sudo nvidia-cuda-mps-control -d

echo "MPS enabled on all GPUs. To stop: echo quit | nvidia-cuda-mps-control"
echo "Pipe dir: /tmp/nvidia-mps  |  Logs: /var/log/nvidia-mps"
nvidia-smi --query-gpu=index,name,compute_mode,persistence_mode --format=csv