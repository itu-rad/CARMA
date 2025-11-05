#!/usr/bin/env bash
# disable_mps_all_gpus_minimal.sh
set -euo pipefail

command -v nvidia-cuda-mps-control >/dev/null || { echo "nvidia-cuda-mps-control not found"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }

# 1) Stop MPS control daemon (and servers)
echo quit | sudo nvidia-cuda-mps-control

# 2) Reset compute mode to DEFAULT on all GPUs (optional but typical after MPS)
GPU_N=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((GPU_N-1))); do
  sudo nvidia-smi -i "$i" -c DEFAULT
done

# 3) (Optional) turn off persistence mode
# sudo nvidia-smi -pm 0

# Status
nvidia-smi --query-gpu=index,name,compute_mode,persistence_mode --format=csv
echo "MPS stopped."
