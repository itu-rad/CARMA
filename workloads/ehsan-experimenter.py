#!/usr/bin/env python3
"""
Run a list of training commands, capture stdout/stderr, and monitor GPUs via nvidia-smi.
Each task gets its own logs named from the script + model + batch size.

Usage:
  python run_and_monitor.py

Customize the COMMANDS list below with your training invocations.
"""

import os
import sys
import shlex
import time
import shutil
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Union

# ========= USER CONFIG =========
# Each entry can be either:
#   - a string command (will be shlex.split'd), OR
#   - a list of tokens (preferred, no shell needed)
COMMANDS: List[Union[str, List[str]]] = [
    # Examples — replace with your own
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "BERT_base_on_wiki.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "BERT_large_on_wiki.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet_cifar100.py", "--batch_size", "32", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet_cifar100.py", "--batch_size", "64", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet_cifar100.py", "--batch_size", "128", "--epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "efficientNet.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0,1", "python", "gpt2_large_on_wiki.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Inception.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Inception.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Inception.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet_cifar100.py", "--batch_size", "32", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet_cifar100.py", "--batch_size", "64", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet_cifar100.py", "--batch_size", "128", "--epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "mobilenet.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet18_cifar100.py", "--batch_size", "32", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet18_cifar100.py", "--batch_size", "64", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet18_cifar100.py", "--batch_size", "128", "--epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet34_cifar100.py", "--batch_size", "32", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet34_cifar100.py", "--batch_size", "64", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet34_cifar100.py", "--batch_size", "128", "--epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50_cifar100.py", "--batch_size", "32", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50_cifar100.py", "--batch_size", "64", "--epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50_cifar100.py", "--batch_size", "128", "--epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "resnet50.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "vgg.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "vgg.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "vgg.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Xception.py", "--batch_size", "32", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Xception.py", "--batch_size", "64", "--num_epochs", "1"],
    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "Xception.py", "--batch_size", "128", "--num_epochs", "1"],

    # ["env", "CUDA_VISIBLE_DEVICES=0,1", "python", "xlnet_base_cased.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0,1", "python", "xlnet_large_cased.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "UNet.py"],

    # ["env", "CUDA_VISIBLE_DEVICES=0", "python", "/home/ehyo/rad-scheduler/workloads/dlrm/dlrm_s_pytorch.py", "--data-set", "terabyte", "--raw-data-file", "/raid/datasets/criteo/data/day", "--loss-function", "bce", "--round-targets", "True", "--learning-rate", "0.05", "--mini-batch-size", "32768", "--nepochs", "1", "--num-batches", "10000", "--arch-sparse-feature-size", "64", "--arch-mlp-bot", "13-512-256-128-64", "--arch-mlp-top", "1024-512-256-1", "--use-gpu", "--memory-map", "--dataset-multiprocessing", "--print-time", "--print-freq", "100", "--num-workers", "16"],

    ["env", "CUDA_VISIBLE_DEVICES=0", "python", "/home/ehyo/rad-scheduler/workloads/vision/references/detection/train.py", "--dataset", "coco", "--data-path", "/raid/datasets/coco", "--model", "maskrcnn_resnet50_fpn", "--epochs", "1", "-b", "8", "--output-dir", "runs"]
]

# How often to sample nvidia-smi (seconds). 1.0 is a good default.
SMI_INTERVAL_SEC = 1.0

# Directory to store logs. Will be created if not present.
RUNS_DIR = Path("runs")

# Optional: set environment variables common to all runs here
COMMON_ENV = os.environ.copy()
# COMMON_ENV["CUDA_VISIBLE_DEVICES"] = "0"  # example: pin to GPU 0
# ========= END USER CONFIG =====


def ensure_nvidia_smi() -> str:
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("ERROR: nvidia-smi not found in PATH. Please install NVIDIA drivers / CUDA toolkit.", file=sys.stderr)
        sys.exit(1)
    return smi

# === DCGM additions: ensure dcgmi present ===
def ensure_dcgmi() -> str:
    dcgmi = shutil.which("dcgmi")
    if not dcgmi:
        print("ERROR: dcgmi not found in PATH. Please install NVIDIA DCGM.", file=sys.stderr)
        sys.exit(1)
    return dcgmi
# === end DCGM additions ===


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def normalize_cmd(cmd: Union[str, List[str]]) -> List[str]:
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return list(cmd)


def find_arg_value(args: List[str], keys: List[str]) -> str:
    """Return the value for any of the provided keys (e.g., ['--batch-size','-b']) if present; else ''."""
    for i, tok in enumerate(args):
        if tok in keys:
            # value may be next token
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                return args[i + 1]
        else:
            # Handle --key=value form
            for k in keys:
                if tok.startswith(k + "="):
                    return tok.split("=", 1)[1]
    return ""

from typing import List

def command_tag(args: List[str]) -> str:
    """
    Build the run folder tag from tokens that come *after* the 'python' token.
    If 'python' isn't found, fall back to the whole arg list.
    """
    py_idx = None
    for i, tok in enumerate(args):
        if tok == "python" or tok.endswith("/python") or tok.endswith("\\python.exe"):
            py_idx = i  # keep last match in case of 'env ... python ...'
    tail = args[py_idx + 1:] if (py_idx is not None and py_idx + 1 < len(args)) else args[:]
    raw = "_".join(tail) if tail else "_".join(args)
    return safe_filename(raw)[:200]  # trim to avoid overly long filenames


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s)


def monitor_gpu(stop_event: threading.Event, log_path: Path, smi_path: str, interval: float):
    """
    Periodically run:
      nvidia-smi --query-gpu=uuid,memory.used,memory.total --format=csv
    Append timestamped rows to CSV.
    """
    # Use csv,noheader,nounits for easier parsing later; include header ourselves
    query = ["uuid", "memory.used", "memory.total"]
    header = "timestamp," + ",".join(query) + "\n"
    cmd = [smi_path,
           "--query-gpu=" + ",".join(query),
           "--format=csv,noheader,nounits"]

    # Ensure parent exists
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as f:
        f.write(header)
        while not stop_event.is_set():
            ts = datetime.now().isoformat()
            try:
                out = subprocess.check_output(cmd, env=COMMON_ENV, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
            except subprocess.CalledProcessError as e:
                out = e.output.decode("utf-8", errors="replace")
            # nvidia-smi returns one line per GPU; we stamp each line with the same timestamp
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                f.write(f"{ts},{line.strip()}\n")
            # sleep last to allow quick exit when stop_event set after process ends
            stop_event.wait(interval)

# === DCGM additions: monitor dcgmi dmon output repeatedly ===
def monitor_dcgm(stop_event: threading.Event, log_path: Path, dcgmi_path: str, interval: float):
    """
    Periodically run (exactly as requested):
      dcgmi dmon -e 203,1001,1002,1003,1006,1007,1008,1004,204,1005,1009,1010,1011,1012,155,156 -c 1
    and append the raw output to 'dcgm.csv' with an ISO timestamp.
    Note: This writes raw dcgmi text blocks (not parsed CSV) for fidelity with dcgmi output.
    """
    cmd = [
        dcgmi_path, "dmon",
        "-e", "203,1001,1002,1003,1006,1007,1008,1004,204,1005,1009,1010,1011,1012,155,156",
        "-c", "1",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Start fresh file with a small preamble for context
    with open(log_path, "w", buffering=1) as f:
        f.write("# dcgmi dmon raw output (one-sample blocks), appended every ~{:.3f}s\n".format(interval))
        f.write("# command: {}\n".format(" ".join(cmd)))
        while not stop_event.is_set():
            ts = datetime.now().isoformat()
            f.write(f"\n# timestamp: {ts}\n")
            try:
                out = subprocess.check_output(cmd, env=COMMON_ENV, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
            except subprocess.CalledProcessError as e:
                out = e.output.decode("utf-8", errors="replace")
            f.write(out.rstrip() + "\n")
            stop_event.wait(interval)
# === end DCGM additions ===


def run_one(cmd_tokens: List[str], runs_dir: Path, smi_path: str, dcgmi_path: str) -> int:  # << changed signature (add dcgmi_path)
    tag = command_tag(cmd_tokens)
    tag = safe_filename(tag)
    tstamp = now_str()
    run_dir = runs_dir / f"{tstamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log file paths
    cmd_txt = run_dir / "command.txt"
    out_log = run_dir / "stdout.log"
    err_log = run_dir / "stderr.log"
    gpu_csv = run_dir / "gpu.csv"
    dcgm_csv = run_dir / "dcgm.csv"  # === DCGM additions ===
    exit_txt = run_dir / "exitcode.txt"

    # Persist the exact command
    with open(cmd_txt, "w") as f:
        f.write(" ".join(shlex.quote(t) for t in cmd_tokens) + "\n")

    print(f"\n=== Starting task: {tag}")
    print(f"    Logs: {run_dir}")

    # Start process
    with open(out_log, "w", buffering=1) as out_f, open(err_log, "w", buffering=1) as err_f:
        proc = subprocess.Popen(
            cmd_tokens,
            stdout=out_f,
            stderr=err_f,
            env=COMMON_ENV,
        )

        # Start GPU monitor thread (nvidia-smi)
        stop_evt = threading.Event()
        mon_smi = threading.Thread(target=monitor_gpu, args=(stop_evt, gpu_csv, smi_path, SMI_INTERVAL_SEC), daemon=True)
        mon_smi.start()

        # === DCGM additions: start dcgmi monitor thread ===
        mon_dcgm = threading.Thread(target=monitor_dcgm, args=(stop_evt, dcgm_csv, dcgmi_path, SMI_INTERVAL_SEC), daemon=True)
        mon_dcgm.start()
        # === end DCGM additions ===

        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: terminating training process...")
            proc.terminate()
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing process...")
                proc.kill()
                rc = proc.wait()
        finally:
            # Stop monitoring
            stop_evt.set()
            mon_smi.join(timeout=5)
            # === DCGM additions ===
            mon_dcgm.join(timeout=5)
            # === end DCGM additions ===

    with open(exit_txt, "w") as f:
        f.write(str(rc) + "\n")

    status = "SUCCESS" if rc == 0 else f"FAIL (exit {rc})"
    print(f"=== Finished: {tag} -> {status}")
    return rc


def main():
    smi_path = ensure_nvidia_smi()
    dcgmi_path = ensure_dcgmi()  # === DCGM additions ===
    RUNS_DIR.mkdir(exist_ok=True, parents=True)

    # Normalize commands
    all_cmds = [normalize_cmd(c) for c in COMMANDS]

    failures = 0
    for cmd in all_cmds:
        rc = run_one(cmd, RUNS_DIR, smi_path, dcgmi_path)  # === DCGM additions ===
        if rc != 0:
            failures += 1

    print("\n========== SUMMARY ==========")
    print(f"Total tasks: {len(all_cmds)} | Failures: {failures}")
    print(f"Logs directory: {RUNS_DIR.resolve()}")


if __name__ == "__main__":
    main()
