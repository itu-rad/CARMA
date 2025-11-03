#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd

FOLDER_RE = re.compile(
    r"""
    ^
    (?P<time>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_     # timestamp
    (?P<script>[A-Za-z0-9_\-]+\.py)                    # script filename
    (?P<rest>.*)                                       # optional args suffix
    $
    """,
    re.X,
)

# Example suffix: _--batch_size_32_--epochs_1
ARG_KV_RE = re.compile(r"_--(?P<key>[A-Za-z0-9_\-]+)_(?P<val>[^_]+)")

# stdout.log regexes
RE_FWD_GB = re.compile(r"GB after forward:\s*([0-9]*\.?[0-9]+)")
RE_BWD_GB = re.compile(r"GB after backward:\s*([0-9]*\.?[0-9]+)")
RE_FAKE_OVERHEAD = re.compile(r"Time taken by faketensor:\s*([0-9]*\.?[0-9]+)")
RE_EXEC_TIME = re.compile(r"execution time:\s*([0-9]*\.?[0-9]+)")
RE_EXEC_TIME_ALT = re.compile(r"time:\s*([0-9]*\.?[0-9]+)")  # <--- added fallback
RE_LARGE_INT_LINE = re.compile(r"^\s*(\d{7,})\s*$")  # e.g., 1667554728

def parse_folder_name(name: str) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    """
    Returns (time_str, script, args_dict)
    """
    m = FOLDER_RE.match(name)
    if not m:
        return None, None, {}
    time_str = m.group("time")
    script = m.group("script")
    rest = m.group("rest") or ""
    args = {}
    for am in ARG_KV_RE.finditer(rest):
        k = am.group("key")
        v = am.group("val")
        args[k] = v
    return time_str, script, args

def parse_gpu_csv(gpu_csv_path: Path, target_uuid: str) -> Optional[float]:
    """
    Reads gpu.csv and returns the maximum memory.used [MiB] for the given UUID.
    Expected columns: uuid, memory.used [MiB], memory.total [MiB]
    Lines may include ' MiB' suffixes; we strip units defensively.
    """
    if not gpu_csv_path.exists():
        return None

    max_used_mib = None
    with gpu_csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Try to detect columns; fall back to positional if needed
        uuid_idx, used_idx = None, None
        if header:
            # normalize header spacing
            normalized = [h.strip().lower() for h in header]
            for i, h in enumerate(normalized):
                if "uuid" in h:
                    uuid_idx = i
                if "memory.used" in h or "memory used" in h:
                    used_idx = i
        for row in ([header] if header and uuid_idx is None else []):
            pass

        # If we didn't detect columns, assume 0=uuid, 1=used
        if uuid_idx is None or used_idx is None:
            uuid_idx, used_idx = 0, 1  # best effort

        # Iterate rows
        with gpu_csv_path.open("r", encoding="utf-8", errors="ignore") as f2:
            r2 = csv.reader(f2)
            _ = next(r2, None)  # skip header again
            for row in r2:
                if not row or len(row) <= max(uuid_idx, used_idx):
                    continue
                uuid = row[uuid_idx].strip()
                used_raw = row[used_idx].strip()
                # remove units like " MiB"
                used_num = re.sub(r"[^0-9.]", "", used_raw)
                try:
                    used_val = float(used_num)
                except ValueError:
                    continue
                if uuid == target_uuid:
                    max_used_mib = used_val if max_used_mib is None else max(max_used_mib, used_val)

    return max_used_mib

def parse_stdout(stdout_path: Path) -> Dict[str, Optional[float]]:
    """
    Extract:
      - GB after forward
      - GB after backward
      - FakeTensor Estimation (bytes -> GB)
      - FakeTensor overhead (s)
      - Execution time(s), Execution time(m), Execution time(h)
    """
    result = {
        "GB after forward": None,
        "GB after backward": None,
        "FakeTensor Estimation(GB)": None,
        "FakeTensor overhead": None,
        "Execution time(s)": None,
        "Execution time(m)": None,
        "Execution time(h)": None,
    }
    if not stdout_path.exists():
        return result

    text = stdout_path.read_text(encoding="utf-8", errors="ignore")

    def find_float(pattern: re.Pattern) -> Optional[float]:
        m = pattern.search(text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    result["GB after forward"] = find_float(RE_FWD_GB)
    result["GB after backward"] = find_float(RE_BWD_GB)
    result["FakeTensor overhead"] = find_float(RE_FAKE_OVERHEAD)

    # Execution time in seconds
    exec_s = find_float(RE_EXEC_TIME)
    if exec_s is None:
        exec_s = find_float(RE_EXEC_TIME_ALT)  # <--- try fallback if needed
    if exec_s is not None:
        result["Execution time(s)"] = exec_s
        result["Execution time(m)"] = exec_s / 60.0
        result["Execution time(h)"] = exec_s / 3600.0

    # FakeTensor Estimation: find a standalone large integer line (bytes)
    bytes_candidates: List[int] = []
    for line in text.splitlines():
        m = RE_LARGE_INT_LINE.match(line)
        if m:
            try:
                val = int(m.group(1))
                # Heuristic: treat as bytes if reasonably large (>= 1MB)
                if val >= 1_000_000:
                    bytes_candidates.append(val)
            except Exception:
                pass
    if bytes_candidates:
        # Prefer the last occurrence (typical logging order); fallback to max
        val_bytes = bytes_candidates[-1]
        result["FakeTensor Estimation(GB)"] = val_bytes / (1024 ** 3)

    return result

def main():
    p = argparse.ArgumentParser(description="Parse run folders and summarize metrics.")
    p.add_argument("--root", type=str, default="runs", help="Root folder containing run subfolders.")
    p.add_argument("--gpu-uuid", type=str, default="GPU-00f900e0-bb6f-792a-1b8a-597214c7e1a1",
                   help="Target GPU UUID to extract max memory.used [MiB] from gpu.csv.")
    p.add_argument("--out", type=str, default="ehsan-summary.csv", help="Output CSV path.")
    args = p.parse_args()

    root = Path(args.root)
    rows = []

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root folder not found or not a directory: {root}")

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        time_str, script, argmap = parse_folder_name(sub.name)
        if time_str is None:
            # Skip folders that don't match the naming scheme
            continue

        gpu_csv = sub / "gpu.csv"
        stdout_log = sub / "stdout.log"

        max_used_mib = parse_gpu_csv(gpu_csv, args.gpu_uuid)
        max_used_gb = (max_used_mib / 1024.0) if max_used_mib is not None else None  # <--- new GB column
        std = parse_stdout(stdout_log)
        
        fk_est = None
        if std["FakeTensor Estimation(GB)"]:
            fk_est = std["FakeTensor Estimation(GB)"] * 1024

        row = {
            "Run folder": sub.name,
            "Time": time_str,
            "Script": script,
            "Model": (script[:-3] if script else None),  # drop .py
            "batch_size": argmap.get("batch_size"),
            "epochs": argmap.get("epochs"),
            "Max memory.used [MB]": max_used_mib,
            "Faketensor Estimate [MB]": fk_est,
            "Max memory.used [GB]": max_used_gb,  # <--- added
            "GB after forward": std["GB after forward"],
            "GB after backward": std["GB after backward"],
            "FakeTensor Estimation(GB)": std["FakeTensor Estimation(GB)"],
            "FakeTensor overhead": std["FakeTensor overhead"],
            "Execution time(s)": std["Execution time(s)"],
            "Execution time(m)": std["Execution time(m)"],
            "Execution time(h)": std["Execution time(h)"],
        }
        rows.append(row)

    if not rows:
        print("No matching run folders found.")
        return

    df = pd.DataFrame(rows)
    # Order columns for readability
    col_order = [
        "Model",
        "Max memory.used [MB]", "Faketensor Estimate [MB]", "Max memory.used [GB]",  # <--- keep GB beside MiB
        "FakeTensor Estimation(GB)", "FakeTensor overhead",
        "Execution time(m)",
    ]
    # Keep any extra columns at the end
    df = df[[c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]]

    # Sort by Time then Script for convenience (Time is lexicographically sortable here)
    df.sort_values(by=["Time", "Script"], inplace=True, na_position="last")

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
