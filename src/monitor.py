import time
import datetime
import pandas as pd
from subprocess import Popen, PIPE
import numpy as np
from threading import Lock

import subprocess
import io
import csv

analyze_configuration = "risk" # can be Normal | risk

# ===== Helper functions for risk concept =======

def _ema_last(s, alpha):  # pd.Series -> float
    return float(s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])

def _p95(s):  return float(pd.Series(s, dtype=float).quantile(0.95))
def _cv(s):
    x = pd.Series(s, dtype=float); m = x.mean()
    return float(x.std(ddof=0)/m) if m and abs(m) > 1e-12 else float("nan")

def _lin_slope(s):
    y = pd.Series(s, dtype=float).to_numpy()
    t = np.arange(len(y), dtype=float); tc = t - t.mean()
    denom = (tc**2).sum()
    if denom == 0.0: return 0.0
    return float(((tc * (y - y.mean())).sum()) / denom)

def _trend_flag(s, thresh=0.003, up_only=True):
    m = _lin_slope(s)
    return int(m > thresh) if up_only else int(abs(m) > thresh)

# weights + trend threshold (tweak as you like)
RISK_W = dict(wT=0.5, wE=0.3, wB=0.1, wC=0.1)  # T=p95, E=EMA, B=CV, C=trend_flag
TREND_THRESH = 0.003
TREND_UP_ONLY = True

# ========= end of window risk helpers ============


# monitoring Window of data
# making sure that race condition does not happen
G_LOCK = Lock()
GV_LOCK = Lock()
Gmetrics_are_valid = False
Gmetrics = pd.DataFrame(columns=["gpu_uuid", "free_gpu_memory", "gpu_utilization", "gract", "smact",
                                                     "smocc", "fp64", "fp32", "fp16", "tc",
                                                     "memory_copy", "dram_active", "pcie_tx_bytes",
                                                     "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes",
                                                     "power", "energy", "gpu_id"])
# executes a shell bash command and return the output
def execute_command(cmd, shell=False, vars={}):
    """
    output: string format terminal based output of a bash shell command
    
    Executes a bash command and return the result
    """
    cmd = cmd.split()
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=shell) as p:
        o=p.stdout.read()
    return o

# returns detected GPUs on the system excluding the one 
# that is specifically for dispaly purposes
def gpu_uuids():
    """
    output: dictionary of gpu_index:gpu_uuid
    
    Detects GPUs on the system using 'nvidia-smi'
    """
    gpus = dict()
    o = execute_command("nvidia-smi -L")
    o = o.split('\n')
    for line in o:
        # Ignoring the display dedicated GPU on the DGX A100 station machine
        if "Display" in line:
            continue
        if "UUID: GPU" in line:
            gpu_uuid = line.split("UUID:")[1].split(")")[0].strip()
            gpus[gpu_uuid] = line.split("GPU")[1].split(":")[0].strip()

    return gpus

# This function's responsibility is to show the list of available GPUs
# Made sure that the MPS daemon process is not considered as being active
# This function is used for implementing exclusive assignment of GPUs to tasks 
# def gpus_activeness():
#     """
#     Analyzing 'nvidia-smi pmon -c 1' command and figuring GPUs status
#     """
#     gpus = gpu_uuids()

#     # reversing keys and values in gpu_uuids dictionary
#     # to be able to find the corresponding uuid
#     tmp_gpus = {value: key for key, value in gpus.items()}    
#     activity_dict = dict.fromkeys(tmp_gpus, 0)

#     # active GPUs
#     active_GPUs = dict()
#     out = execute_command("nvidia-smi pmon -c 1")
#     out = out.split("\n")

#     for line in out:
#         if line.startswith("#"):
#             pass
#         elif len(line) != 0:
#             tmp_list = line.strip().split()
#             if tmp_list[1] != '-' and tmp_list[7] != 'nvidia-cuda-mps':
#                 print(tmp_list[7])
#                 if tmp_list[0] in active_GPUs:
#                     active_GPUs[tmp_list[0]].append(tmp_list[1])
#                 else:
#                     active_GPUs[tmp_list[0]] = [tmp_list[1]]
    
#     for active_gpu_index in active_GPUs:
#         activity_dict[active_gpu_index] = 1

#     out_dict = dict()

#     for index in tmp_gpus:
#         tmp = tmp_gpus[index]
#         out_dict[tmp] = activity_dict[index]

#     return out_dict

def gpus_activeness():
    """
    Parse `nvidia-smi pmon -c 1` and return {gpu_uuid: 0|1}, ignoring MPS daemons.
    0 = idle, 1 = has non-MPS process.
    """
    gpus = gpu_uuids()                           # {uuid: "0"/"1"/...}
    id_by_uuid = {u: str(i) for u, i in gpus.items()}
    active_gpu_indices = set()

    out = execute_command("nvidia-smi pmon -c 1")
    if not out:
        # fail-safe: assume all idle
        return {u: 0 for u in gpus.keys()}

    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        gpu_idx, pid = toks[0], toks[1]
        cmd = toks[-1].rstrip(",")               # last token is command (strip trailing comma)

        # mark active only for real PIDs and non-MPS commands
        if pid != "-" and cmd not in {"nvidia-cuda-mps", "nvidia-cuda-mps-control"}:
            active_gpu_indices.add(gpu_idx)

    # build uuid -> 0/1 map
    activity = {}
    for uuid, idx in id_by_uuid.items():
        activity[uuid] = 1 if idx in active_gpu_indices else 0
    return activity


def pmon_rows():
    """Yield dicts per PMON row: {'idx': '0','pid': '2346212','type':'M+C','cmd':'python'}"""
    out = execute_command("nvidia-smi pmon -c 1")
    if not out:
        return []
    rows = []

    IGNORE = {"nvidia-cuda-mps", "nvidia-cuda-mps-control"}  # <- skip both daemons

    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue

        cmd = toks[-1].rstrip(",")
        if cmd in IGNORE:             # <- ignore MPS rows
            continue

        rows.append({
            "idx": toks[0],
            "pid": toks[1],
            "type": toks[2],
            "cmd": toks[-1].rstrip(","),   # strip trailing comma if exists
        })
    return rows

# This function checks whether a key <PID> is seen in the output
def is_in_pmon(pid: str) -> bool:
    """True if *any* PMON row has this PID (exact match)."""
    for r in pmon_rows():
        if r["pid"] == pid:
            return True
    return False



# this function is responsible for calculating EWMA
def _ema_last(series, alpha=0.2):
    """Return last value of EMA for a pandas Series."""
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def gpu_mem_usage():
    # uuid, used, total, free (MiB)
    o = execute_command(
        "nvidia-smi --query-gpu=uuid,memory.used,memory.total,memory.free --format=csv,noheader,nounits"
    )
    result = {}
    for uuid, used, total, free in csv.reader(io.StringIO(o)):
        result[uuid.strip()] = int(free)
    return result




# This function gets the metrics from dcgmi tool
def dcgmi_monitor():
    """
    output: dictionary of gpu_uuids:smact
    Uses dcgmi to monitor available GPUs and GPU MIG instances
    """

    result = dict()
    # ====== compute resources ========
    # 203 gpu utilization [0]
    # 1001 gract [1], 1002 smact [2], 1003 smocc [3], 1006 fp64 [4], 1007 fp32 [5], 1008 fp16 [6], 1004 tensor core [7]
    
    # ======== memory =========
    # 204 memory copy [8], 1005 drama [9]
    
    # ======= data transfer ============
    # 1009 pcie_tx_bytes [10], 1010 pcie_rx_bytes [11]
    # 1011 nvlink_tx_bytes [12], 1012 nvlink_rx_bytes [13]
    
    # ====== power and energy =======
    # 155 power [14], 156 energy [15]
    
    result = dict()

    o = execute_command("dcgmi dmon -e 203,1001,1002,1003,1006,1007,1008,1004,204,1005,1009,1010,1011,1012,155,156 -c 1")
    o = o.split("\n")

    for line in o:
        if line.startswith("#") or line.startswith("ID") or line == "":
            continue
        gpu_id = line.split("GPU")[1].split()[0].strip()
        metrics = line.split("GPU")[1].split()[1:]
        assert(len(metrics) == 16)
        result[gpu_id] = metrics 

    return result





# This function is used for no-stopping monitoring of GPUs and logging them
# also the latest monitored data gets updated here, which
# is used for making collocation decisions
header_flag = True
# developed and tested
def monitor_logger(window = 30):
    """
    - monitors detected GPUs with <timestep> seconds intervals for <window> number of minutes

    - uses 'gpu_mem_usage()' and 'dcgmi_monitor()' functions
    """
    INTERVAL = 1
    next_tick = time.monotonic()
    
    cols = ["gpu_uuid", "free_gpu_memory", "gpu_utilization", "gract", "smact",
        "smocc", "fp64", "fp32", "fp16", "tc",
        "memory_copy", "drama", "pcie_tx_bytes",
        "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes",
        "power", "energy", "gpu_id"]
    
    window_monitored_metrics = pd.DataFrame(columns=cols)

    while(True):
        # gather
        # === gather ===
        gpus = gpu_uuids()             # dict: uuid -> id
        free_mem = gpu_mem_usage()           # dict: uuid -> [mem_used, mem_cap, util]
        dcgm = dcgmi_monitor()         # dict: gpu_id -> [gract, smact, ... energy]
        num_gpus = len(gpus)
        max_rows = window * max(1, num_gpus)

        temp = pd.DataFrame(columns=cols)
        for gpu_uuid, gpu_id in gpus.items():
            free_val = int(free_mem.get(gpu_uuid, 0))     # <- value, not dict
            dcgm_metrics = dcgm[gpu_id]                   # length 16 (203/1001/…/156)
            row = [gpu_uuid, free_val] + dcgm_metrics + [gpu_id]
            window_monitored_metrics.loc[len(window_monitored_metrics)] = row
            temp.loc[len(temp)] = row

        # === trim to rolling window ===
        if len(window_monitored_metrics) > max_rows:
            window_monitored_metrics = window_monitored_metrics.tail(max_rows).reset_index(drop=True)
            
        # ---- atomic publish of a snapshot ----
        with G_LOCK:
                globals()["Gmetrics"] = window_monitored_metrics.copy(deep=False)
        # --- mark as valid once the rolling window is filled (do this once) ---
        if len(window_monitored_metrics) >= max_rows:
            with GV_LOCK:
                if not globals().get("Gmetrics_are_valid", False):
                    globals()["Gmetrics_are_valid"] = True

        # === csv logging (with timestamp) ===
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        temp1 = temp.assign(time=[now] * len(temp))
        global header_flag
        if header_flag:
            temp1.to_csv('dcgmi_metrics.csv', mode='a', index=False)
            header_flag = False
        else:
            temp1.to_csv('dcgmi_metrics.csv', mode='a', header=False, index=False)

        # === 1s tick ===
        next_tick += INTERVAL
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()





# This function is responsible for giving decision-making data to the collocation logic
def analyze_Gmetrics(data=None):
    """
    Summarize rolling-window Gmetrics per gpu_uuid:
      active, last mem used, total cap, available, mean smact/smocc/drama.
    Returns a DataFrame indexed by gpu_uuid.
    """
    while True:
        with GV_LOCK:
            ready = globals().get("Gmetrics_are_valid", False)
        if ready:
            break
        time.sleep(0.1)  # waits only until window fills (once at startup)

    if data is None:
        with G_LOCK:
            data = globals()["Gmetrics"]

    if data is None or data.empty:
        return pd.DataFrame(columns=[
            "free_gpu_memory",
            "smact","smocc","drama"
        ])

    df = data.copy()
    # ensure numeric dtypes
    for c in ["free_gpu_memory","smact","smocc","drama"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    gpu_ids = gpu_uuids()              # dict: uuid -> id
    gpus_activity = gpus_activeness()  # dict: uuid -> bool/int

    grouped = df.groupby("gpu_uuid", sort=False)
    analyzed = {}

    for uuid in gpu_ids:
        if uuid not in grouped.groups:
            analyzed[uuid] = [gpus_activity.get(uuid, 0), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            continue

        g = grouped.get_group(uuid)
        free_mem = max(0, int(g["free_gpu_memory"].iloc[-1]) - 512)

        # This is for setting the alpha value according to the size of the monitoring window

        W = len(g)
        alpha = 2.0/(W+1.0) if W > 0 else 0.5  # window-aware EMA

        if globals()["analyze_configuration"] == "Normal":
            smact = g["smact"].mean()
            smocc = g["smocc"].mean()
            drama = g["drama"].mean()
        elif globals()["analyze_configuration"] == "EWMA":
            alpha = 0.5  # tweakable; ~effective window ≈ 2/(alpha)-1 samples
            smact = _ema_last(g["smact"], alpha=alpha)
            smocc = _ema_last(g["smocc"], alpha=alpha)
            drama = _ema_last(g["drama"], alpha=alpha)
        else:
            # --- Linear Risk (mean, median, p95, p50, EMA) ---
            # alpha already computed above from window size W

            RISK_V2_W = {
                "w_mean":   0.20,
                "w_median": 0.20,
                "w_p95":    0.30,
                "w_p50":    0.10,
                "w_ema":    0.20,
            }

            def _risk(series):
                s = pd.to_numeric(series, errors="coerce").dropna()
                if s.empty:
                    return float("nan")
                mean_v   = float(s.mean())
                median_v = float(s.median())
                p95_v    = float(s.quantile(0.95))
                p50_v    = float(s.quantile(0.50))
                ema_v    = float(_ema_last(s, alpha))
                return (RISK_V2_W["w_mean"]   * mean_v   +
                        RISK_V2_W["w_median"] * median_v +
                        RISK_V2_W["w_p95"]    * p95_v    +
                        RISK_V2_W["w_p50"]    * p50_v    +
                        RISK_V2_W["w_ema"]    * ema_v)

            smact = _risk(g["smact"])
            smocc = _risk(g["smocc"])
            drama = _risk(g["drama"])

        analyzed[uuid] = [
            free_mem,
            smact,
            smocc,
            drama,
        ]

    out = pd.DataFrame.from_dict(
        analyzed,
        orient="index",
        columns=[
            "GPU_mem_available",
            "smact","smocc","drama"
        ],
    )

    print(out)
    
    return out



# This function is responsible for monitoing system CPU and Memory usage
def top_extractor(process_names = ["python"]):
    # they will keep the accumulative %CPU and %MEM for all wanted processes
    CPU_util = 0
    Mem_util = 0
    all_memory = 0
    free_available_system_memory = 0
    memory_used_total = 0

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    top = subprocess.Popen("top -i -b -n 1".split(), stdout=subprocess.PIPE)
    # print(top)

    for line in io.TextIOWrapper(top.stdout, encoding="utf-8"):
        line = line.lstrip()

        if line.startswith("top") or line.startswith("Tasks") or line.startswith("%") or line.startswith("PID") or line.startswith(" "):
                pass
        else:
            word_vector = line.strip().split()
            if (line.startswith("KiB") or line.startswith("MiB")) and len(word_vector) != 0:
                    if word_vector[1] == "Mem":
                        
                        if word_vector[4] == "total,":
                            all_memory = float(word_vector[3]) / 1000

                        if word_vector[6] == "free,":
                            free_available_system_memory = float(word_vector[5]) / 1000

                        if word_vector[8] == "used,":
                            memory_used_total = float(word_vector[7]) / 1000

            elif len(word_vector) != 0:
                    if word_vector[11] in process_names:
                        CPU_util += float(word_vector[8])
                        Mem_util += float(word_vector[9])

    to_return = pd.DataFrame(columns=["total_memory", "free_memory", "used_memory", 
                                                     "%CPU", "%MEM", "time"])
    data_to_return = [all_memory, free_available_system_memory, memory_used_total, CPU_util, Mem_util, now]

    to_return.loc[len(to_return)] = data_to_return
    return to_return


header_flag_top = True
def top_system_logger():
    while True:
        # time.sleep(3)
        a = top_extractor()

        global header_flag_top
        if header_flag_top == True:
            a.to_csv('top_data.csv', mode='a', index = False)
            header_flag_top = False
        else:
            a.to_csv('top_data.csv', mode='a', header = False, index=False)


def pid_on_system(pid: str) -> bool:
    result = execute_command("ps -e")
    return pid in result




# from threading import Thread
# Thread(target = monitor_logger).start()


# while True:
#     print(analyze_Gmetrics())
#     time.sleep(5)