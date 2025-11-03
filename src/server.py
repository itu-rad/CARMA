import socket
import time
import datetime
from threading import Thread, Lock
import subprocess
import os
import logging

from typing import Set



import monitor
import rad_parser
from task_queue import Task, Tasks
from itertools import cycle, islice
from load_yaml import load_yaml

# for getting the launched task PID
def launch_and_get_pid(cmd: str) -> int | None:
    p = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,      # receives the echoed PID
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,   # <<< unique SID per launch
    )
    pid_line = p.stdout.readline().strip() if p.stdout else ""
    if p.stdout:
        p.stdout.close()            # don't wait; job keeps running
    try:
        return int(pid_line)
    except ValueError:
        return None
# ending the logic for getting PID

# --- add helpers (just above descendants) ---
def _proc_children_once(pid: int) -> list[int]:
    PROC_CHILDREN_HELPER = "/usr/bin/proc_children_helper"
    """Direct children via /proc (more reliable than pgrep)."""
    try:
        out = subprocess.check_output([PROC_CHILDREN_HELPER, str(pid)], text=True)
        # print("children for the pid ", pid, ": ", [int(x) for x in out.split() if x.isdigit()])
        return [int(x) for x in out.split() if x.isdigit()]

    except Exception:
        print("could not read children")
        return []

def _session_id(pid: int) -> str | None:
    """Return POSIX session id (SID) of a pid, or None."""
    try:
        out = subprocess.check_output(["ps", "-o", "sid=", "-p", str(pid)], text=True).strip()

        # print("found the session id for pid ", pid, " is: ", out)
        return out if out else None
    except subprocess.CalledProcessError:
        return None

def _pids_in_same_session(launcher_pid: int) -> Set[int]:
    """All PIDs that share the same SID as launcher (helps with MPS, fork/exec)."""
    sid = _session_id(launcher_pid)
    if not sid:
        return set()
    try:
        # List all pids with their sid, filter by sid
        out = subprocess.check_output(["ps", "-e", "-o", "pid=,sid="], text=True)
    except subprocess.CalledProcessError:
        return set()
    s = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == sid and parts[0].isdigit():
            s.add(int(parts[0]))
    
    # print("pids in the same session got and aggregated: ", s)
    return s

# --- replace descendants() with a proc-based, recursive walk + SID union ---
def descendants(pid: int) -> set[int]:
    """All descendants of pid (via /proc), plus same-session PIDs for MPS cases."""
    seen: Set[int] = set()
    frontier = [pid]
    while frontier:
        p = frontier.pop()
        if p in seen:
            continue
        seen.add(p)
        # use /proc children (robust across shells/conda)
        kids = _proc_children_once(p)
        frontier.extend(kids)
    # union same-session peers (captures MPS client that exec'd out of the tree)
    return seen | _pids_in_same_session(pid)


# --- pid resolve logic: also accept same-session matches ---
def resolve_gpu_pid(launcher_pid: int, timeout=30, poll=0.5) -> int:
    # deadline for checking for the appearance
    deadline = time.time() + timeout
    sid_l = _session_id(launcher_pid)  # compute once
    last_seen = None

    while time.time() < deadline:
        # if the launcher died, fall back
        if not monitor.pid_on_system(str(launcher_pid)):
            return last_seen or launcher_pid
        
        # children can change over time — refresh every iteration
        cand = descendants(launcher_pid)
        if sid_l:
            cand |= _pids_in_same_session(launcher_pid)  # belt & suspenders

        found = None
        # getting processes on all GPUs
        for row in monitor.pmon_rows():

            cmd = row.get("cmd", "")

            # ignoring the row when it has mps daemon
            if cmd in {"nvidia-cuda-mps", "nvidia-cuda-mps-control"}:
                continue
            
            pid_s = row.get("pid", "")
            if pid_s.isdigit():
                gpu_pid = int(pid_s)
                if gpu_pid in cand:
                    found = gpu_pid
                    break

        if found is not None:
            # (optional) return only if seen twice in a row to avoid transient matches
            if last_seen == found:
                return found
            last_seen = found
                
        time.sleep(poll)

    
    return last_seen or launcher_pid


def _async_resolve_and_update(launcher_pid: int, gpu_uuids: list[str]) -> None:
    try:
        real_pid = resolve_gpu_pid(launcher_pid, timeout=1000, poll=0.5)
        print("resolved the PID, and will update the table")
        for u in gpu_uuids:
            gpus_state.at[u, "CPU_task_PID"] = int(real_pid)
            # kick off the grace window immediately if PMON already sees it
            print("updated the validity table!", gpus_state)

            # updating the table if even we see it here, however, update will take care of it :)
            if monitor.is_in_pmon(str(real_pid)):
                print("oh! wait, I saw it here right after resolving!")
                mark_seen_now(u)

    except Exception as e:
        logging.exception("async resolve failed for %s: %s", launcher_pid, e)


# logger for keeping track of submission, dispatch, and termination time
logging.basicConfig(filename='std.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 

# loading mapping policy
cfg = load_yaml()

policy = cfg.get("mapper", {}).get("policy", "exclusive")
print("Configured mapping policy:", policy)
# end of reading the mapper's policy

estimator = cfg.get("mapper", {}).get("estimator", "None")
print("Configured mapping estimator:", estimator)
# end of reading the GPU memory estimator

recovery_dir = cfg.get("recovery", {}).get("dir", "/home/ehyo/rad-scheduler")
print("Configured recovery directory:", recovery_dir)
# end of reading the recovery mechanism to look directory

# locks for avoiding race condition
lock = Lock()
recover_lock = Lock()

# queues for submitted jobs, and
main_queue = Tasks()
recovery_queue = Tasks()



# ============= for having Round-Robin selection logic of GPUs ============
gpu_UUIDs = monitor.gpu_uuids()

GPU_IDs = []
for gpu in gpu_UUIDs:
    GPU_IDs.append(gpu)

# round-robin generator
round_robin_generator = cycle(GPU_IDs)

def select_ids(n):
    """
    Selects n IDs in a round-robin manner.
    Args:
        n (int): Number of IDs to select.
    Returns:
        list: List of selected IDs.
    """
    return list(islice(round_robin_generator, n))
# ============= End of Round Robin selection logic of GPUs =================

# keeps track of the handles crashes
# The inital logic is to recover a (OOM) crashed task once
# since the next step is handling it with relaunching
# with an exclusive access to a GPU/ GPUs
handled_crashes = []


# data strcture to keep track of
# what tasks are assigned to which tasks
# if they are moved from CPU --> GPU

patience = cfg.get("monitor", {}).get("patience", "10")
monitoring_window_size = cfg.get("monitor", {}).get("window", "30")

# finished reading the monitoring patience

import pandas as pd
gpus_state = pd.DataFrame(
    {
        "CPU_task_PID": pd.Series(dtype="str"),   # last PID sent to this GPU (nullable int)
        "validity":     pd.Series(dtype="boolean"), # ready for next decision
        "gpu_seen_at":  pd.Series(dtype="float64"), # time.monotonic() when first seen on GPU
    },
)

# defining the state of the GPU when a task gets launched for it
def init_gpu_state(uuid_to_id: dict[str, str]) -> pd.DataFrame:
    idx = pd.Index(list(uuid_to_id.keys()), name="GPU_uuid", dtype="string")
    df = pd.DataFrame(index=idx)
    # mapping column (index-aware)
    df["GPU_id"]       = pd.Series([str(uuid_to_id[u]) for u in idx], index=idx, dtype="string")
    # state columns (index-aware, with desired dtypes)
    df["CPU_task_PID"] = pd.Series(pd.NA,   index=idx, dtype="Int64")
    df["validity"]     = pd.Series(True,   index=idx, dtype="boolean")
    df["gpu_seen_at"]  = pd.Series(pd.NA,   index=idx, dtype="Float64")
    return df

#  ====== initialized GPUs ======
gpus_state = init_gpu_state(gpu_UUIDs)


def launch_task(gpu_uuid: str, pid: int) -> None:
    # assumes fixed set of GPUs initialized; raise if unknown
    if gpu_uuid not in gpus_state.index:
        raise KeyError(f"Unknown GPU UUID: {gpu_uuid}")
    gpus_state.at[gpu_uuid, "CPU_task_PID"] = int(pid)
    gpus_state.at[gpu_uuid, "validity"] = False
    gpus_state.at[gpu_uuid, "gpu_seen_at"] = pd.NA


def mark_seen_now(gpu_uuid: str, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    if pd.isna(gpus_state.at[gpu_uuid, "gpu_seen_at"]):
        gpus_state.at[gpu_uuid, "gpu_seen_at"] = float(now)


def update():
    now = time.monotonic()
    for gpu_uuid in gpus_state.index:
        # print(gpu_uuid, str(gpus_state.loc[gpu_uuid, "CPU_task_PID"]))
        # print(time.monotonic() - gpus_state.loc[gpu_uuid, "gpu_seen_at"], gpus_state.loc[gpu_uuid, "validity"])
        # print(gpu_uuid)
        # print(monitor.is_in_pmon(str(gpus_state.loc[gpu_uuid, "CPU_task_PID"])))

        pid_val = gpus_state.loc[gpu_uuid, "CPU_task_PID"]

        # Hard lesson: if the task crashes, we need to validate the GPU and bring it back
        if pd.isna(pid_val):
            gpus_state.loc[gpu_uuid, "validity"] = True
            gpus_state.at[gpu_uuid, "CPU_task_PID"] = pd.NA
            gpus_state.at[gpu_uuid, "gpu_seen_at"] = pd.NA
            continue
        
        pid = str(pid_val)

        # if process died, free the slot
        if not monitor.pid_on_system(pid):
            gpus_state.loc[gpu_uuid, "validity"] = True
            gpus_state.loc[gpu_uuid, "gpu_seen_at"] = pd.NA
            gpus_state.loc[gpu_uuid, "CPU_task_PID"] = pd.NA
            continue
        
        seen_at = gpus_state.at[gpu_uuid, "gpu_seen_at"]

        # set the first-seen timestamp once
        if pd.isna(seen_at) and monitor.is_in_pmon(pid):
            gpus_state.at[gpu_uuid, "gpu_seen_at"] = now
            seen_at = now

        # after grace, mark available (don’t require PMON to still show the PID)
        if not pd.isna(seen_at) and (now - float(seen_at) > 30):
            gpus_state.at[gpu_uuid, "validity"] = True
            gpus_state.at[gpu_uuid, "CPU_task_PID"] = pd.NA
            gpus_state.at[gpu_uuid, "gpu_seen_at"] = pd.NA


def all_available_GPUs():
    return gpus_state.index[gpus_state["validity"].fillna(False)].tolist()


# The process to keep track of available GPUs keeping track of process PIDs
# 1. initializing GPUs state
# print(gpu_UUIDs)

# print(gpus_state)

# 2. when adding a task to a GPU
# launch_task(GPU_UUID, TASK_PID)

# 3. having update in the loop to validate the GPUs for collocation
# update()

print("Initialized the gpus_state tracker: ", gpus_state)

# to get the list of available GPUs for collocation, not for exclusive policy
# for exclusive case, we need to get idle GPUs and make sure that non of them are in the CPU->GPU PID transfer
# it is way more efficient than waiting to make sure that a task moves from CPU --> GPU
# print(monitor.gpus_activeness())
# print(all_available_GPUs())
# exit()

# command generator function
def command_generator(dir, gpus_identifiers, command_to_execute, now, a):
    command = f"""cd {dir} ; \
                export CUDA_VISIBLE_DEVICES={gpus_identifiers} ; \
                exec 3>&1 ; \
                {{ time ( \
                    {{ \
                        conda run --no-capture-output -p /opt/miniconda3/envs/tf {command_to_execute} & pid=$! ; \
                        echo $pid >&3 ; \
                        wait $pid ; \
                        if [ $? -eq 0 ]; then \
                            echo 'Successful' >> {dir}/err-{now}-{a.task_id}.log ; \
                        else \
                            echo 'unsuccessful' >> {dir}/err-{now}-{a.task_id}.log ; \
                        fi ; \
                    }} 1> {dir}/out-{now}-{a.task_id}.log 2>> {dir}/err-{now}-{a.task_id}.log \
                ) ; }} 2> {dir}/time-{now}-{a.task_id}.et ; \
                exec 3>&-"""
    return command


# this function is responsible for implementing recovery method
# taking care of the system's robustness
def recovery(dirs = [globals()["recovery_dir"]]):
    # print("recovery process started ...")
    """
        This is the function that checks error files and adds OOM found to the high-priority queue
    """
    # going through all of the error files and detecting OOM error and adding to the recovery queue
    # for the next phase of scheduling

    # TODO: making it more general to go through submitted jobs from different users directories

    list_of_files = []

    for base in dirs:
        for file in os.listdir(base):
            if file.startswith("err") and file.endswith(".log"):
                file = os.path.join(base, file)
                list_of_files.append(file)


    crashes = 0
    all_executions = 0
    for iterator in list_of_files:
        if iterator in handled_crashes:                 # if a task is recovered once, no need to be added again
                                                        # initial policy is to recover and give an idle A100 GPU
            continue
        else:
            all_executions += 1
            file = open(f'{iterator}', 'r')             # reading a file
            Lines = file.readlines()                    # reading the lines of that file

            for line in Lines: # going through lines of an opened file to find if the execution crashed due to OOM
                if "unsuccessful" in line or "OOM" in line or "Non-OK-status" in line or "RESOURCE_EXHAUSTED" in line:
                    crashes += 1
                    
                    handled_crashes.append(iterator)    # We add it to the handled one to prevent over-scheduling
                    opener = open(f'{iterator}', 'r')   # We open the err file that has OOM 
                    Lines = opener.readlines()          # The goal is to fetch the information in the head of the err file

                    recovery_data = Lines[0].split('+')
                    # print(recovery_data)

                    tmp_dir = recovery_data[0]          # directory
                    tmp_file = recovery_data[3]         # rad file
                    tmp_user = recovery_data[4]         # user
                    tmp_task_id = recovery_data[5][:-1]      # task_id that was tokenized in the system

                    recovered_task = Task(tmp_user, tmp_dir, tmp_file)      # made the task object
                    recovered_task.set_id(tmp_task_id)                      # set the task_id to the initial one

                    recovered_task.set_if_recovered()
                    with recover_lock:
                        recovery_queue.enqueue(recovered_task)                  # putting the task in the recovery queue
                    print("OOM FOUND: recovery queue is filled with the task that has problem: ", recovered_task, recovered_task._to_string())
                    print("length of the queue:", recovery_queue.length())
                    logging.info(f"Recovered: {recovered_task}")
                    break
    # print("end of checking for failures ...")


# for launching bash command from Python
def command_executor(command):
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    pass

# the function getting tasks from submit interface and queuing them
def server():
    host = socket.gethostname()
    port = 5001

    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((host, port))

    while True:
        # configure how many client the server can listen simultaneously
        # It can be configured for any reason
        server_socket.listen(10)
        while True:
            conn, address = server_socket.accept()
            print("Connection from: " + str(address))
            data = conn.recv(1024).decode()

            if not data:
                break
            message = "Got your task and queued it."

            conn.send(message.encode())
            # print(data)
            
            user, dir, task = data.split('+')
            task = "/" + task[1:]
            
            print(user, dir, task)

            a = Task(user, dir, task)

            with lock:
                main_queue.enqueue(a)
                logging.info(f"queued {a.task_id} - {a.task}")

        conn.close()



# this is the module implementing different policies for 
# assigning GPUs to tasks/ collocate tasks under different policies
def scheduler(policy = policy):
    # exclusive, round-robin
    # oracle-first-fit, oracle-most-GMem-available, oracle-best-fit, oracle-least_GPU_utiltized
    # most-GMem-available-RR, least_GPU_utilized-RR, 
    # estimate-most-GMem-available (estimators needs to be set for the experiment first)

    # it needs to be a never-ending loops checking task queue, and GPUs, and making decisions
    estimator = globals()["estimator"]
    # horus
    # faketensor, GPUMemNet

    # This is the index where the system considers the estimation to be found
    esIndex = 8
    if estimator == "None":
        print("No Estimator!")
    elif estimator == "horus":
        print("horus, index = 9")
        esIndex = 9
    elif estimator == "faketensor":
        print("faketensor, index = 10")
        esIndex = 10
    elif estimator == "GPUMemNet":
        print("GPUMemNet, index = 11")
        esIndex = 11

    while True:
        time.sleep(1)

        # Scanning for OOM crashes
        # assumption: OOMs might happen because of the automatic collocation decisions
        # GPU memory estimators are not perfect
        # also GPU memory fragmentation can happen

        recovery()

        update()

        print("updated the table: ", gpus_state)

        print(command_executor("nvidia-smi pmon -c 1"))
        # print(gpus_state)
        # if there are some tasks waiting to be recovered/ served
        if main_queue.length() != 0 or recovery_queue.length() != 0:
            
            # 1. initial approach was to wait till the warmup was up
            # and the task has allocated its GPU memory requirement
            # but it could be unrepresentatively short or long for different tasks

            # here is the code body that implements exclusive policy
            # waiting as some tasks take time to start using GPU
            # time.sleep(60)


            # We improve the blindfolded waiting by tracking task PID
            # 2. we follow PIDs and consider monitoring data valid for next steps
            # when the process is moved from CPU to GPU

            idle_gpus_to_send_job = list()
            gpus_activeness = monitor.gpus_activeness()
            for gpu in gpus_activeness:
                if gpus_activeness[gpu] == 0:
                    idle_gpus_to_send_job.append(gpu)
            
            print("idle gpus: ", idle_gpus_to_send_job)
            print("available GPUs:", all_available_GPUs())

            # the logic that makes sure that we do not send task
            # to a GPU that is not available since it is waiting 
            # for its process to transfer from CPU to GPU
            idle_and_available = [g for g in idle_gpus_to_send_job if g in set(all_available_GPUs())]
            
            # checking the job at the head of the recovery/ queue
            a = None
            main_queue_flag = None
            user, dir, task = None, None, None

            # Having higher priority for the tasks that need to be recovered
            if recovery_queue.length() != 0:
                with recover_lock:
                    a = recovery_queue.check()
                user, dir, task = a.user, a.dir, a.task
                main_queue_flag = False
            else:
                with lock:
                    a = main_queue.check()
                user, dir, task = a.user, a.dir, a.task
                main_queue_flag = True

            # for reading the task specification
            command = f"cat {task}"
            print("this is what we want to parse and work on: ", task, command)
            ret = subprocess.run(command, capture_output=True, shell=True)
            commands = ret.stdout.decode()
            commands_to_execute = commands.split("\n")

            # finding conda environment name
            env_name = None
            for command in commands_to_execute:
                if "activate" in command:
                    env_name = commands_to_execute[1].split("activate")[1].strip()
                    break

            if env_name == None:
                    env_name = "tf"

            # enabling the conda environment
            environment = f"/opt/miniconda3/envs/{env_name}"
            print("conda environment to activate: ", env_name, environment)

            #Exclusive policy to check with 
            # finding the python code to execute 
            command_to_execute = None
            for command in commands_to_execute:
                if "python" in command:
                    command_to_execute = command
                    break
            if command_to_execute == None:
                print("the command could not be found in the submitted job profile!", task)

            number_of_GPUs_requested = int(commands_to_execute[7])

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            # we select idle GPUs that are also available
            if len(idle_and_available) >= number_of_GPUs_requested:
                assigned_gpus = idle_and_available[:number_of_GPUs_requested]

                a.set_service_time(now)
                a.set_status("dispatched")
                
                gpus_identifiers = ""
                for gpu in assigned_gpus:
                    if len(gpus_identifiers) > 0: 
                        gpus_identifiers += f",{gpu}"
                    else:
                        gpus_identifiers += f"{gpu}"

                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)                
                
                # discarding the task, as it has got the resources and got submitted for the execution
                if main_queue_flag == True:
                    main_queue.dequeue()
                else:
                    recovery_queue.dequeue()

                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'

                # logging the GPU to task mapping
                logging.info(f"dispatched {a.task_id} - {a.task} - {gpus_identifiers}")

                Thread(target = command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)

                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # now the task is mapped to a GPU, so it availability needs to be updated
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus:
                        launch_task(gpu_uuid, pid)


                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus)),
                        daemon=True
                    ).start()
                
                print(gpus_state)
                continue

            # ========================================================================================
            # ================================= ORACLE - FIRST FIT ===================================
            # ========================================================================================
            elif policy == "oracle-FF" and main_queue.length() != 0 and recovery_queue.length() == 0:

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True


                command = f"cat {task}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])
                print("memory requirement: ", gpu_memory_requirement)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()

                # --- thresholds ---

                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement & utils
                # ===============
                
                # --- filtering: memory + utilization gate (same logic as MAGM) ---
                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_requirement + 2048)
                ]
                
                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # only GPUs currently "available" by your adaptive state machine
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print("candidate & available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue

                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # FIRST-FIT: just take the first N rows as they appear (no sorting)
                assigned_gpus = candidate_gpus.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"


                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                Thread(target = command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
                
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()
                
                # print(gpus_state)

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Oracle-FF Collocated task on GPUs")

                continue

            # ========================================================================================
            # ================================= ORACLE - BEST FIT ===================================
            # ========================================================================================
            elif (policy == "oracle-BF") and (main_queue.length() != 0) and (recovery_queue.length() == 0):
                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True


                command = f"cat {task}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)

                print("command to execute found: ", command_to_execute)

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])
                print("memory requirement: ", gpu_memory_requirement)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()

                # --- thresholds ---

                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement & utils
                # ===============
                
                # --- filtering: memory + utilization gate (same logic as MAGM) ---
                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_requirement + 2048)
                ]
                
                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # only GPUs currently "available" by your adaptive state machine
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print("candidate & available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue

                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # BEST-FIT: pick the *smallest* sufficient GPUs
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=True, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"


                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'
                
                # now as we are here, shows that we have passed the conditions of having both 
                # enough number of GPUs, and enought GPU memory
                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                Thread(target = command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
                
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()
                
                # print(gpus_state)

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Oracle-BF Collocated task on GPUs")

                continue

            # ========================================================================================
            # ==================================== ORACLE - MAGM =====================================
            # ========================================================================================
            # === This policy collocates knowing the GMem, relying on the recovery method ============
            # =================== for OOMs due to memry fragmentation ================================
            # ========================================================================================
            elif policy == "oracle-MAGM" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True

                        
                command = f"cat {task}"

                # print(gpus_state)

                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("conda environment to activate: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)


                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])

                print("memory requirement: ", gpu_memory_requirement)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                
                print(gpus_with_metrics)


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ===============

                # Single-threshold policy (0–1 scale from DCGM)
                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_requirement + 2048)
                ]

                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # Keep only GPUs currently marked "available" by the availability logic
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print(gpus_state)

                print("candidate and available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                
                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # SORTING THE GPUS TO PRIORITIZE THE ONES WITH THE MOST AVAILABLE GPU MEMORY
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"


                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)

                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                Thread(target = command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
                
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()
                
                # print(gpus_state)

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Oracle-MAGM Collocated task on GPUs")

                continue

            # =========================================================================
            # ============================== Oracle - LUG =============================
            # =========================================================================
            # =========================================================================

            elif policy == "oracle-LUG" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True
                        
                command = f"cat {task}"
                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                # gpu memory requirement
                gpu_memory_requirement = int(commands_to_execute[8])

                print("memory requirement: ", gpu_memory_requirement)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                print(gpus_with_metrics)
                
                # Single-threshold policy (0–1 scale from DCGM)
                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                # condition 1: mem + utilization screen
                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_requirement + 2048)
                ]

                candidate_gpus = temp_.loc[
                        ~(
                            (temp_['smact'] >= THR_SMACT)
                            & (
                                (temp_['smocc'] >= THR_SMOCC)
                                | (temp_['drama'] >= THR_DRAMA)
                            )
                        )
                    ].copy()
                
                # Keep only GPUs currently marked "available" by the availability logic
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print(gpus_state)
                print("candidate and available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue

                # condition 2: number of GPUs requested
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # LUG: sort by smact ascending (least utilized first)
                sorted_ = candidate_gpus.sort_values(by="smact", ascending=True, kind="mergesort")
                assigned_gpus = sorted_.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)

                if main_queue_flag is True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")


                Thread(target=command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)

                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()

                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Oracle-LUG Collocated task on GPUs")


                continue


            # ========================================================================================
            # =========================== Round Robin with Recovery ==================================
            # ========================================================================================
            elif policy == "OR-RR" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True

                        
                command = f"cat {task}"
                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("conda environment to activate: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)


                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                avail = set(all_available_GPUs())
                assigned_gpus = []

                print(gpus_state)

                print("available GPUs: ", avail)

                # single pass over the whole RR ring
                N = len(GPU_IDs)
                seen = set()   

                while len(assigned_gpus) < number_of_GPUs_requested and len(seen) < N:
                    gid = next(round_robin_generator)  # RR pick
                    if gid in seen:
                        continue
                    seen.add(gid)
                    if gid in avail and gid not in assigned_gpus:
                        assigned_gpus.append(gid)                             

                if len(assigned_gpus) < number_of_GPUs_requested:
                    print("OR-RR: not enough available GPUs in this RR pass; skipping dispatch.")
                    continue
                
                
                print("assigned GPUs: ", assigned_gpus)

                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                
                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'
                
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                Thread(target=command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
            
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus:
                        launch_task(gpu_uuid, pid)
                    
                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus)),
                        daemon=True
                    ).start()

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Only Recovery - Round Robin Collocated!")

                continue

            # ============================================================================
            # =========================== OR - MAGM ======================================
            # ============================================================================
            elif policy == "OR-MAGM" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True

                        
                command = f"cat {task}"
                print("this is what we want collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)


                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                print(gpus_with_metrics)

                # ===== Filtering =====
                # Thresholds (same as MAGM)
                THR_SMACT = 0.80
                THR_SMOCC = 0.45
                THR_DRAMA = 0.40

                # 1) Fixed available-memory screen: keep GPUs with >= 5GB (5120 MiB) free
                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= 5120
                ]

                # 2) Utilization screen (exclude "hot" GPUs)
                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # 3) Availability screen (must be currently available)
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()


                print(gpus_state)
                print("candidate and available GPUs:\n", candidate_gpus)

                print("candidate GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                

                # ensure enough GPUs
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)


                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # Sort by most available GPU memory (desc), pick top-k
                sorted_ = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted_.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)
                
                if main_queue_flag is True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()


                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'
                
                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                Thread(target=command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)


                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "OR-MAGM (>= 5GB free) collocated task on GPUs.")

                continue

            # ============================================
            # ============================================
            # ================== OR-LUG ==================
            # ============================================
            # ============================================

            elif policy == "OR-LUG" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True
                        
                command = f"cat {task}"
                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")


                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                print(gpus_with_metrics)
                
                # Single-threshold policy
                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                # === Memory filter only: keep GPUs with >= 5GB available ===
                temp_ = gpus_with_metrics.loc[gpus_with_metrics['GPU_mem_available'] >= 5120]

                # utilization screen (same as before)
                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # Keep only GPUs currently marked "available" by the availability logic
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()


                print(gpus_state)
                print("candidate and available GPUs:\n", candidate_gpus)


                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    continue

                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                # LUG: sort by smact ascending (least utilized first)
                sorted_ = candidate_gpus.sort_values(by="smact", ascending=True, kind="mergesort")   
                assigned_gpus = sorted_.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # writing logs to the system log
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)

                if main_queue_flag is True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'


                Thread(target=command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
                
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()

                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "OR-LUG collocated task on GPUs.")

                
                continue









            # ========================================================================================
            # ========================================================================================
            # ================================ EST - MAGM ============================================
            # ========================================================================================
            # ========================================================================================
            elif policy == "EST-MAGM" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True

                        
                command = f"cat {task}"

                # print(gpus_state)

                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("conda environment to activate: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!", task)


                # gpu memory estimation
                gpu_memory_estimation = int(float(commands_to_execute[esIndex].strip()))

                print("memory estimation: ", gpu_memory_estimation)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                
                print(gpus_with_metrics)


                # Finding the GPUs that the task can get
                # ===============
                # condition 1 for filtering the GPUs based on the GPU memory requirement/ util
                # ===============

                # Single-threshold policy (0–1 scale from DCGM)
                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_estimation + 2048)
                ]

                candidate_gpus = temp_.loc[
                    ~(
                        (temp_['smact'] >= THR_SMACT)
                        & (
                            (temp_['smocc'] >= THR_SMOCC)
                            | (temp_['drama'] >= THR_DRAMA)
                        )
                    )
                ].copy()

                # Keep only GPUs currently marked "available" by the availability logic
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print(gpus_state)

                print("candidate and available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue
                
                # ===============
                # condition 2: checking for the number of GPUs requested
                # ===============
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)
                
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # SORTING THE GPUS TO PRIORITIZE THE ONES WITH THE MOST AVAILABLE GPU MEMORY
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")
                assigned_gpus = sorted.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"


                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)

                if main_queue_flag == True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()

                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")

                Thread(target = command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)
                
                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()
                
                # print(gpus_state)

                # just a message
                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "Oracle-MAGM Collocated task on GPUs")

                continue
            
            # =================================================
            # ================= EST - LUG =====================
            # =================================================
            elif policy == "EST-LUG" and (main_queue.length() != 0 and recovery_queue.length() == 0):

                a = None
                user, dir, task = None, None, None
                main_queue_flag = None

                # Having higher priority for the tasks that need to be recovered
                if recovery_queue.length() != 0:
                    with recover_lock:
                        a = recovery_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = False
                else:
                    with lock:
                        a = main_queue.check()
                    user, dir, task = a.user, a.dir, a.task
                    main_queue_flag = True
                        
                command = f"cat {task}"
                print("this is what we want to parse and work on and collocate: ", task, command)
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/opt/miniconda3/envs/{env_name}"
                print("environment: ", env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                # gpu memory estimation
                gpu_memory_estimation = int(float(commands_to_execute[esIndex].strip()))

                print("memory requirement: ", gpu_memory_estimation)

                # getting the monitored data about the server's GPUs
                gpus_with_metrics = monitor.analyze_Gmetrics()
                print(gpus_with_metrics)
                
                # Single-threshold policy (0–1 scale from DCGM)
                THR_SMACT = 0.80   # SMs quite busy
                THR_SMOCC = 0.45   # many resident warps
                THR_DRAMA = 0.40   # memory interface busy

                # condition 1: mem + utilization screen
                temp_ = gpus_with_metrics.loc[
                    gpus_with_metrics['GPU_mem_available'] >= (gpu_memory_estimation + 2048)
                ]

                candidate_gpus = temp_.loc[
                        ~(
                            (temp_['smact'] >= THR_SMACT)
                            & (
                                (temp_['smocc'] >= THR_SMOCC)
                                | (temp_['drama'] >= THR_DRAMA)
                            )
                        )
                    ].copy()
                
                # Keep only GPUs currently marked "available" by the availability logic
                avail = set(all_available_GPUs())
                candidate_gpus = candidate_gpus.loc[candidate_gpus.index.isin(avail)].copy()

                print(gpus_state)
                print("candidate and available GPUs:\n", candidate_gpus)

                if candidate_gpus.empty:
                    print("no candidate gpus at all!")
                    continue

                # condition 2: number of GPUs requested
                number_of_GPUs_requested = int(commands_to_execute[7])
                print("number of gpus requested: ", number_of_GPUs_requested)

                if len(candidate_gpus) < number_of_GPUs_requested:
                    print("Not enough GPUs to submit the task to!")
                    continue
                else:
                    print("The gpus that we can send the task to: \n", candidate_gpus)

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # LUG: sort by smact ascending (least utilized first)
                sorted_ = candidate_gpus.sort_values(by="smact", ascending=True, kind="mergesort")
                assigned_gpus = sorted_.head(number_of_GPUs_requested)

                print("assigned GPUs: ", assigned_gpus)
                a.set_service_time(now)
                a.set_status("dispatched")
                    
                gpus_identifiers = ""
                for gpu in assigned_gpus.index:
                    if len(gpus_identifiers) > 0:
                        gpus_identifiers += f",{gpu}"

                    else:
                        gpus_identifiers += f"{gpu}"

                # generating the command that will execute
                command = command_generator(dir, gpus_identifiers, command_to_execute, now, a)

                if main_queue_flag is True:
                    with lock:
                        main_queue.dequeue()
                else:
                    with recover_lock:
                        recovery_queue.dequeue()
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{task}+{user}+{a.task_id}" > {dir}/err-{now}-{a.task_id}.log'

                logging.info(f"dispatched {a.task_id} - {gpus_identifiers}")


                Thread(target=command_executor, args=(to_write,)).start()
                pid = launch_and_get_pid(command)

                if pid is None:
                    logging.error(f"Failed to capture PID for {a.task_id}; leaving GPUs available")
                else:
                    # Immediately mark these GPUs unavailable using the launcher PID
                    for gpu_uuid in assigned_gpus.index:
                        launch_task(gpu_uuid, pid)

                    # Resolve real GPU-using PID in the background and update state when found
                    Thread(
                        target=_async_resolve_and_update,
                        args=(pid, list(assigned_gpus.index)),
                        daemon=True
                    ).start()

                time_point = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                print(time_point, "EST-LUG Collocated task on GPUs")


            # ========================================================================================
            # =========================== Policy with resource estimator =============================
            # ========================================================================================
            # === This policy collocates based on estimation, relying on the recovery method =========
            # ========================================================================================

            elif policy == "ml_predictor" and recovery_queue.length() == 0:
                # using my resource predictor based on the gathered dataset 

                a = None
                user, dir, file = None, None, None

                with lock:
                    a = main_queue.dequeue()
                    user, dir, file = a.user, a.dir, a.file
                    a.set_service_time(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
                    a.set_status("dispatched")

                command = f"cd {dir} ; cat {file}"
                ret = subprocess.run(command, capture_output=True, shell=True)
                commands = ret.stdout.decode()
                commands_to_execute = commands.split("\n")

                now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                # finding conda environment name
                env_name = None
                for command in commands_to_execute:
                    if "activate" in command:
                        env_name = commands_to_execute[1].split("activate")[1].strip()
                        break
                if env_name == None:
                        env_name = "tf"

                # enabling the conda environment
                environment = f"/home/{user}/.conda/envs/{env_name}"
                print(env_name, environment)

                # finding the python code to execute 
                command_to_execute = None
                for command in commands_to_execute:
                    if "python" in command:
                        command_to_execute = command
                        break
                if command_to_execute == None:
                    print("the command could not be found in the submitted job profile!")

                print("command to execute found: ", command_to_execute)

                # sending the info for the parser to give out the feature for the estimator
                # file = "model.txt", dataset = "/raid/datasets/imagenet", batch_size = 32
                cnn_features, fc_features, overhead = rad_parser.analyze_model_summary(f"{dir}/{commands_to_execute[3]}", commands_to_execute[4], int(commands_to_execute[5]))

                # loading the memory consumption predictor :)
                global cnn_loaded_model
                global fc_loaded_model

                cnn_memory_predictor = cnn_loaded_model
                fc_memory_predictor = fc_loaded_model

                cnn_predicted_memory = cnn_memory_predictor.predict(cnn_features)
                fc_predicted_memory = fc_memory_predictor.predict(fc_features)

                print(cnn_predicted_memory, fc_predicted_memory, overhead)

                all_memory_estimation = cnn_predicted_memory[0] + fc_predicted_memory[0] + overhead
                # ============== getting the list of metrics and detecting GPUs that can be candidates for sending more tasks =========
                time.sleep(61)
                gpus_with_metrics = monitor.Gmetrics
                # print("decision: \n", gpus_with_metrics)
                temp_ = gpus_with_metrics.loc[gpus_with_metrics['GPU_mem_available'] > (all_memory_estimation)]
                candidate_gpus = temp_.loc[gpus_with_metrics['smact'] <= 0.8]

                # candidate_gpus = temp_
                
                # print("candidate GPUs:\n", candidate_gpus)

                # by="GPU_mem_available", ascending=False, kind="mergesort"
                # sorted = candidate_gpus.sort_values(by="smact", kind="mergesort")
                sorted = candidate_gpus.sort_values(by="GPU_mem_available", ascending=False, kind="mergesort")

                print("gpus sorted:\n", sorted)

                if candidate_gpus.empty:
                    print("No GPUs to submit job to!")
                    with lock:
                        main_queue.put_it_back(a)
                    continue
                else:
                    print("The gpus that we can send job to :) \n", candidate_gpus)
                    
                    # GPU selected here :)
                    candidate_gpu_to_collocate_job = sorted.index[0]

                    print("candidate GPU: ", candidate_gpu_to_collocate_job)
                # ====================================================================

                # TODO: looking for the list of the GPUs to find the first-fit
                # TODO: it can also be best-fit

                # writing logs to the system log
                logging.info(f"dispatched {a.task_id} - {candidate_gpu_to_collocate_job}")

                # generating the command that will execute
                # command = f'cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={idle_gpu_to_send_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2> err-{user}-{now}-{file}-{a.task_id}.log ; }} 1> time1-{user}-{now}-{file}-{a.task_id}.et 2> time2-{a.task_id}.et || echo "fail" &'
                command = f"""cd {dir} ; . /opt/anaconda/etc/profile.d/conda.sh ; conda activate {environment} ; export CUDA_VISIBLE_DEVICES={candidate_gpu_to_collocate_job} ; {{ time {command_to_execute} 1> out-{user}-{now}-{file}-{a.task_id}.log 2>> err-{user}-{now}-{file}-{a.task_id}.log ; }} 2> time-{user}-{now}-{file}-{a.task_id}.et & pid=$!
                    wait $pid 
                    if [ $? -eq 0 ]; then
                        echo 'Successful' >> err-{user}-{now}-{file}-{a.task_id}.log
                    else
                        echo 'unsuccessful' >>  err-{user}-{now}-{file}-{a.task_id}.log
                    fi
                    """
                
                to_write = f'echo "{dir}+{environment}+{command_to_execute}+{file}+{user}+{a.task_id}" > err-{user}-{now}-{file}-{a.task_id}.log'

                # print(command)
                # print(to_write)

                # subprocess.run(to_write, shell=True, check=True, executable='/bin/bash')
                # subprocess.run(command, shell=True, check=True, executable='/bin/bash')
                Thread(target = command_executor, args=(to_write,)).start()
                Thread(target = command_executor, args=(command,)).start()

            timepoint = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print(timepoint, "Number of tasks waiting in the queue: ", main_queue.length())           
        else:
            # no task in the waiting queue
            # print("No task in the queue!")
            pass
    
    # time to check if there has been any crashes to be handled


if __name__ == '__main__':
    # server module responsible for getting tasks and queuing them
    Thread(target = server).start()

    # scheduler module responsible for selecting from the queue
    # and mapping them on the available GPUs
    # GPU collocation happens in this 
    Thread(target = scheduler).start()

    # monitoring facilities
    # module responsible for monitoring GPUs
    # keeping latest window of monitored data updated inside "Gmetrics"
    # to hand it over to decision making process
    Thread(target = monitor.monitor_logger).start()

    # for logging system wide statistics for further study
    # Thread(target = system_use_utilization_logger).start()    # no need to keep this since is inaccurate
    Thread(target = monitor.top_system_logger).start()