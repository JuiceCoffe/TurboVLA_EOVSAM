#!/usr/bin/env python3
"""Launch training with persistent process and system memory diagnostics."""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

PROC = Path("/proc")
MEM_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal",
            "SwapFree", "SwapCached", "Shmem", "Slab", "SReclaimable", "AnonPages")
STATUS_KEYS = ("VmRSS", "VmHWM", "VmSize", "VmSwap", "RssAnon", "RssFile", "RssShmem", "Threads")
PSS_KEYS = ("Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Private_Clean", "Private_Dirty", "SwapPss")

def utc_now():
    return datetime.now().astimezone().isoformat()

def key_values(path, keys):
    result = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in keys:
            result[key] = int(rest.strip().split()[0])
    return result

def process_inventory(root_pid, detailed):
    inventory = {}
    for folder in PROC.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            status = folder.joinpath("status").read_text()
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            pid = int(folder.name)
            row = {"pid": pid, "ppid": int(fields["PPid"]), "name": fields["Name"].strip(),
                   "uid": int(fields["Uid"].split()[0]),
                   "memory_kib": {k: int(fields[k].split()[0]) for k in STATUS_KEYS if k in fields}}
            inventory[pid] = row
        except (OSError, ValueError, KeyError):
            continue
    descendants = {root_pid}
    while True:
        found = {p for p, row in inventory.items() if row["ppid"] in descendants}
        if found <= descendants:
            break
        descendants |= found
    tracked = []
    for pid in sorted(descendants):
        if pid not in inventory:
            continue
        row = inventory[pid]
        folder = PROC / str(pid)
        try:
            cmd = folder.joinpath("cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
            row["role"] = ("launcher" if pid == root_pid else
                           "data_worker" if row["name"] == "pt_data_worker" else
                           "evaluation" if "experiments/libero/evaluate.py" in cmd else
                           "train_rank" if "experiments/libero/train.py" in cmd else
                           "wandb" if "wandb" in cmd else "other_child")
            env = folder.joinpath("environ").read_bytes().split(b"\0")
            row["rank"] = next((x.split(b"=", 1)[1].decode() for x in env if x.startswith(b"RANK=")), None)
            stat = folder.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            row["state"] = stat[0]
            row["cpu_ticks"] = int(stat[11]) + int(stat[12])
            row["start_ticks"] = int(stat[19])
            row["major_faults"] = int(stat[9])
            if detailed:
                try:
                    row["pss_kib"] = key_values(folder / "smaps_rollup", PSS_KEYS)
                except OSError as exc:
                    row["pss_error"] = str(exc)
        except OSError:
            row["exited_during_sample"] = True
        tracked.append(row)
    totals = {}
    for row in tracked:
        role = row.get("role", "unknown")
        total = totals.setdefault(role, {"count": 0, "rss_kib": 0, "swap_kib": 0})
        total["count"] += 1
        total["rss_kib"] += row["memory_kib"].get("VmRSS", 0)
        total["swap_kib"] += row["memory_kib"].get("VmSwap", 0)
        if "pss_kib" in row:
            total["pss_kib"] = total.get("pss_kib", 0) + row["pss_kib"].get("Pss", 0)
            total["private_kib"] = total.get("private_kib", 0) + row["pss_kib"].get("Private_Clean", 0) + row["pss_kib"].get("Private_Dirty", 0)
    other = sorted((row for pid, row in inventory.items() if pid not in descendants),
                   key=lambda r: r["memory_kib"].get("VmRSS", 0), reverse=True)[:10]
    return tracked, totals, other

def cgroup_memory(root_pid):
    result = {}
    try:
        entries = (PROC / str(root_pid) / "cgroup").read_text().splitlines()
        v2 = next((x.split(":", 2)[2] for x in entries if x.startswith("0::")), None)
        if v2 is not None:
            folder = Path("/sys/fs/cgroup") / v2.lstrip("/")
            for name in ("memory.current", "memory.max", "memory.events", "memory.events.local"):
                path = folder / name
                if path.exists():
                    result[name] = path.read_text().strip()
    except OSError as exc:
        result["read_error"] = str(exc)
    return result

def sample(root_pid, detailed, progress):
    mem = key_values(PROC / "meminfo", MEM_KEYS)
    tracked, totals, other = process_inventory(root_pid, detailed)
    vmstat = {}
    for line in (PROC / "vmstat").read_text().splitlines():
        key, value = line.split()
        if key in ("pswpin", "pswpout", "pgmajfault", "oom_kill", "pgscan_kswapd", "pgscan_direct"):
            vmstat[key] = int(value)
    result = {"time": utc_now(), "monotonic": time.monotonic(), "root_pid": root_pid,
              "progress": dict(progress), "system_memory_kib": mem, "vmstat": vmstat,
              "detailed_pss": detailed, "processes": tracked, "role_totals": totals,
              "other_top_rss": other, "cgroup": cgroup_memory(root_pid)}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        result["gpu_csv"] = gpu.stdout.strip()
        if gpu.returncode:
            result["gpu_error"] = gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["gpu_error"] = str(exc)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--pss-interval", type=float, default=300)
    parser.add_argument("--low-memory-gb", type=float, default=40)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0 or args.pss_interval <= 0:
        parser.error("command and positive intervals are required")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd()
    hashes = {}
    for rel in ("turbovla/training/trainer.py", "turbovla/training/pi05.py",
                "turbovla/data/mixed_suite.py", "turbovla/models/eovsam_encoder.py",
                "scripts/launch_libero_4gpu_hustvl40.sh", "scripts/launch_libero_2gpu_hustvl2027.sh", "scripts/start_monitored_libero_2gpu.sh", "scripts/monitor_training_memory.py"):
        file = repo / rel
        if file.is_file():
            hashes[rel] = hashlib.sha256(file.read_bytes()).hexdigest()
    metadata = {"started": utc_now(), "command": command, "cwd": str(repo),
                "run_timestamp": os.environ.get("RUN_TIMESTAMP"),
                "num_workers": os.environ.get("NUM_WORKERS", "2"), "interval_seconds": args.interval,
                "pss_interval_seconds": args.pss_interval, "low_memory_gb": args.low_memory_gb,
                "code_sha256": hashes, "host": os.uname().nodename}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    metadata["root_pid"] = process.pid
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Memory monitor: {out} (root PID {process.pid})", flush=True)
    stop = threading.Event()
    progress = {"stage": "initialization", "global_step": None}
    def watch():
        last_pss = -float("inf")
        with (out / "memory.jsonl").open("a", buffering=1) as memory:
            while not stop.is_set():
                try:
                    mem = key_values(PROC / "meminfo", MEM_KEYS)
                    detailed = time.monotonic() - last_pss >= args.pss_interval or mem["MemAvailable"] / 1024**2 < args.low_memory_gb
                    if detailed:
                        last_pss = time.monotonic()
                    snapshot = sample(process.pid, detailed, progress)
                    memory.write(json.dumps(snapshot) + "\n")
                except Exception as exc:
                    memory.write(json.dumps({"time": utc_now(), "monitor_error": repr(exc)}) + "\n")
                stop.wait(args.interval)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    interrupted = False
    def handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    tail = ""
    try:
        with (out / "training.log").open("ab", buffering=0) as log:
            while True:
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                log.write(chunk)
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                tail = (tail + text)[-8192:]
                steps = list(re.finditer(r"training:[^\r\n]*?(\d+)/(\d+)", tail))
                if steps:
                    progress["global_step"] = int(steps[-1].group(1))
                    progress["max_steps"] = int(steps[-1].group(2))
                markers = list(re.finditer(r"training:[^\r\n]*?\d+/\d+|evaluating .*? at step \d+|evaluation metrics at step \d+", text))
                for marker in markers:
                    progress["stage"] = "evaluation" if marker.group().startswith("evaluating") else "training"
                if "model-load lock" in text or "host RAM available" in text:
                    progress["stage"] = "evaluation_model_load"
                if re.search(r"task=\d+ episode=\d+ success=", text):
                    progress["stage"] = "evaluation_rollout"
        code = process.wait()
    finally:
        stop.set()
        thread.join(timeout=12)
        process.stdout.close()
    (out / "exit.json").write_text(json.dumps({"time": utc_now(), "exit_code": code, "interrupted": interrupted, "progress": progress}, indent=2) + "\n")
    return code if code >= 0 else 128 - code

if __name__ == "__main__":
    sys.exit(main())
