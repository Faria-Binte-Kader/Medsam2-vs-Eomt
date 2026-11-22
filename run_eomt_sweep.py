"""Run the whole EoMT learning-curve sweep with identical settings, spread over the available GPUs.

  python run_eomt_sweep.py --dry-run                  # list the jobs and the exact commands
  python run_eomt_sweep.py --gpus 0 1 2               # train, select on validation, prune weights; one job per GPU at a time

Jobs: the six frozen nested sizes (1, 2, 5, 10, 20, all) plus extra random orderings (--extra-orders, default 1 2) for the
small sizes (--extra-sizes, default 1 2), which only measure how much a tiny subset depends on which patients were drawn.
Every job uses the same steps, learning rate, tumor-free ratio and seed (the values below were fixed from the validation
pilot, before any test evaluation). Each job: train -> `eval_eomt.py --select --prune` (checkpoint chosen on validation only).
A finished job (best_step.json present) is skipped, so the sweep can be restarted. The test split is not touched here.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import config as C

SIZES = ["all", "20", "10", "5", "2", "1"]  # longest first so the GPUs stay busy
SETTINGS = {"steps": 6000, "val_every": 1000, "ckpt_every": 250}


def job_name(size, order):
    return f"sweep_n{size}" + (f"_order{order}" if order else "")


def make_jobs(extra_orders, extra_sizes):
    jobs = [(s, 0) for s in SIZES] + [(s, o) for o in extra_orders for s in extra_sizes]
    return jobs


def commands(size, order, steps, val_every, ckpt_every):
    name = job_name(size, order)
    train = [sys.executable, "train_eomt.py", "--n-patients", size, "--steps", str(steps), "--compile",
             "--val-every", str(val_every), "--ckpt-every", str(ckpt_every), "--order-seed", str(order), "--run-name", name]
    select = [sys.executable, "eval_eomt.py", "--run-dir", str(C.OUTPUT_DIR / "eomt" / name), "--select", "--prune"]
    return name, train, select


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids to use, one job at a time on each")
    ap.add_argument("--steps", type=int, default=SETTINGS["steps"])
    ap.add_argument("--extra-orders", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--extra-sizes", nargs="*", default=["1", "2"], choices=SIZES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = make_jobs(a.extra_orders, a.extra_sizes)
    pending = []
    for size, order in jobs:
        name, train, select = commands(size, order, a.steps, SETTINGS["val_every"], SETTINGS["ckpt_every"])
        done = (C.OUTPUT_DIR / "eomt" / name / "best_step.json").exists()
        print(f"{'done   ' if done else 'pending'} {name}")
        if a.dry_run:
            print("   ", " ".join(train), "\n   ", " ".join(select))
        elif not done:
            pending.append((name, train, select))
    if a.dry_run or not pending:
        return

    log_dir = C.OUTPUT_DIR / "eomt" / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running = {}  # gpu -> (process, name)
    failed = []
    while pending or running:
        for gpu in a.gpus:
            if gpu not in running and pending:
                name, train, select = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                shell = " && ".join(shlex.join(map(str, c)) for c in (train, select))
                log = open(log_dir / f"{name}.log", "w")
                running[gpu] = (subprocess.Popen(shell, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), name)
                print(f"[gpu {gpu}] started {name}", flush=True)
        time.sleep(30)
        for gpu, (proc, name) in list(running.items()):
            if proc.poll() is not None:
                print(f"[gpu {gpu}] {'finished' if proc.returncode == 0 else 'FAILED'} {name} (log: {log_dir / (name + '.log')})", flush=True)
                if proc.returncode != 0:
                    failed.append(name)
                del running[gpu]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
