#!/usr/bin/env python3
"""Config-driven remote runner for ANN-Benchmarks.

This orchestrates the upstream `erikbern/ann-benchmarks` workflow on a remote
Linux host over SSH:

1. prepares or updates a remote checkout,
2. optionally syncs local overlay files (for custom algorithms),
3. builds the required Docker images,
4. runs `python run.py` dataset by dataset with remote monitoring,
5. exports CSV tables and renders the website/plots,
6. syncs artifacts back locally.

The goal is reproducible runs with explicit guardrails against wasted compute.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_OUTPUT_ROOT = REPO / "benchmarks" / "annb_remote_runs"


class CommandError(RuntimeError):
    pass


@dataclass
class RemoteTarget:
    host: str
    user: str | None = None
    port: int | None = None
    identity_file: str | None = None
    extra_ssh_args: list[str] | None = None

    def destination(self) -> str:
        return f"{self.user + '@' if self.user else ''}{self.host}"


@dataclass
class RunResult:
    dataset: str
    algorithm: str | None
    elapsed: float | None
    expected_results: int
    status: str
    error: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def info(msg: str) -> None:
    print(msg, flush=True)


def shell_join(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(x) for x in args)


def remote_quote(text: str) -> str:
    return shlex.quote(text)


def build_ssh_base(target: RemoteTarget) -> list[str]:
    cmd = ["ssh"]
    if target.port:
        cmd.extend(["-p", str(target.port)])
    if target.identity_file:
        cmd.extend(["-i", target.identity_file])
    if target.extra_ssh_args:
        cmd.extend(target.extra_ssh_args)
    cmd.append(target.destination())
    return cmd


def run_local(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if check and proc.returncode != 0:
        raise CommandError(proc.stdout or f"command failed: {shell_join(cmd)}")
    return proc


def run_remote(
    target: RemoteTarget,
    script: str,
    *,
    capture: bool = True,
    check: bool = True,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> subprocess.CompletedProcess[str]:
    remote_argv = ["bash", "-lc", script]
    if sudo:
        if sudo_pythonpath:
            remote_argv = ["sudo", "env", f"PYTHONPATH={sudo_pythonpath}", *remote_argv]
        else:
            remote_argv = ["sudo", *remote_argv]
    remote_cmd = shell_join(remote_argv)
    cmd = [*build_ssh_base(target), remote_cmd]
    return run_local(cmd, capture=capture, check=check)


def rsync_to_remote(
    target: RemoteTarget,
    local_path: Path,
    remote_path: str,
    *,
    delete: bool = False,
) -> None:
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    cmd.extend(["--exclude", ".git", "--exclude", "__pycache__"])
    if target.port:
        ssh_cmd = ["ssh", "-p", str(target.port)]
        if target.identity_file:
            ssh_cmd.extend(["-i", target.identity_file])
        if target.extra_ssh_args:
            ssh_cmd.extend(target.extra_ssh_args)
        cmd.extend(["-e", shell_join(ssh_cmd)])
    elif target.identity_file or target.extra_ssh_args:
        ssh_cmd = ["ssh"]
        if target.identity_file:
            ssh_cmd.extend(["-i", target.identity_file])
        if target.extra_ssh_args:
            ssh_cmd.extend(target.extra_ssh_args)
        cmd.extend(["-e", shell_join(ssh_cmd)])

    source = str(local_path)
    if local_path.is_dir() and not source.endswith("/"):
        source += "/"
    cmd.extend([source, f"{target.destination()}:{remote_path}"])
    run_local(cmd)


def rsync_from_remote(
    target: RemoteTarget,
    remote_path: str,
    local_path: Path,
    *,
    optional: bool = False,
) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az"]
    if target.port:
        ssh_cmd = ["ssh", "-p", str(target.port)]
        if target.identity_file:
            ssh_cmd.extend(["-i", target.identity_file])
        if target.extra_ssh_args:
            ssh_cmd.extend(target.extra_ssh_args)
        cmd.extend(["-e", shell_join(ssh_cmd)])
    elif target.identity_file or target.extra_ssh_args:
        ssh_cmd = ["ssh"]
        if target.identity_file:
            ssh_cmd.extend(["-i", target.identity_file])
        if target.extra_ssh_args:
            ssh_cmd.extend(target.extra_ssh_args)
        cmd.extend(["-e", shell_join(ssh_cmd)])
    cmd.extend([f"{target.destination()}:{remote_path}", str(local_path)])
    run_local(cmd, check=not optional)


def require(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"missing required config key: {key}")
    return config[key]


def remote_python_json(target: RemoteTarget, checkout: str, code: str) -> Any:
    proc = run_remote(
        target,
        f"cd {remote_quote(checkout)} && python3 - <<'PY'\n{code}\nPY",
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("remote python command returned no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "remote python command did not end with JSON output:\n"
            f"{proc.stdout.rstrip()}"
        ) from exc


def resolve_remote_root(target: RemoteTarget, remote_root: str) -> str:
    proc = run_remote(
        target,
        f"python3 - <<'PY'\nimport os\nprint(os.path.expanduser({remote_root!r}))\nPY",
    )
    return proc.stdout.strip()


def resolve_remote_user_site(target: RemoteTarget) -> str:
    proc = run_remote(
        target,
        "python3 - <<'PY'\nimport site\nprint(site.getusersitepackages())\nPY",
    )
    return proc.stdout.strip()


def resolve_remote_cpu_count(target: RemoteTarget) -> int:
    proc = run_remote(
        target,
        "python3 - <<'PY'\nimport os\nprint(os.cpu_count() or 1)\nPY",
    )
    return max(1, int(proc.stdout.strip()))


def terminate_remote_job(
    target: RemoteTarget,
    pid: int,
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> None:
    script = f"""
set +e
kill -- -{pid} 2>/dev/null || true
sleep 2
kill -9 -- -{pid} 2>/dev/null || true
"""
    run_remote(target, script, check=False, sudo=sudo, sudo_pythonpath=sudo_pythonpath)


def touch_remote_file(
    target: RemoteTarget,
    path: str,
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> None:
    run_remote(
        target,
        f"mkdir -p {remote_quote(os.path.dirname(path))} && touch {remote_quote(path)}",
        check=False,
        sudo=sudo,
        sudo_pythonpath=sudo_pythonpath,
    )


def cleanup_remote_state(
    target: RemoteTarget,
    remote_checkout: str,
    *,
    clear_outputs: bool = False,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> None:
    info("==> Cleanup stale remote benchmark state")
    script = f"""
set +e
pkill -f {remote_quote(remote_checkout)} 2>/dev/null || true
pkill -f 'python -u run_algorithm.py' 2>/dev/null || true
docker ps -a --format '{{{{.ID}}}} {{{{.Image}}}}' | awk '$2 ~ /^ann-benchmarks/ {{print $1}}' | xargs -r docker rm -f
docker container prune -f >/dev/null 2>&1 || true
"""
    if clear_outputs:
        script += f"""
rm -rf {remote_quote(remote_checkout)}/results
rm -rf {remote_quote(remote_checkout)}/figures
rm -rf {remote_quote(remote_checkout)}/benchmarks/remote_logs
mkdir -p {remote_quote(remote_checkout)}/benchmarks/remote_logs
"""
    run_remote(target, script, check=False, sudo=sudo, sudo_pythonpath=sudo_pythonpath)


def remote_preflight(target: RemoteTarget) -> None:
    info("==> Remote preflight")
    script = """
set -euo pipefail
for x in bash git python3 docker rsync; do
  command -v "$x" >/dev/null || { echo "missing:$x"; exit 1; }
done
python3 --version
docker --version
uname -a
"""
    proc = run_remote(target, script)
    print(proc.stdout.rstrip())


def prepare_checkout(
    target: RemoteTarget,
    repo_cfg: dict[str, Any],
    run_root: str,
) -> str:
    repo_url = require(repo_cfg, "url")
    ref = repo_cfg.get("ref", "main")
    checkout_dir = repo_cfg.get("checkout_dir", "ann-benchmarks")
    remote_checkout = f"{run_root.rstrip('/')}/{checkout_dir}"
    info("==> Prepare remote checkout")
    script = f"""
set -euo pipefail
mkdir -p {remote_quote(run_root)}
if [ ! -d {remote_quote(remote_checkout)}/.git ]; then
  git clone --branch {remote_quote(ref)} --depth 1 {remote_quote(repo_url)} {remote_quote(remote_checkout)}
fi
cd {remote_quote(remote_checkout)}
git fetch --tags origin
git checkout {remote_quote(ref)}
git pull --ff-only origin {remote_quote(ref)} || true
python3 -m pip install -r requirements.txt
"""
    proc = run_remote(target, script)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    return remote_checkout


def sync_overlays(target: RemoteTarget, remote_checkout: str, sync_items: list[dict[str, Any]]) -> None:
    if not sync_items:
        return
    info("==> Sync local overlays")
    for item in sync_items:
        local_cfg_path = Path(require(item, "local_path")).expanduser()
        local_path = local_cfg_path if local_cfg_path.is_absolute() else (REPO / local_cfg_path).resolve()
        remote_rel = require(item, "remote_rel_path").lstrip("/")
        remote_path = f"{remote_checkout.rstrip('/')}/{remote_rel}"
        delete = bool(item.get("delete", False))
        if not local_path.exists():
            raise FileNotFoundError(f"overlay path does not exist: {local_path}")
        run_remote(
            target,
            f"mkdir -p {remote_quote(remote_path if local_path.is_dir() else os.path.dirname(remote_path))}",
        )
        rsync_to_remote(target, local_path, remote_path, delete=delete)
        info(f"  synced {local_path} -> {remote_rel}")


def resolve_dataset_list(
    target: RemoteTarget,
    remote_checkout: str,
    datasets: list[str] | None,
) -> list[str]:
    if datasets:
        return datasets
    info("==> Resolve dataset list from remote checkout")
    code = """
import json
from ann_benchmarks.datasets import DATASETS
print(json.dumps(sorted(DATASETS.keys())))
"""
    return remote_python_json(target, remote_checkout, code)


def build_algorithms(
    target: RemoteTarget,
    remote_checkout: str,
    build_cfg: dict[str, Any],
    build_dirs_override: list[str] | None = None,
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> None:
    if build_cfg.get("skip"):
        return
    build_dirs = build_dirs_override if build_dirs_override is not None else (build_cfg.get("algorithm_dirs") or [])
    base_image = build_cfg.get("base_image", "ann-benchmarks")
    proc = int(build_cfg.get("proc", 1))
    build_args = build_cfg.get("build_args") or []
    build_arg_flags = " ".join(f"--build-arg {remote_quote(x)}" for x in build_args)
    info("==> Build Docker images")
    if build_dirs:
        lines = [
            "set -euo pipefail",
            f"cd {remote_quote(remote_checkout)}",
            f"docker build {build_arg_flags} --rm -t {remote_quote(base_image)} -f ann_benchmarks/algorithms/base/Dockerfile .",
        ]
        for alg in build_dirs:
            if alg == "base":
                continue
            lines.append(
                f"docker build {build_arg_flags} --rm -t ann-benchmarks-{remote_quote(alg)} "
                f"-f ann_benchmarks/algorithms/{remote_quote(alg)}/Dockerfile ."
            )
        proc_out = run_remote(target, "\n".join(lines), sudo=sudo, sudo_pythonpath=sudo_pythonpath)
    else:
        proc_out = run_remote(
            target,
            f"cd {remote_quote(remote_checkout)} && python3 install.py --proc {proc}" +
            (f" {build_arg_flags}" if build_arg_flags else ""),
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
    if proc_out.stdout.strip():
        print(proc_out.stdout.rstrip())


def resolve_build_dir(
    algorithm: str | None,
    build_cfg: dict[str, Any],
) -> str | None:
    if algorithm is None:
        return None
    mapping = build_cfg.get("algorithm_dir_by_algorithm") or {}
    if algorithm in mapping:
        return str(mapping[algorithm])
    build_dirs = set(build_cfg.get("algorithm_dirs") or [])
    if algorithm in build_dirs:
        return algorithm
    if "-" in algorithm:
        prefix = algorithm.split("-", 1)[0]
        if prefix in build_dirs:
            return prefix
    return algorithm


def ensure_algorithm_built(
    target: RemoteTarget,
    remote_checkout: str,
    algorithm: str | None,
    build_cfg: dict[str, Any],
    built_dirs: set[str],
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> None:
    build_dir = resolve_build_dir(algorithm, build_cfg)
    dirs_to_build: list[str] = []
    if "base" not in built_dirs:
        dirs_to_build.append("base")
    if build_dir and build_dir not in built_dirs:
        dirs_to_build.append(build_dir)
    if not dirs_to_build:
        return
    build_algorithms(
        target,
        remote_checkout,
        build_cfg,
        build_dirs_override=dirs_to_build,
        sudo=sudo,
        sudo_pythonpath=sudo_pythonpath,
    )
    built_dirs.update(dirs_to_build)


def apply_parameter_pruning(
    target: RemoteTarget,
    remote_checkout: str,
    parameter_cfg: dict[str, Any],
    allowed_algorithm_dirs: set[str] | None = None,
) -> None:
    if not parameter_cfg or not parameter_cfg.get("enabled", False):
        return
    info("==> Prune algorithm parameter grids in remote config.yml files")
    payload = json.dumps(parameter_cfg)
    allowed = sorted(allowed_algorithm_dirs) if allowed_algorithm_dirs else []
    code = f"""
import json
from pathlib import Path
import yaml

cfg = json.loads({payload!r})
base_dir = Path("ann_benchmarks/algorithms")
global_cfg = cfg.get("global", {{}})
per_algo = cfg.get("algorithms", {{}})
allowed = {allowed!r}

def limit_value(value, keep, mode):
    if keep is None:
        return value
    if isinstance(value, list):
        return value[-keep:] if mode == "tail" else value[:keep]
    return value

def prune_run_group(run_group, rules):
    keep_args = rules.get("keep_args")
    keep_query_args = rules.get("keep_query_args")
    mode = rules.get("select", "head")

    if "args" in run_group:
        run_group["args"] = limit_value(run_group["args"], keep_args, mode)
    if "query_args" in run_group:
        run_group["query_args"] = limit_value(run_group["query_args"], keep_query_args, mode)
    if "arg_groups" in run_group:
        run_group["arg_groups"] = limit_value(run_group["arg_groups"], keep_args, mode)
    if "query_arg_groups" in run_group:
        run_group["query_arg_groups"] = limit_value(run_group["query_arg_groups"], keep_query_args, mode)
    return run_group

def maybe_replace_run_groups(entry, rules, metric):
    replacements = rules.get("replace_run_groups_by_definition", {{}})
    metric_replacements = rules.get("replace_run_groups_by_metric_definition", {{}})
    name = entry.get("name")
    if metric in metric_replacements and name in metric_replacements[metric]:
        entry["run_groups"] = metric_replacements[metric][name]
        return True
    if name in replacements:
        entry["run_groups"] = replacements[name]
        return True
    return False

for config_path in base_dir.glob("*/config.yml"):
    algo_dir = config_path.parent.name
    if allowed and algo_dir not in allowed:
        continue
    rules = dict(global_cfg)
    rules.update(per_algo.get(algo_dir, {{}}))
    if not rules:
        continue
    if "enabled" in rules and not rules["enabled"]:
        continue

    with config_path.open() as fh:
        data = yaml.safe_load(fh)

    changed = False
    for point_type, point_cfg in (data or {{}}).items():
        if not isinstance(point_cfg, dict):
            continue
        for metric, entries in point_cfg.items():
            if not isinstance(entries, list):
                continue
            keep_defs = rules.get("keep_definitions")
            select_mode = rules.get("select", "head")
            if keep_defs is not None:
                entries = entries[-keep_defs:] if select_mode == "tail" else entries[:keep_defs]
                point_cfg[metric] = entries
                changed = True
            for entry in entries:
                if maybe_replace_run_groups(entry, rules, metric):
                    changed = True
                    continue
                run_groups = entry.get("run_groups", {{}})
                keep_run_groups = rules.get("keep_run_groups")
                if keep_run_groups is not None:
                    items = list(run_groups.items())
                    items = items[-keep_run_groups:] if select_mode == "tail" else items[:keep_run_groups]
                    entry["run_groups"] = dict(items)
                    run_groups = entry["run_groups"]
                    changed = True
                for rg_name, rg in run_groups.items():
                    prune_run_group(rg, rules)
                    changed = True

    if changed:
        with config_path.open("w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        print(f"pruned {{config_path}}")
"""
    proc = run_remote(
        target,
        f"cd {remote_quote(remote_checkout)} && python3 - <<'PY'\n{code}\nPY",
    )
    if proc.stdout.strip():
        print(proc.stdout.rstrip())


def estimate_expected_results(
    target: RemoteTarget,
    remote_checkout: str,
    dataset: str,
    count: int,
    algorithm: str | None,
    run_disabled: bool,
) -> int:
    code = f"""
import json
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.definitions import get_definitions
dataset_name = {dataset!r}
dataset, dimension = get_dataset(dataset_name)
defs = get_definitions(
    dimension=dimension,
    point_type=dataset.attrs.get("point_type", "float"),
    distance_metric=dataset.attrs["distance"],
    count={count},
    base_dir="ann_benchmarks/algorithms",
)
if {algorithm!r} is not None:
    defs = [d for d in defs if d.algorithm == {algorithm!r}]
if not {run_disabled!r}:
    defs = [d for d in defs if not d.disabled]
total = sum(len(d.query_argument_groups or [[]]) for d in defs)
print(json.dumps({{"expected": total}}))
"""
    try:
        return int(remote_python_json(target, remote_checkout, code)["expected"])
    except Exception as exc:
        info(
            f"warning: could not estimate expected results for dataset={dataset}"
            f" algorithm={algorithm or 'all'}: {exc}"
        )
        return 0


def launch_dataset_run(
    target: RemoteTarget,
    remote_checkout: str,
    dataset: str,
    run_cfg: dict[str, Any],
    algorithm: str | None,
    *,
    parallelism_override: int | None = None,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
 ) -> tuple[int, str, str, str]:
    count = int(run_cfg.get("count", 10))
    runs = int(run_cfg.get("runs", 5))
    parallelism = int(parallelism_override or run_cfg.get("parallelism", 1))
    timeout = int(run_cfg.get("timeout_seconds", 7200))
    force = bool(run_cfg.get("force", False))
    batch = bool(run_cfg.get("batch", False))
    local = bool(run_cfg.get("local", False))
    run_disabled = bool(run_cfg.get("run_disabled", False))
    max_n_algorithms = int(run_cfg.get("max_n_algorithms", -1))
    log_dir = run_cfg.get("log_dir", "benchmarks/remote_logs")
    log_name = f"{dataset}{'-' + algorithm if algorithm else ''}.log"
    remote_log = f"{remote_checkout.rstrip('/')}/{log_dir.strip('/')}/{log_name}"
    remote_exit = f"{remote_log}.exit"
    remote_heartbeat = f"{remote_log}.heartbeat"
    lease_seconds = int(run_cfg.get("heartbeat_timeout_seconds", 180))
    args = [
        "python3",
        "run.py",
        "--dataset", dataset,
        "--count", str(count),
        "--runs", str(runs),
        "--parallelism", str(parallelism),
        "--timeout", str(timeout),
        "--max-n-algorithms", str(max_n_algorithms),
    ]
    if algorithm:
        args.extend(["--algorithm", algorithm])
    if force:
        args.append("--force")
    if batch:
        args.append("--batch")
    if local:
        args.append("--local")
    if run_disabled:
        args.append("--run-disabled")
    cmd = shell_join(args)
    wrapped_cmd = cmd + f'; code=$?; printf "%s" "$code" > {shlex.quote(remote_exit)}; exit "$code"'
    script = f"""
set -euo pipefail
cd {remote_quote(remote_checkout)}
mkdir -p {remote_quote(f"{remote_checkout.rstrip('/')}/{log_dir.strip('/')}")}
rm -f {remote_quote(remote_exit)}
touch {remote_quote(remote_heartbeat)}
nohup setsid bash -lc {remote_quote(wrapped_cmd)} > {remote_quote(remote_log)} 2>&1 &
job_pid=$!
nohup bash -lc '
while kill -0 -'"$job_pid"' 2>/dev/null; do
  now=$(date +%s)
  hb=$(stat -c %Y {remote_quote(remote_heartbeat)} 2>/dev/null || echo 0)
  if [ $((now - hb)) -gt {lease_seconds} ]; then
    kill -- -'"$job_pid"' 2>/dev/null || true
    sleep 2
    kill -9 -- -'"$job_pid"' 2>/dev/null || true
    exit 0
  fi
  sleep 15
done
' >/dev/null 2>&1 &
echo "$job_pid"
"""
    proc = run_remote(target, script, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
    pid = int(proc.stdout.strip())
    return pid, remote_log, remote_exit, remote_heartbeat


def monitor_dataset_run(
    target: RemoteTarget,
    remote_checkout: str,
    dataset: str,
    algorithm: str | None,
    count: int,
    pid: int,
    remote_log: str,
    remote_exit: str,
    remote_heartbeat: str,
    expected_results: int,
    monitor_cfg: dict[str, Any],
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> float:
    poll_seconds = int(monitor_cfg.get("poll_seconds", 60))
    stall_minutes = int(monitor_cfg.get("stall_minutes", 60))
    min_free_disk_gb = float(monitor_cfg.get("min_free_disk_gb", 10))
    stop_on_stall = bool(monitor_cfg.get("stop_on_stall", False))
    stop_on_low_disk = bool(monitor_cfg.get("stop_on_low_disk", True))
    max_wall_seconds = monitor_cfg.get("max_wall_seconds")
    label = f"{dataset}/{algorithm or 'all'}"
    last_progress_at = time.time()
    last_file_count = -1
    last_log_size = -1
    started_at = time.time()
    info(f"==> Monitor {label} (pid={pid}, expected_results={expected_results})")

    while True:
        touch_remote_file(target, remote_heartbeat, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
        result_dir = f"results/{dataset}/{count}"
        if algorithm:
            result_dir += f"/{algorithm}"
        remote_status_script = """
set +e
cd %(checkout)s
result_dir=%(result_dir)s
file_count=$(find "$result_dir" -type f -name '*.hdf5' 2>/dev/null | wc -l | tr -d ' ')
log_size=$(wc -c < %(remote_log)s 2>/dev/null | tr -d ' ' || echo 0)
free_gb=$(df -BG . | awk 'NR==2{gsub(/G/, "", $4); print $4}')
alive=0
if kill -0 %(pid)s 2>/dev/null; then alive=1; fi
ps_line=$(ps -p %(pid)s -o pid=,etime=,%%cpu=,%%mem=,rss=,stat=,cmd= 2>/dev/null || true)
docker_ps_b64=$(docker ps --format '{{.Names}}|{{.Status}}' 2>/dev/null | head -20 | base64 | tr -d '\\n')
tail_b64=$(tail -n 20 %(remote_log)s 2>/dev/null | base64 | tr -d '\\n')
exit_code=$(cat %(remote_exit)s 2>/dev/null || true)
export FILE_COUNT="$file_count" LOG_SIZE="$log_size" FREE_GB="$free_gb" ALIVE="$alive"
export PS_LINE="$ps_line" DOCKER_PS_B64="$docker_ps_b64" TAIL_B64="$tail_b64" EXIT_CODE="$exit_code"
python3 - <<'PY'
import base64
import json
import os
payload = {
    "alive": int(os.environ["ALIVE"]),
    "file_count": int(os.environ["FILE_COUNT"] or 0),
    "log_size": int(os.environ["LOG_SIZE"] or 0),
    "free_gb": float(os.environ["FREE_GB"] or 0),
    "ps_line": os.environ.get("PS_LINE", ""),
    "docker_ps": base64.b64decode(os.environ.get("DOCKER_PS_B64", "") or b"").decode("utf-8", "replace"),
    "tail": base64.b64decode(os.environ.get("TAIL_B64", "") or b"").decode("utf-8", "replace"),
    "exit_code": os.environ.get("EXIT_CODE", "").strip(),
}
print(json.dumps(payload))
PY
""" % {
            "checkout": remote_quote(remote_checkout),
            "result_dir": remote_quote(result_dir),
            "remote_log": remote_quote(remote_log),
            "remote_exit": remote_quote(remote_exit),
            "pid": pid,
        }
        proc = run_remote(
            target,
            remote_status_script,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        status = json.loads(proc.stdout)
        progressed = status["file_count"] > last_file_count or status["log_size"] > last_log_size
        if progressed:
            last_progress_at = time.time()
            last_file_count = status["file_count"]
            last_log_size = status["log_size"]

        print(
            f"[{label}] results={status['file_count']}/{expected_results} "
            f"log_bytes={status['log_size']} free_gb={status['free_gb']} alive={status['alive']}"
        )
        if status["ps_line"]:
            print(status["ps_line"])
        if status["docker_ps"]:
            print(status["docker_ps"])
        if status["tail"]:
            print(status["tail"])

        if status["free_gb"] < min_free_disk_gb:
            msg = f"free disk {status['free_gb']} GB below threshold {min_free_disk_gb} GB"
            if stop_on_low_disk:
                terminate_remote_job(target, pid, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
                raise RuntimeError(msg)
            print(f"warning: {msg}")

        if not status["alive"]:
            exit_code = status.get("exit_code")
            if exit_code not in ("", "0"):
                raise RuntimeError(f"dataset {dataset} exited with code {exit_code}")
            return time.time() - started_at

        stalled_for = (time.time() - last_progress_at) / 60.0
        if stalled_for >= stall_minutes:
            msg = f"no new results/log growth for {stalled_for:.1f} minutes on {label}"
            if stop_on_stall:
                terminate_remote_job(target, pid, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
                raise RuntimeError(msg)
            print(f"warning: {msg}")

        elapsed = time.time() - started_at
        if max_wall_seconds is not None and elapsed > float(max_wall_seconds):
            terminate_remote_job(target, pid, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
            raise RuntimeError(
                f"dataset {dataset} exceeded wall-clock budget: {elapsed:.1f}s > {float(max_wall_seconds):.1f}s"
            )

        time.sleep(poll_seconds)


def run_algorithm_job(
    target: RemoteTarget,
    remote_checkout: str,
    dataset: str,
    algorithm: str | None,
    count: int,
    run_cfg: dict[str, Any],
    monitor_cfg: dict[str, Any],
    *,
    parallelism_override: int | None = None,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> RunResult:
    expected = estimate_expected_results(
        target,
        remote_checkout,
        dataset,
        count,
        algorithm,
        bool(run_cfg.get("run_disabled", False)),
    )
    pid, remote_log, remote_exit, remote_heartbeat = launch_dataset_run(
        target,
        remote_checkout,
        dataset,
        run_cfg,
        algorithm,
        parallelism_override=parallelism_override,
        sudo=sudo,
        sudo_pythonpath=sudo_pythonpath,
    )
    try:
        elapsed = monitor_dataset_run(
            target,
            remote_checkout,
            dataset,
            algorithm,
            count,
            pid,
            remote_log,
            remote_exit,
            remote_heartbeat,
            expected,
            monitor_cfg,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
    except Exception as exc:
        terminate_remote_job(target, pid, sudo=sudo, sudo_pythonpath=sudo_pythonpath)
        return RunResult(
            dataset=dataset,
            algorithm=algorithm,
            elapsed=None,
            expected_results=expected,
            status="failed",
            error=str(exc),
        )
    return RunResult(
        dataset=dataset,
        algorithm=algorithm,
        elapsed=elapsed,
        expected_results=expected,
        status="completed",
    )


def export_artifacts(
    target: RemoteTarget,
    remote_checkout: str,
    datasets: list[str],
    run_cfg: dict[str, Any],
    export_cfg: dict[str, Any],
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> str:
    count = int(run_cfg.get("count", 10))
    artifact_root = export_cfg.get("remote_artifact_dir", f"artifacts/{utc_stamp()}")
    website_dir = f"{artifact_root.rstrip('/')}/website"
    csv_path = f"{artifact_root.rstrip('/')}/results.csv"
    summary_path = f"{artifact_root.rstrip('/')}/summary.json"
    plot_datasets = datasets if export_cfg.get("plot_each_dataset", True) else []
    info("==> Export CSV and render website")
    lines = [
        "set -euo pipefail",
        f"cd {remote_quote(remote_checkout)}",
        f"mkdir -p {remote_quote(website_dir)}",
        f"python3 data_export.py --output {remote_quote(csv_path)}",
        f"python3 create_website.py --outputdir {remote_quote(website_dir)} --scatter",
        f"python3 - <<'PY'\n"
        "import json, os\n"
        f"artifact_root = {artifact_root!r}\n"
        f"summary = {{'datasets': {datasets!r}, 'count': {count}, 'artifact_root': artifact_root}}\n"
        f"open({summary_path!r}, 'w').write(json.dumps(summary, indent=2))\n"
        "PY",
    ]
    for dataset in plot_datasets:
        lines.append(
            f"python3 plot.py --dataset {remote_quote(dataset)} >/dev/null 2>&1 || true"
        )
    run_remote(target, "\n".join(lines), sudo=sudo, sudo_pythonpath=sudo_pythonpath)
    return artifact_root


def sync_artifacts(
    target: RemoteTarget,
    remote_checkout: str,
    remote_artifact_root: str,
    local_output_dir: Path,
    sync_cfg: dict[str, Any],
) -> None:
    if sync_cfg.get("skip_download"):
        return
    info("==> Sync artifacts back locally")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    for rel in sync_cfg.get("paths", ["results", "figures"]):
        rsync_from_remote(
            target,
            f"{remote_checkout.rstrip('/')}/{rel.strip('/')}/",
            local_output_dir / rel,
            optional=True,
        )
    rsync_from_remote(
        target,
        f"{remote_checkout.rstrip('/')}/{remote_artifact_root.strip('/')}/",
        local_output_dir / "artifacts",
        optional=True,
    )


def run_dataset_pipeline(
    target: RemoteTarget,
    remote_checkout: str,
    dataset: str,
    algorithms: list[str | None],
    run_cfg: dict[str, Any],
    monitor_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    orchestrator_cfg: dict[str, Any],
    *,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> list[RunResult]:
    count = int(run_cfg.get("count", 10))
    baseline_algorithm = policy_cfg.get("baseline_algorithm")
    baseline_multiplier = float(policy_cfg.get("baseline_multiplier", 1.15))
    skip_rest_if_baseline_fails = bool(policy_cfg.get("skip_rest_if_baseline_fails", True))
    baseline_parallelism = int(orchestrator_cfg.get("baseline_parallelism", run_cfg.get("parallelism", 1)))
    followup_parallelism = int(orchestrator_cfg.get("followup_parallelism", 1))
    max_followups = int(orchestrator_cfg.get("max_concurrent_followups", 1))

    results: list[RunResult] = []
    followup_algorithms = list(algorithms)

    if baseline_algorithm and baseline_algorithm in followup_algorithms:
        baseline_monitor_cfg = dict(monitor_cfg)
        baseline_result = run_algorithm_job(
            target,
            remote_checkout,
            dataset,
            baseline_algorithm,
            count,
            run_cfg,
            baseline_monitor_cfg,
            parallelism_override=baseline_parallelism,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        results.append(baseline_result)
        followup_algorithms = [a for a in followup_algorithms if a != baseline_algorithm]

        if baseline_result.status != "completed":
            if skip_rest_if_baseline_fails:
                info(f"==> Baseline {baseline_algorithm} failed on {dataset}; skipping remaining algorithms")
                return results
            return results

        assert baseline_result.elapsed is not None
        baseline_elapsed = baseline_result.elapsed
        info(f"==> Baseline {baseline_algorithm} completed on {dataset} in {baseline_elapsed:.1f}s")
    else:
        baseline_elapsed = None

    if not followup_algorithms:
        return results

    if baseline_elapsed is not None:
        budget = baseline_elapsed * baseline_multiplier
        info(
            f"==> Launching up to {max_followups} follow-up algorithms for {dataset} "
            f"with {followup_parallelism} inner workers each and {budget:.1f}s budget"
        )
    else:
        info(
            f"==> Launching up to {max_followups} algorithms for {dataset} "
            f"with {followup_parallelism} inner workers each"
        )

    def worker(algorithm: str | None) -> RunResult:
        effective_monitor_cfg = dict(monitor_cfg)
        if baseline_elapsed is not None and algorithm is not None:
            effective_monitor_cfg["max_wall_seconds"] = baseline_elapsed * baseline_multiplier
            info(
                f"==> Budget for {algorithm} on {dataset}: "
                f"{effective_monitor_cfg['max_wall_seconds']:.1f}s "
                f"(baseline {baseline_elapsed:.1f}s * {baseline_multiplier:.2f})"
            )
        return run_algorithm_job(
            target,
            remote_checkout,
            dataset,
            algorithm,
            count,
            run_cfg,
            effective_monitor_cfg,
            parallelism_override=followup_parallelism,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_followups) as pool:
        future_map = {pool.submit(worker, algorithm): algorithm for algorithm in followup_algorithms}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            label = result.algorithm or "all"
            if result.status == "completed":
                info(f"==> Completed {label} on {dataset} in {result.elapsed:.1f}s")
            else:
                info(f"==> Failed {label} on {dataset}: {result.error}")

    return results


def run_benchmark_pipeline(
    target: RemoteTarget,
    remote_checkout: str,
    datasets: list[str],
    algorithms: list[str | None],
    run_cfg: dict[str, Any],
    monitor_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    orchestrator_cfg: dict[str, Any],
    build_cfg: dict[str, Any],
    built_dirs: set[str],
    *,
    remote_cpus: int,
    sudo: bool = False,
    sudo_pythonpath: str | None = None,
) -> list[RunResult]:
    baseline_algorithm = policy_cfg.get("baseline_algorithm")
    baseline_multiplier = float(policy_cfg.get("baseline_multiplier", 1.15))
    skip_rest_if_baseline_fails = bool(policy_cfg.get("skip_rest_if_baseline_fails", True))
    baseline_parallelism = int(orchestrator_cfg.get("baseline_parallelism", run_cfg.get("parallelism", 1)))
    followup_parallelism = int(orchestrator_cfg.get("followup_parallelism", 1))
    max_concurrent_baselines = int(orchestrator_cfg.get("max_concurrent_baselines", 1))
    max_concurrent_followups = int(orchestrator_cfg.get("max_concurrent_followups", 1))
    total_core_budget = int(orchestrator_cfg.get("total_core_budget", remote_cpus))
    count = int(run_cfg.get("count", 10))

    datasets_pending = list(datasets)
    pending_followups: dict[str, list[str | None]] = {}
    results: list[RunResult] = []
    active_future_info: dict[concurrent.futures.Future[RunResult], tuple[str, str, int]] = {}
    active_baselines = 0
    active_followups = 0
    used_cores = 0

    def can_launch(job_kind: str, cores: int) -> bool:
        if used_cores + cores > total_core_budget:
            return False
        if job_kind == "baseline":
            return active_baselines < max_concurrent_baselines
        return active_followups < max_concurrent_followups

    def launch_baseline(pool: concurrent.futures.ThreadPoolExecutor, dataset: str) -> bool:
        nonlocal active_baselines, used_cores
        if not can_launch("baseline", baseline_parallelism):
            return False
        ensure_algorithm_built(
            target,
            remote_checkout,
            baseline_algorithm,
            build_cfg,
            built_dirs,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        future = pool.submit(
            run_algorithm_job,
            target,
            remote_checkout,
            dataset,
            baseline_algorithm,
            count,
            run_cfg,
            dict(monitor_cfg),
            parallelism_override=baseline_parallelism,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        active_future_info[future] = (dataset, "baseline", baseline_parallelism)
        active_baselines += 1
        used_cores += baseline_parallelism
        info(f"==> Launched baseline {baseline_algorithm} on {dataset} with {baseline_parallelism} workers")
        return True

    def launch_followup(
        pool: concurrent.futures.ThreadPoolExecutor,
        dataset: str,
        algorithm: str | None,
        baseline_elapsed: float,
    ) -> bool:
        nonlocal active_followups, used_cores
        if not can_launch("followup", followup_parallelism):
            return False
        ensure_algorithm_built(
            target,
            remote_checkout,
            algorithm,
            build_cfg,
            built_dirs,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        effective_monitor_cfg = dict(monitor_cfg)
        if algorithm is not None:
            effective_monitor_cfg["max_wall_seconds"] = baseline_elapsed * baseline_multiplier
            info(
                f"==> Budget for {algorithm} on {dataset}: "
                f"{effective_monitor_cfg['max_wall_seconds']:.1f}s "
                f"(baseline {baseline_elapsed:.1f}s * {baseline_multiplier:.2f})"
            )
        future = pool.submit(
            run_algorithm_job,
            target,
            remote_checkout,
            dataset,
            algorithm,
            count,
            run_cfg,
            effective_monitor_cfg,
            parallelism_override=followup_parallelism,
            sudo=sudo,
            sudo_pythonpath=sudo_pythonpath,
        )
        active_future_info[future] = (dataset, "followup", followup_parallelism)
        active_followups += 1
        used_cores += followup_parallelism
        info(f"==> Launched {algorithm or 'all'} on {dataset} with {followup_parallelism} workers")
        return True

    def drain_launches(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        made_progress = True
        while made_progress:
            made_progress = False
            while datasets_pending and baseline_algorithm and can_launch("baseline", baseline_parallelism):
                dataset = datasets_pending.pop(0)
                if launch_baseline(pool, dataset):
                    made_progress = True
                else:
                    datasets_pending.insert(0, dataset)
                    break
            ready_datasets = [
                dataset
                for dataset, algs in pending_followups.items()
                if algs and dataset in baseline_elapsed_by_dataset
            ]
            for dataset in ready_datasets:
                while pending_followups[dataset] and can_launch("followup", followup_parallelism):
                    algorithm = pending_followups[dataset].pop(0)
                    if launch_followup(pool, dataset, algorithm, baseline_elapsed_by_dataset[dataset]):
                        made_progress = True
                    else:
                        pending_followups[dataset].insert(0, algorithm)
                        break

    baseline_elapsed_by_dataset: dict[str, float] = {}
    if not baseline_algorithm:
        for dataset in datasets:
            pending_followups[dataset] = list(algorithms)
            baseline_elapsed_by_dataset[dataset] = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(datasets) * max(1, len(algorithms))) as pool:
        if baseline_algorithm:
            for dataset in datasets:
                pending_followups[dataset] = [a for a in algorithms if a != baseline_algorithm]
        drain_launches(pool)

        while active_future_info or datasets_pending or any(pending_followups.values()):
            if not active_future_info:
                drain_launches(pool)
                if not active_future_info:
                    raise RuntimeError("scheduler deadlock: no active jobs and pending work remains")

            done, _ = concurrent.futures.wait(
                list(active_future_info.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                dataset, job_kind, cores = active_future_info.pop(future)
                used_cores -= cores
                if job_kind == "baseline":
                    active_baselines -= 1
                else:
                    active_followups -= 1

                result = future.result()
                results.append(result)
                label = result.algorithm or "all"
                if result.status == "completed":
                    info(f"==> Completed {label} on {dataset} in {result.elapsed:.1f}s")
                else:
                    info(f"==> Failed {label} on {dataset}: {result.error}")

                if job_kind == "baseline":
                    if result.status == "completed" and result.elapsed is not None:
                        baseline_elapsed_by_dataset[dataset] = result.elapsed
                        info(f"==> Baseline {baseline_algorithm} completed on {dataset} in {result.elapsed:.1f}s")
                    elif skip_rest_if_baseline_fails:
                        pending_followups[dataset] = []
                        info(f"==> Baseline {baseline_algorithm} failed on {dataset}; skipping remaining algorithms")
                    else:
                        pending_followups[dataset] = []

            drain_launches(pool)

    return results


def run_workflow(config: dict[str, Any], *, dry_run: bool) -> None:
    target_cfg = require(config, "remote")
    repo_cfg = require(config, "ann_benchmarks_repo")
    run_cfg = config.get("run", {})
    monitor_cfg = config.get("monitor", {})
    build_cfg = config.get("build", {})
    export_cfg = config.get("export", {})
    sync_cfg = config.get("download", {})
    sync_items = config.get("sync", [])
    policy_cfg = config.get("policy", {})
    orchestrator_cfg = dict(config.get("orchestrator", {}))
    parameter_cfg = config.get("parameter_pruning", {})
    remote_root = config.get("remote_root", "~/annb-runs")
    local_output_cfg = Path(config.get("local_output_root", DEFAULT_LOCAL_OUTPUT_ROOT)).expanduser()
    local_output_root = local_output_cfg if local_output_cfg.is_absolute() else (REPO / local_output_cfg).resolve()
    local_output_dir = local_output_root / utc_stamp()

    target = RemoteTarget(
        host=require(target_cfg, "host"),
        user=target_cfg.get("user"),
        port=target_cfg.get("port"),
        identity_file=str(Path(target_cfg["identity_file"]).expanduser()) if target_cfg.get("identity_file") else None,
        extra_ssh_args=target_cfg.get("extra_ssh_args"),
    )
    docker_sudo = bool(target_cfg.get("docker_sudo", False))

    if dry_run:
        print(json.dumps(config, indent=2))
        return

    remote_preflight(target)
    remote_root = resolve_remote_root(target, remote_root)
    sudo_pythonpath = resolve_remote_user_site(target) if docker_sudo else None
    remote_cpus = resolve_remote_cpu_count(target)
    followup_parallelism = int(orchestrator_cfg.get("followup_parallelism", 1))
    if "max_concurrent_followups" not in orchestrator_cfg:
        orchestrator_cfg["max_concurrent_followups"] = max(1, remote_cpus // max(1, followup_parallelism))
    if "baseline_parallelism" not in orchestrator_cfg:
        orchestrator_cfg["baseline_parallelism"] = int(run_cfg.get("parallelism", 1))
    info(
        f"==> Orchestrator plan: remote_cpus={remote_cpus}, "
        f"baseline_parallelism={orchestrator_cfg['baseline_parallelism']}, "
        f"max_concurrent_baselines={orchestrator_cfg.get('max_concurrent_baselines', 1)}, "
        f"followup_parallelism={followup_parallelism}, "
        f"max_concurrent_followups={orchestrator_cfg['max_concurrent_followups']}, "
        f"total_core_budget={orchestrator_cfg.get('total_core_budget', remote_cpus)}"
    )
    remote_checkout = prepare_checkout(target, repo_cfg, remote_root)
    cleanup_remote_state(
        target,
        remote_checkout,
        clear_outputs=True,
        sudo=docker_sudo,
        sudo_pythonpath=sudo_pythonpath,
    )
    sync_overlays(target, remote_checkout, sync_items)
    datasets = resolve_dataset_list(target, remote_checkout, config.get("datasets"))
    algorithms = run_cfg.get("algorithms") or [None]
    baseline_algorithm = policy_cfg.get("baseline_algorithm")
    if baseline_algorithm and baseline_algorithm in algorithms:
        algorithms = [baseline_algorithm] + [a for a in algorithms if a != baseline_algorithm]
    selected_build_dirs = {
        build_dir
        for build_dir in (resolve_build_dir(algorithm, build_cfg) for algorithm in algorithms)
        if build_dir
    }
    apply_parameter_pruning(
        target,
        remote_checkout,
        parameter_cfg,
        allowed_algorithm_dirs=selected_build_dirs,
    )
    built_dirs: set[str] = set()
    if baseline_algorithm:
        ensure_algorithm_built(
            target,
            remote_checkout,
            baseline_algorithm,
            build_cfg,
            built_dirs,
            sudo=docker_sudo,
            sudo_pythonpath=sudo_pythonpath,
        )

    run_benchmark_pipeline(
        target,
        remote_checkout,
        datasets,
        algorithms,
        run_cfg,
        monitor_cfg,
        policy_cfg,
        orchestrator_cfg,
        build_cfg,
        built_dirs,
        remote_cpus=remote_cpus,
        sudo=docker_sudo,
        sudo_pythonpath=sudo_pythonpath,
    )

    remote_artifact_root = export_artifacts(
        target,
        remote_checkout,
        datasets,
        run_cfg,
        export_cfg,
        sudo=docker_sudo,
        sudo_pythonpath=sudo_pythonpath,
    )
    sync_artifacts(target, remote_checkout, remote_artifact_root, local_output_dir, sync_cfg)
    info(f"==> Local artifacts: {local_output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ANN-Benchmarks remotely over SSH with JSON config and monitoring."
    )
    parser.add_argument("config", help="Path to the JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the config only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(Path(args.config))
    run_workflow(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
