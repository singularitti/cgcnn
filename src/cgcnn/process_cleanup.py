from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ProcessInfo",
    "TimedProcessInfo",
    "cleanup_orphaned_python_workers",
    "cleanup_stale_torch_shm_managers",
    "ensure_orphaned_worker_reaper",
    "find_orphaned_python_workers",
    "find_torch_shm_managers",
    "list_processes",
    "list_timed_processes",
]

_PYTHON_MULTIPROCESS_MARKERS = (
    "from multiprocessing.spawn import spawn_main",
    "from multiprocessing.resource_tracker import main",
)
_REAPER_THREADS: dict[str, tuple[threading.Thread, threading.Event]] = {}
_REAPER_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str


@dataclass(frozen=True)
class TimedProcessInfo:
    pid: int
    ppid: int
    elapsed_seconds: int
    command: str


def list_processes() -> list[ProcessInfo]:
    result = subprocess.run(
        ["ps", "-Ao", "pid,ppid,command"],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[ProcessInfo] = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) != 3:
            continue
        pid_str, ppid_str, command = parts
        try:
            processes.append(ProcessInfo(pid=int(pid_str), ppid=int(ppid_str), command=command))
        except ValueError:
            continue
    return processes


def _parse_elapsed_seconds(etime: str) -> int:
    days = 0
    remainder = etime
    if "-" in etime:
        day_part, remainder = etime.split("-", 1)
        days = int(day_part)
    time_parts = [int(part) for part in remainder.split(":")]
    if len(time_parts) == 2:
        hours = 0
        minutes, seconds = time_parts
    elif len(time_parts) == 3:
        hours, minutes, seconds = time_parts
    else:
        raise ValueError(f"Unsupported elapsed time format: {etime}")
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def list_timed_processes() -> list[TimedProcessInfo]:
    result = subprocess.run(
        ["ps", "-Ao", "pid,ppid,etime,command"],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: list[TimedProcessInfo] = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 3)
        if len(parts) != 4:
            continue
        pid_str, ppid_str, elapsed_str, command = parts
        try:
            processes.append(
                TimedProcessInfo(
                    pid=int(pid_str),
                    ppid=int(ppid_str),
                    elapsed_seconds=_parse_elapsed_seconds(elapsed_str),
                    command=command,
                )
            )
        except ValueError:
            continue
    return processes


def _is_matching_python_worker(process: ProcessInfo, python_executable: str) -> bool:
    if process.ppid != 1:
        return False
    if process.pid == os.getpid():
        return False
    normalized_python = str(Path(python_executable).resolve())
    if not process.command.startswith(normalized_python):
        return False
    return any(marker in process.command for marker in _PYTHON_MULTIPROCESS_MARKERS)


def find_orphaned_python_workers(python_executable: str) -> list[ProcessInfo]:
    return [
        process
        for process in list_processes()
        if _is_matching_python_worker(process, python_executable)
    ]


def find_torch_shm_managers() -> list[TimedProcessInfo]:
    return [
        process
        for process in list_timed_processes()
        if process.command == "torch_shm_manager"
    ]


def cleanup_orphaned_python_workers(python_executable: str) -> list[int]:
    cleaned_pids: list[int] = []
    for process in find_orphaned_python_workers(python_executable):
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        cleaned_pids.append(process.pid)
    return cleaned_pids


def cleanup_stale_torch_shm_managers(
    retain_count: int,
    min_age_seconds: int = 120,
) -> list[int]:
    if retain_count < 0:
        raise ValueError("retain_count must be non-negative")
    managers = sorted(
        find_torch_shm_managers(),
        key=lambda process: (process.elapsed_seconds, process.pid),
    )
    keep_pids = {process.pid for process in managers[:retain_count]}
    cleaned_pids: list[int] = []
    for process in managers[retain_count:]:
        if process.elapsed_seconds < min_age_seconds:
            continue
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        cleaned_pids.append(process.pid)
    return cleaned_pids


def _reap_orphaned_workers_forever(
    python_executable: str,
    stop_event: threading.Event,
    interval_seconds: float,
    torch_shm_retain_count: int,
    torch_shm_min_age_seconds: int,
) -> None:
    while not stop_event.wait(interval_seconds):
        try:
            cleanup_orphaned_python_workers(python_executable)
        except Exception:
            pass
        if torch_shm_retain_count > 0:
            try:
                cleanup_stale_torch_shm_managers(
                    retain_count=torch_shm_retain_count,
                    min_age_seconds=torch_shm_min_age_seconds,
                )
            except Exception:
                pass


def ensure_orphaned_worker_reaper(
    python_executable: str,
    interval_seconds: float = 60.0,
    torch_shm_retain_count: int = 0,
    torch_shm_min_age_seconds: int = 120,
) -> None:
    normalized_python = str(Path(python_executable).resolve())
    with _REAPER_LOCK:
        existing = _REAPER_THREADS.get(normalized_python)
        if existing and existing[0].is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_reap_orphaned_workers_forever,
            args=(
                normalized_python,
                stop_event,
                interval_seconds,
                torch_shm_retain_count,
                torch_shm_min_age_seconds,
            ),
            name="cgcnn-orphaned-worker-reaper",
            daemon=True,
        )
        _REAPER_THREADS[normalized_python] = (thread, stop_event)
        thread.start()
