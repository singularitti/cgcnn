from __future__ import annotations

import signal
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cgcnn.process_cleanup import (
    ProcessInfo,
    TimedProcessInfo,
    cleanup_orphaned_python_workers,
    cleanup_stale_torch_shm_managers,
    ensure_orphaned_worker_reaper,
    find_orphaned_python_workers,
)


class ProcessCleanupTests(unittest.TestCase):
    def test_find_orphaned_python_workers_matches_only_expected_processes(self):
        python_executable = sys.executable
        resolved_python = str(Path(python_executable).resolve())
        processes = [
            ProcessInfo(
                pid=101,
                ppid=1,
                command=(
                    f"{resolved_python} -c "
                    "'from multiprocessing.spawn import spawn_main; spawn_main()'"
                ),
            ),
            ProcessInfo(
                pid=102,
                ppid=1,
                command=(
                    f"{resolved_python} -c "
                    "'from multiprocessing.resource_tracker import main; main(5)'"
                ),
            ),
            ProcessInfo(
                pid=103,
                ppid=42929,
                command=(
                    f"{resolved_python} -c "
                    "'from multiprocessing.spawn import spawn_main; spawn_main()'"
                ),
            ),
            ProcessInfo(
                pid=104,
                ppid=1,
                command="/usr/bin/python3 -c 'from multiprocessing.spawn import spawn_main'",
            ),
            ProcessInfo(pid=105, ppid=1, command=f"{resolved_python} train.py"),
        ]

        with patch("cgcnn.process_cleanup.list_processes", return_value=processes), patch(
            "cgcnn.process_cleanup.os.getpid", return_value=999
        ):
            orphaned = find_orphaned_python_workers(python_executable)

        self.assertEqual([process.pid for process in orphaned], [101, 102])

    def test_cleanup_orphaned_python_workers_sends_sigterm(self):
        orphaned = [
            ProcessInfo(
                pid=201,
                ppid=1,
                command=(
                    "/tmp/project/.venv/bin/python -c "
                    "'from multiprocessing.spawn import spawn_main; spawn_main()'"
                ),
            ),
            ProcessInfo(
                pid=202,
                ppid=1,
                command=(
                    "/tmp/project/.venv/bin/python -c "
                    "'from multiprocessing.resource_tracker import main; main(5)'"
                ),
            ),
        ]

        with patch(
            "cgcnn.process_cleanup.find_orphaned_python_workers", return_value=orphaned
        ), patch("cgcnn.process_cleanup.os.kill") as mock_kill:
            cleaned = cleanup_orphaned_python_workers(sys.executable)

        self.assertEqual(cleaned, [201, 202])
        self.assertEqual(
            mock_kill.call_args_list,
            [((201, signal.SIGTERM),), ((202, signal.SIGTERM),)],
        )

    def test_cleanup_stale_torch_shm_managers_keeps_newest_buffer(self):
        managers = [
            TimedProcessInfo(pid=301, ppid=1, elapsed_seconds=900, command="torch_shm_manager"),
            TimedProcessInfo(pid=302, ppid=1, elapsed_seconds=600, command="torch_shm_manager"),
            TimedProcessInfo(pid=303, ppid=1, elapsed_seconds=300, command="torch_shm_manager"),
            TimedProcessInfo(pid=304, ppid=1, elapsed_seconds=60, command="torch_shm_manager"),
        ]

        with patch(
            "cgcnn.process_cleanup.find_torch_shm_managers", return_value=managers
        ), patch("cgcnn.process_cleanup.os.kill") as mock_kill:
            cleaned = cleanup_stale_torch_shm_managers(
                retain_count=2,
                min_age_seconds=120,
            )

        self.assertEqual(cleaned, [302, 301])
        self.assertEqual(
            mock_kill.call_args_list,
            [((302, signal.SIGTERM),), ((301, signal.SIGTERM),)],
        )

    def test_ensure_orphaned_worker_reaper_starts_only_once_per_python(self):
        with patch.dict("cgcnn.process_cleanup._REAPER_THREADS", {}, clear=True), patch(
            "cgcnn.process_cleanup.threading.Thread"
        ) as mock_thread:
            thread_instance = mock_thread.return_value
            thread_instance.is_alive.return_value = True

            ensure_orphaned_worker_reaper("/tmp/project/.venv/bin/python", interval_seconds=1.0)
            ensure_orphaned_worker_reaper("/tmp/project/.venv/bin/python", interval_seconds=1.0)

        self.assertEqual(mock_thread.call_count, 1)
        self.assertEqual(thread_instance.start.call_count, 1)


if __name__ == "__main__":
    unittest.main()
