from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from pr import launch_allele_workers


class LauncherTests(unittest.TestCase):
    def test_pending_manifest_does_not_spawn_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            manifest = {
                "ready_for_worker": False,
                "status": "pending_inputs",
                "jobs": [{"job_id": "job-%02d" % index, "job_config": "jobs/job-%02d.json" % index}
                         for index in range(15)],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(launch_allele_workers.subprocess, "Popen") as popen:
                result = launch_allele_workers.launch(root)
            popen.assert_not_called()
            self.assertEqual(result["status"], "pending_inputs")
            self.assertFalse(result["fit_called"])
            self.assertEqual(result["started_jobs"], [])

    def test_memory_wait_resumes_without_second_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "active").mkdir()
            (root / "jobs").mkdir()
            jobs = []
            for index in range(15):
                job_id = "job-%02d" % index
                job_path = root / "jobs" / job_id / "job.json"
                job_path.parent.mkdir()
                job_path.write_text("{}", encoding="utf-8")
                jobs.append({"job_id": job_id, "job_config": "jobs/%s/job.json" % job_id})
            (root / "manifest.json").write_text(
                json.dumps({"ready_for_worker": True, "jobs": jobs}), encoding="utf-8")
            memory_calls = {"count": 0}
            sleep_calls = {"count": 0}

            def available_memory():
                memory_calls["count"] += 1
                return (
                    launch_allele_workers.MIN_AVAILABLE_BYTES - 1
                    if memory_calls["count"] == 1
                    else launch_allele_workers.MIN_AVAILABLE_BYTES + 1
                )

            class CompletedProcess:
                def __init__(self, job_id):
                    self.job_id = job_id

                def poll(self):
                    return 0

                def wait(self):
                    (root / "jobs" / self.job_id / "status.json").write_text(
                        json.dumps({"status": "budget_not_converged", "fit_called": True}),
                        encoding="utf-8")
                    return 0

            def popen(command, **kwargs):
                return CompletedProcess(Path(command[-1]).parent.name)

            with mock.patch.object(
                    launch_allele_workers, "_available_memory_bytes",
                    side_effect=available_memory), \
                 mock.patch.object(
                     launch_allele_workers.time, "sleep",
                     side_effect=lambda seconds: sleep_calls.__setitem__(
                         "count", sleep_calls["count"] + 1)), \
                 mock.patch.object(
                     launch_allele_workers.subprocess, "Popen", side_effect=popen):
                result = launch_allele_workers.launch(root, max_workers=1)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["started_jobs"]), 15)
            self.assertEqual(result["deferred_jobs"], [])
            self.assertEqual(len(result["waiting_events"]), 1)
            self.assertEqual(sleep_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
