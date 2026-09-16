import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pr import allele_experiment as experiment
from pr.contact_model import aggregate_from_arrays, sphere_forward
from pr.paired_run import PairedStart, save_paired_start


TEST_SEED = 20260915


def _one_mb_data():
    records = [
        (0, 0, 0, 1_000_000),
        (0, 1_000_000, 0, 2_000_000),
        (1, 0, 1, 1_000_000),
        (0, 0, 1, 0),
        (0, 1_000_000, 1, 1_000_000),
        (0, 2_000_000, 1, 2_000_000),
        (0, 0, 0, 0),
    ]
    ci, p1, cj, p2 = (np.asarray(part, dtype=np.int64) for part in zip(*records))
    return aggregate_from_arrays(
        ("chrA", "chrB"), (3_000_000, 3_000_000),
        ci, p1, cj, p2, 1_000_000,
    )


def _cohort_for(data):
    return {
        "schema": "test-cohort",
        "source_path": str(experiment.DEFAULT_SOURCE_PATH),
        "source_sha256": experiment.EXPECTED_SOURCE_SHA256,
        "bin_size_bp": experiment.BIN_SIZE,
        "raw_records": int(data.raw_records),
        "cis_records": int(data.raw_cis_offdiag + data.raw_same_bin),
        "same_bin_records": int(data.raw_same_bin),
        "cis_offdiag_records": int(data.raw_cis_offdiag),
        "inter_records": int(data.raw_inter),
        "n_loci": data.n_loci,
        "n_physical_beads": 2 * data.n_loci,
        "n_eligible_pairs": data.n_pairs,
        "n_zero_eligible_pairs": int((data.counts == 0).sum()),
        "exposure_mode": data.exposure_mode,
        "count_mode": data.count_mode,
        "phase_or_reference_read": False,
        "training_payload": "seven-column SNP-free pairs only",
    }


def _released_protocol(path):
    payload = {
        "schema": "test-post020-protocol",
        "status": "frozen",
        "fit": {
            "maxiter": 480,
            "maxfun": 1470,
            "maxls": 20,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "checkpoint_every": 20,
        },
        "design": {
            "bin_size_bp": 1_000_000,
            "job_count": 15,
            "variant_count": 5,
            "bundle_count": 3,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_starts(root, data):
    rng = np.random.default_rng(TEST_SEED)
    for index, bundle_id in enumerate(experiment.BUNDLE_IDS):
        bundle_root = root / bundle_id
        bundle_root.mkdir(parents=True)
        start = PairedStart(
            bundle_id,
            sphere_forward(rng.normal(0.0, 0.08, size=(2, data.n_loci, 3))),
        )
        save_paired_start(bundle_root / "x0_normalized.npz", start)


def _write_synthetic_gate(root, protocol_sha, **overrides):
    gate_root = root / "026-fixture" / "results"
    gate_root.mkdir(parents=True)
    terminal = gate_root / "terminal_manifest.json"
    evaluation = gate_root / "evaluation.json"
    terminal.write_text("{\"terminal\": true}\n", encoding="utf-8")
    evaluation.write_text("{\"evaluation\": true}\n", encoding="utf-8")
    payload = {
        "schema": "post020-synthetic-terminal-gate-v1",
        "status": "passed",
        "protocol_json_sha256": protocol_sha,
        "planned_fit_count": 20,
        "terminal_fit_count": 20,
        "finite_valid_endpoint_count": 20,
        "coordinates_hash_locked": True,
        "numerical_checks_passed": True,
        "evaluation_complete": True,
        "terminal_manifest_path": str(terminal),
        "terminal_manifest_sha256": hashlib.sha256(terminal.read_bytes()).hexdigest(),
        "evaluation_path": str(evaluation),
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
    }
    payload.update(overrides)
    gate = gate_root / "implementation_gate.json"
    gate.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def _all_keys(payload):
    keys = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(str(key).lower())
            keys.extend(_all_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.extend(_all_keys(value))
    return keys


class ProtocolTests(unittest.TestCase):
    def test_release_hash_and_frozen_contract_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            _payload, digest = _released_protocol(path)
            result = experiment.verify_protocol_release(path, digest, None)
            self.assertEqual(result["sha256"], digest)
            with self.assertRaisesRegex(experiment.ExperimentError, "SHA mismatch"):
                experiment.verify_protocol_release(path, "0" * 64, None)

    def test_synthetic_gate_rejects_prepare_only_nested_status_and_short_terminal_set(self):
        cases = (
            {"status": "prepared_no_optimizer"},
            {"status": "prepared_no_optimizer", "metadata": {"status": "passed"}},
            {"terminal_fit_count": 19},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            _payload, digest = _released_protocol(protocol)
            for index, overrides in enumerate(cases):
                case_root = root / ("case-%d" % index)
                gate = _write_synthetic_gate(case_root, digest, **overrides)
                result = experiment._synthetic_gate(gate, digest)
                self.assertEqual(result["status"], "invalid")

    def test_complete_synthetic_gate_requires_the_026_results_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            _payload, digest = _released_protocol(protocol)
            gate = _write_synthetic_gate(root, digest)
            result = experiment._synthetic_gate(gate, digest)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["terminal_fit_count"], 20)

    def test_draft_protocol_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            payload, _digest = _released_protocol(path)
            payload["status"] = "draft"
            path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(experiment.ExperimentError, "not released"):
                experiment.verify_protocol_release(path, digest, None)


class PrepareFinalizeTests(unittest.TestCase):
    def test_prepare_writes_complete_15_job_manifest_without_fit(self):
        data = _one_mb_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            _payload, digest = _released_protocol(protocol)
            starts = root / "starts"
            _write_starts(starts, data)
            gate = _write_synthetic_gate(root, digest)
            study_root = root / "027-test-study"
            with mock.patch.object(experiment, "_load_real_cohort",
                                   return_value=(data, _cohort_for(data))), \
                 mock.patch.object(experiment, "run_one_fit") as fit:
                prepared = experiment.prepare(
                    run_root=study_root,
                    release_sha256=digest,
                    protocol_json=protocol,
                    protocol_md=None,
                    synthetic_gate_path=gate,
                    starts_root=starts,
                )
            fit.assert_not_called()
            self.assertEqual(prepared["status"], "ready_for_worker")
            manifest = experiment.read_json(study_root / "manifest.json")
            self.assertTrue(manifest["ready_for_worker"])
            self.assertEqual(len(manifest["jobs"]), 15)
            self.assertEqual(manifest["study_number"], 27)
            self.assertFalse(manifest["fit_called"])
            self.assertEqual(len(experiment.read_json(study_root / "x0_manifest.json")["bundles"]), 3)
            for row in manifest["jobs"]:
                self.assertFalse(row["fit_called"])
                job = experiment.read_json(study_root / row["job_config"])
                self.assertNotIn("reference", _all_keys(job))
                self.assertNotIn("truth", _all_keys(job))
                self.assertNotIn("phase", _all_keys(job))

    def test_complete_x0_without_synthetic_gate_stays_pending(self):
        data = _one_mb_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            _payload, digest = _released_protocol(protocol)
            starts = root / "starts"
            _write_starts(starts, data)
            with mock.patch.object(experiment, "_load_real_cohort",
                                   return_value=(data, _cohort_for(data))):
                prepared = experiment.prepare(
                    run_root=root / "027-no-gate-study",
                    release_sha256=digest,
                    protocol_json=protocol,
                    protocol_md=None,
                    starts_root=starts,
                )
            self.assertEqual(prepared["status"], "pending_synthetic_gate")
            manifest = experiment.read_json(Path(prepared["manifest"]))
            job_path = Path(prepared["study_root"]) / manifest["jobs"][0]["job_config"]
            self.assertEqual(experiment.worker(job_path)["status"], "pending_synthetic_gate")

    def test_missing_x0_produces_pending_manifest_and_worker_does_not_fit(self):
        data = _one_mb_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            _payload, digest = _released_protocol(protocol)
            starts = root / "starts"
            starts.mkdir()
            with mock.patch.object(experiment, "_load_real_cohort",
                                   return_value=(data, _cohort_for(data))), \
                 mock.patch.object(experiment, "run_one_fit") as fit:
                prepared = experiment.prepare(
                    run_root=root / "027-pending-study",
                    release_sha256=digest,
                    protocol_json=protocol,
                    protocol_md=None,
                    starts_root=starts,
                )
                manifest = experiment.read_json(Path(prepared["manifest"]))
                job_path = Path(prepared["study_root"]) / manifest["jobs"][0]["job_config"]
                result = experiment.worker(job_path)
            fit.assert_not_called()
            self.assertEqual(prepared["status"], "pending_inputs")
            self.assertEqual(result["status"], "pending_inputs")
            self.assertTrue(all(row["status"] == "pending_input" for row in manifest["jobs"]))
            finalized = experiment.finalize(prepared["study_root"])
            self.assertEqual(finalized["status"], "partial")
            self.assertEqual(finalized["job_count"], 15)
            self.assertEqual(len(finalized["selection"]), 5)
            self.assertTrue(all(item["status"] == "n/a" for item in finalized["selection"]))
            self.assertTrue(finalized["all_15_manifest_arms_preserved"])

    def test_allocate_study_root_starts_at_027_after_reserved_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "025-init").mkdir()
            (root / "026-synthetic").mkdir()
            candidate = experiment.allocate_study_root(root)
            self.assertTrue(candidate.name.startswith("027-"))


if __name__ == "__main__":
    unittest.main()
