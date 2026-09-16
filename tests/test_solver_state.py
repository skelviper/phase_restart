import tempfile
import unittest
from pathlib import Path

import numpy as np

from pr import contact_model
from pr.solver_state import (SolverStateError, export_present_3dg,
                             load_solver_state, sha256_file,
                             write_presence_mask, write_solver_state)


def fixture(bin_size):
    ci = np.asarray([0, 0, 1, 0], dtype=np.int64)
    p1 = np.asarray([1, bin_size + 1, 1, 1], dtype=np.int64)
    cj = np.asarray([0, 1, 1, 1], dtype=np.int64)
    p2 = np.asarray([bin_size + 1, 1, bin_size + 1, 2 * bin_size + 1], dtype=np.int64)
    return contact_model.aggregate_from_arrays(
        ("chrA", "chrB"), (3 * bin_size, 3 * bin_size), ci, p1, cj, p2, bin_size)


class SolverStateTests(unittest.TestCase):
    def test_two_chromosome_two_resolution_export_and_immutability(self):
        for bin_size in (10, 20):
            with self.subTest(bin_size=bin_size), tempfile.TemporaryDirectory() as directory:
                data = fixture(bin_size)
                raw_y = np.arange(2 * data.n_loci * 3, dtype=np.float64).reshape(2, data.n_loci, 3)
                raw_y = (raw_y - raw_y.mean()) / 200.0
                coordinates = contact_model.sphere_forward(raw_y)
                p = 0.75
                q = float(contact_model.q_from_p(p))
                theta = np.concatenate((raw_y.reshape(-1), np.asarray([q])))
                root = Path(directory)
                state_path = root / "solver_state.npz"
                record = write_solver_state(state_path, coordinates=coordinates, raw_y=raw_y,
                                            theta=theta, p=p, q=q)
                state_hash = record["sha256"]
                loaded = load_solver_state(state_path)
                self.assertTrue(np.array_equal(loaded["theta"], theta))
                mask = np.ones((2, data.n_loci), dtype=bool)
                mask[0, data.chromosome_slice(0).start] = False
                mask[1, data.chromosome_slice(1).start + 1] = False
                write_presence_mask(root / "presence_mask.npz", mask, data.n_loci)
                exported = export_present_3dg(root / "export.3dg", data, coordinates, mask)
                self.assertEqual(exported["rows"], int(mask.sum()))
                self.assertEqual(sha256_file(state_path), state_hash)
                lines = (root / "export.3dg").read_text(encoding="utf-8").splitlines()
                self.assertFalse(any(line.startswith("c01a\t0\t") for line in lines))
                self.assertFalse(any(line.startswith("c02b\t%d\t" % bin_size) for line in lines))
                self.assertTrue(any(line.startswith("c01b\t%d\t" % (2 * bin_size)) for line in lines))
                with self.assertRaises(SolverStateError):
                    write_solver_state(state_path, coordinates=coordinates, raw_y=raw_y,
                                       theta=theta, p=p, q=q)
                with self.assertRaises(SolverStateError):
                    export_present_3dg(root / "export.3dg", data, coordinates, mask)

    def test_rejects_nonfinite_and_inconsistent_state(self):
        data = fixture(10)
        raw_y = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        coordinates = contact_model.sphere_forward(raw_y)
        q = float(contact_model.q_from_p(0.75))
        theta = np.concatenate((raw_y.reshape(-1), np.asarray([q])))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SolverStateError):
                write_solver_state(Path(directory) / "bad.npz", coordinates=coordinates + 0.1,
                                   raw_y=raw_y, theta=theta, p=0.75, q=q)
            broken = raw_y.copy()
            broken[0, 0, 0] = np.nan
            with self.assertRaises(SolverStateError):
                write_solver_state(Path(directory) / "nan.npz", coordinates=coordinates,
                                   raw_y=broken, theta=theta, p=0.75, q=q)

    def test_rejects_each_solver_contract_mismatch_and_mask_overwrite(self):
        data = fixture(10)
        raw_y = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        coordinates = contact_model.sphere_forward(raw_y)
        p = 0.75
        q = float(contact_model.q_from_p(p))
        theta = np.concatenate((raw_y.reshape(-1), np.asarray([q])))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_raw_theta = theta.copy()
            bad_raw_theta[0] = 0.01
            with self.assertRaisesRegex(SolverStateError, "theta_raw_y_max_abs"):
                write_solver_state(root / "bad-raw-theta.npz", coordinates=coordinates,
                                   raw_y=raw_y, theta=bad_raw_theta, p=p, q=q)
            bad_q_theta = theta.copy()
            bad_q_theta[-1] += 0.01
            with self.assertRaisesRegex(SolverStateError, "theta_q_abs"):
                write_solver_state(root / "bad-q-theta.npz", coordinates=coordinates,
                                   raw_y=raw_y, theta=bad_q_theta, p=p, q=q)
            with self.assertRaisesRegex(SolverStateError, "q_p_abs"):
                write_solver_state(root / "bad-pq.npz", coordinates=coordinates,
                                   raw_y=raw_y, theta=theta, p=0.70, q=q)
            mask_path = root / "presence-mask.npz"
            mask = np.ones((2, data.n_loci), dtype=bool)
            write_presence_mask(mask_path, mask, data.n_loci)
            with self.assertRaises(SolverStateError):
                write_presence_mask(mask_path, mask, data.n_loci)


if __name__ == "__main__":
    unittest.main()
