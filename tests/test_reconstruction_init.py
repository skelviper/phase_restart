import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from pr import genome, reconstruction_init as init
from pr.paths import SNPFREE


class FullGridExpansionTests(unittest.TestCase):
    def test_header_order_numeric_interpolation_and_endpoint_fill(self):
        names = ("chrZ", "chrA")
        lengths = (13, 11)
        source = {
            "c01a": (np.array([10, 0]), np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
            "c02a": (np.array([5]), np.array([[1.0, 2.0, 3.0]])),
        }
        expanded = init.expand_tracks_to_full_grid(source, names, lengths, 5, "consensus")

        self.assertEqual(expanded["coords"].shape, (2, 6, 3))
        self.assertEqual(expanded["layout"]["names"], names)
        np.testing.assert_allclose(expanded["coords"][0, :3, 0], [0.0, 5.0, 10.0])
        np.testing.assert_allclose(expanded["coords"][0, 3:], [[1.0, 2.0, 3.0]] * 3)
        np.testing.assert_allclose(expanded["coords"][0], expanded["coords"][1])

        chr_z = expanded["per_track"]["c01a"]
        chr_a = expanded["per_track"]["c02a"]
        self.assertEqual((chr_z["exact"], chr_z["interpolated"], chr_z["endpoint_filled"]), (2, 1, 0))
        self.assertEqual((chr_a["exact"], chr_a["interpolated"], chr_a["endpoint_filled"]), (1, 0, 2))
        self.assertTrue(expanded["per_track"]["c01b"]["source_copy_reused"])

    def test_full_grid_has_forty_tracks_in_header_order(self):
        names = tuple("chr%d" % index for index in range(1, 21))
        lengths = (5,) * 20
        source = {}
        for chromosome in range(20):
            for copy in range(2):
                track = "c%02d%s" % (chromosome + 1, "ab"[copy])
                source[track] = (np.array([0]), np.array([[chromosome, copy, 1.0]], dtype=float))
        expanded = init.expand_tracks_to_full_grid(source, names, lengths, 5, "random")

        self.assertEqual(expanded["coords"].shape, (2, 20, 3))
        self.assertEqual(len(expanded["layout"]["track_order"]), 40)
        self.assertEqual(expanded["layout"]["track_order"][:4], ("c01a", "c01b", "c02a", "c02b"))
        self.assertEqual(expanded["layout"]["track_order"][-2:], ("c20a", "c20b"))
        self.assertEqual(len(expanded["per_track"]), 40)


class InitializationTests(unittest.TestCase):
    @staticmethod
    def _consensus_tracks():
        return {
            "c01a": (np.array([0, 10]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
            "c02a": (np.array([0, 10]), np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])),
        }

    @staticmethod
    def _random_tracks():
        tracks = {}
        for chromosome in range(2):
            for copy in range(2):
                track = "c%02d%s" % (chromosome + 1, "ab"[copy])
                tracks[track] = (
                    np.array([0, 10]),
                    np.array([
                        [float(chromosome), float(copy), 0.0],
                        [float(chromosome) + 0.5, float(copy) + 0.25, 0.5],
                    ]),
                )
        return tracks

    def test_consensus_half_difference_is_nonzero_and_seed_reproducible(self):
        first = init.initialize_from_tracks(
            self._consensus_tracks(), ("chrZ", "chrA"), (13, 11), 5, "consensus", seed=1103
        )
        second = init.initialize_from_tracks(
            self._consensus_tracks(), ("chrZ", "chrA"), (13, 11), 5, "consensus", seed=1103
        )

        self.assertEqual(first["coords"].shape, (2, 6, 3))
        np.testing.assert_array_equal(first["coords"], second["coords"])
        half_difference = first["metadata"]["half_difference"]
        self.assertTrue(half_difference["nonzero"])
        self.assertAlmostEqual(half_difference["per_chromosome"][0]["target_rms"], 0.06, places=14)
        self.assertAlmostEqual(half_difference["per_chromosome"][0]["actual_rms"], 0.06, places=14)
        self.assertGreater(half_difference["per_chromosome"][0]["post_final_field_rms"], 0.0)
        self.assertGreater(np.linalg.norm(first["coords"][0] - first["coords"][1]), 0.0)
        self.assertTrue(np.isfinite(first["coords"]).all())
        self.assertLess(np.linalg.norm(first["coords"], axis=2).max(), 1.0)
        self.assertEqual(first["metadata"]["coordinate_axis_order"], "copy_first_a_b_locus_xyz")
        self.assertEqual(len(first["metadata"]["track_mapping"]), 4)

    def test_global_copy_swap_is_an_explicit_gauge_operation(self):
        result = init.initialize_from_tracks(
            self._random_tracks(), ("chrZ", "chrA"), (13, 11), 5, "random", seed=2207
        )
        swapped = init.swap_copy_first(result["coords"])
        np.testing.assert_array_equal(swapped[0], result["coords"][1])
        np.testing.assert_array_equal(swapped[1], result["coords"][0])
        np.testing.assert_array_equal(init.swap_copy_first(swapped), result["coords"])


class WarmStartTests(unittest.TestCase):
    @staticmethod
    def _coarse_layer():
        coords = np.array([
            [[0.30, 0.00, 0.00], [0.10, 0.00, 0.00], [-0.20, 0.10, 0.00], [-0.30, 0.10, 0.00]],
            [[0.35, 0.05, 0.00], [0.15, 0.05, 0.00], [-0.15, 0.15, 0.00], [-0.25, 0.15, 0.00]],
        ])
        positions = np.array([10, 0, 10, 0], dtype=np.int64)
        chromosome_index = np.array([0, 0, 1, 1], dtype=np.int32)
        return coords, positions, chromosome_index

    def test_warm_start_interpolates_new_loci_without_global_rescale(self):
        coords, positions, chromosome_index = self._coarse_layer()
        result = init.warm_start_from_layer(
            coords, positions, chromosome_index, ("chrZ", "chrA"), (20, 15), 5,
            candidate_base_seed=1103,
        )

        self.assertEqual(result["coords"].shape, (2, 7, 3))
        self.assertEqual(result["metadata"]["normalization"]["mode"],
                         "preserve_previous_all_cell_frame_and_scale")
        self.assertEqual(result["metadata"]["normalization"]["clipped_coordinates"], 0)
        self.assertEqual(result["metadata"]["seed"], 3301 + 5 // 1_000_000 + 1103)
        np.testing.assert_allclose(result["coords"][0, 0], coords[0, 1])
        np.testing.assert_allclose(result["coords"][0, 2], coords[0, 0])
        np.testing.assert_allclose(result["coords"][1, 4], coords[1, 3])
        np.testing.assert_allclose(result["coords"][1, 6], coords[1, 2])
        self.assertGreater(result["metadata"]["perturbation"]["perturbed_coordinates"], 0)
        self.assertLess(np.linalg.norm(result["coords"], axis=2).max(), 1.0)

    def test_warm_start_preserves_complete_layer_frame_and_scale(self):
        coords = np.array([
            [[0.10, 0.00, 0.00], [0.20, 0.00, 0.00], [-0.10, 0.50, 0.00], [-0.20, 0.50, 0.00]],
            [[0.15, 0.05, 0.00], [0.25, 0.05, 0.00], [-0.05, 0.55, 0.00], [-0.15, 0.55, 0.00]],
        ])
        positions = np.array([0, 10, 0, 10], dtype=np.int64)
        chromosome_index = np.array([0, 0, 1, 1], dtype=np.int32)
        result = init.warm_start_from_layer(
            coords, positions, chromosome_index, ("chrZ", "chrA"), (20, 20), 10,
            candidate_base_seed=2207,
        )

        np.testing.assert_array_equal(result["coords"], coords)
        self.assertEqual(result["metadata"]["perturbation"]["perturbed_coordinates"], 0)
        self.assertEqual(result["metadata"]["normalization"]["clipped_coordinates"], 0)

    def test_warm_start_clips_outside_unit_ball(self):
        coords = np.full((2, 2, 3), 1.2, dtype=float)
        result = init.warm_start_from_layer(
            coords, np.array([0, 0]), np.array([0, 1]), ("chrZ", "chrA"), (5, 5), 5,
            candidate_base_seed=1103,
        )
        self.assertEqual(result["metadata"]["normalization"]["clipped_coordinates"], 4)
        self.assertLess(np.linalg.norm(result["coords"], axis=2).max(), 1.0)


class ApprovedSourceTests(unittest.TestCase):
    def _source_spec(self, source_path, gate_path, digest):
        return {
            "consensus": {
                "candidate": "consensus",
                "tag": "consensus",
                "stage": "blind",
                "path": source_path,
                "gate_path": gate_path,
                "sha256": digest,
                "base_seed": 1103,
            }
        }

    def test_hash_mismatch_and_nonblind_oracle_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "toy.3dg")
            gate_path = os.path.join(directory, "gate.json")
            with open(source_path, "w") as handle:
                handle.write("#chromosome: c01a 10\n")
                handle.write("c01a\t0\t0\t0\t0\n")
            with open(source_path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            with open(gate_path, "w") as handle:
                json.dump([{
                    "stage": "blind", "tag": "consensus", "path": source_path,
                    "sha256": digest,
                }], handle)

            bad_hash = self._source_spec(source_path, gate_path, "0" * 64)
            with mock.patch.object(init, "APPROVED_SOURCES", bad_hash):
                with self.assertRaisesRegex(init.InitializationError, "SHA256 mismatch"):
                    init.load_approved_source("consensus", ("chr1",), (10,))

            with open(gate_path, "w") as handle:
                json.dump([{
                    "stage": "eval", "tag": "consensus", "path": source_path,
                    "sha256": digest,
                }], handle)
            with mock.patch.object(init, "APPROVED_SOURCES", self._source_spec(source_path, gate_path, digest)):
                with self.assertRaisesRegex(init.InitializationError, "not blind"):
                    init.load_approved_source("consensus", ("chr1",), (10,))
                with self.assertRaisesRegex(init.InitializationError, "only approved blind candidates"):
                    init.load_approved_source("oracle", ("chr1",), (10,))

    def test_approved_014_sources_expand_without_phase_or_reference_access(self):
        headers = genome.chrom_lengths(SNPFREE)
        names = tuple(name for name, _length in headers)
        lengths = tuple(length for _name, length in headers)
        expected_sha = {
            "consensus": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
            "random": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
        }
        for candidate in ("consensus", "random"):
            result = init.initialize_approved_candidate(candidate, names, lengths, 1_000_000)
            self.assertEqual(result["coords"].shape, (2, 2645, 3))
            self.assertEqual(result["metadata"]["n_tracks"], 40)
            self.assertEqual(result["metadata"]["source"]["gate_stage"], "blind")
            self.assertEqual(result["metadata"]["source"]["source_sha256"], expected_sha[candidate])
            self.assertTrue(np.isfinite(result["coords"]).all())
            self.assertLess(np.linalg.norm(result["coords"], axis=2).max(), 1.0)
            self.assertGreaterEqual(result["metadata"]["elapsed_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
