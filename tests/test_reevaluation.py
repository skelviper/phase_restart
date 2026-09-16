import gzip
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import run
from pr import fdg, genome, gwfdg, pairs7, reevaluate, refeval, s0
from pr.paths import BIN, OFF, SNPFREE


class OpenGate:
    def require(self, stage):
        self.stage = stage


def _track(points):
    return {OFF + i * BIN: np.asarray(point, dtype=float) for i, point in enumerate(points)}


def _structures(n_bins=60):
    t = np.arange(n_bins, dtype=float)
    mat = np.column_stack((t, 0.09 * t * t + np.sin(t * 0.31), np.cos(t * 0.17) * 4.0))
    pat = np.column_stack((np.sin(t * 0.37) * 7.0, 0.71 * t + np.cos(t * 0.11),
                           np.cos(t * 0.29) * 5.0))
    ref = {"chr1(mat)": _track(mat), "chr1(pat)": _track(pat)}
    ours = {"c01a": _track(mat), "c01b": _track(pat)}
    return ours, ref


def _all_pairs(n_bins):
    return np.triu_indices(n_bins, k=1)


def _swapped(structs):
    return {"c01a": structs["c01b"], "c01b": structs["c01a"]}


class RefEvaluationTests(unittest.TestCase):
    def test_gauge_is_geometric_not_truth_selected(self):
        ours, ref = _structures(24)
        b1, b2 = _all_pairs(24)
        da = refeval.ours_dists(ours, 0, 0, b1, b2, 24)
        db = refeval.ours_dists(ours, 0, 1, b1, b2, 24)
        # 有意让 phase labels 对每个 pair 选择距离更远的 candidate copy。
        labels = np.where(da < db, 0, 1).astype(np.int8)
        result = refeval.r1_accuracy(ours, 0, "chr1", labels, b1, b2, 24, ref=ref)
        self.assertEqual(result["gauge"]["orientation"], "direct")
        self.assertLess(result["accuracy"], 0.05)
        self.assertGreater(result["legacy_truth_max"], 0.95)
        changed_truth = refeval.r1_accuracy(ours, 0, "chr1", 1 - labels, b1, b2, 24, ref=ref)
        self.assertEqual(changed_truth["gauge"]["orientation"], "direct")

    def test_global_copy_swap_preserves_primary_readouts(self):
        ours, ref = _structures(60)
        b1, b2 = _all_pairs(60)
        labels = np.where((b1 + 2 * b2) % 2, 0, 1).astype(np.int8)
        before_r1 = refeval.r1_accuracy(ours, 0, "chr1", labels, b1, b2, 60, ref=ref)
        before_r2 = refeval.r2_table(ours, 0, "chr1", ref, 60)
        before_r3 = refeval.r3_fragments(ours, 0, "chr1", ref, 60)
        swapped = _swapped(ours)
        after_r1 = refeval.r1_accuracy(swapped, 0, "chr1", labels, b1, b2, 60, ref=ref)
        after_r2 = refeval.r2_table(swapped, 0, "chr1", ref, 60)
        after_r3 = refeval.r3_fragments(swapped, 0, "chr1", ref, 60)
        self.assertAlmostEqual(before_r1["accuracy"], after_r1["accuracy"], places=12)
        self.assertAlmostEqual(before_r2["contrast"], after_r2["contrast"], places=12)
        self.assertEqual(before_r3["frac_consistent"], after_r3["frac_consistent"])
        self.assertEqual(before_r3["longest_run"], after_r3["longest_run"])
        self.assertEqual(before_r3["n_walls"], after_r3["n_walls"])

    def test_geometry_tie_uses_average_and_collapse_is_not_r3_success(self):
        ours, ref = _structures(60)
        collapsed = {"c01a": ours["c01a"], "c01b": ours["c01a"]}
        b1, b2 = _all_pairs(60)
        labels = np.where((b1 + b2) % 2, 0, 1).astype(np.int8)
        r1 = refeval.r1_accuracy(collapsed, 0, "chr1", labels, b1, b2, 60, ref=ref)
        r2 = refeval.r2_table(collapsed, 0, "chr1", ref, 60)
        r3 = refeval.r3_fragments(collapsed, 0, "chr1", ref, 60)
        self.assertEqual(r1["gauge"]["status"], "tie_average_orientations")
        self.assertAlmostEqual(r1["accuracy"], 0.5, places=12)
        self.assertAlmostEqual(r2["contrast"], 0.0, places=12)
        self.assertFalse(r3["applicable"])
        self.assertEqual(r3["frac_consistent"], None)
        self.assertEqual(r3["n_fragments_tied"], 3)

    def test_consensus_is_not_degenerate_two_copy_metric(self):
        ours, ref = _structures(30)
        consensus = {"c01a": ours["c01a"]}
        b1, b2 = _all_pairs(30)
        labels = np.zeros(len(b1), dtype=np.int8)
        r1 = refeval.r1_accuracy(consensus, 0, "chr1", labels, b1, b2, 30, ref=ref)
        r2 = refeval.r2_table(consensus, 0, "chr1", ref, 30)
        r3 = refeval.r3_fragments(consensus, 0, "chr1", ref, 30)
        self.assertFalse(r1["applicable"])
        self.assertIsNone(r1["accuracy"])
        self.assertFalse(r2["applicable"])
        self.assertIsNone(r2["contrast"])
        self.assertIn("a0_mat", r2["single_track_baseline"])
        self.assertIn("a0_pat", r2["single_track_baseline"])
        self.assertFalse(r3["applicable"])
        self.assertIsNone(r3["frac_consistent"])

    def test_r3_gap_breaks_majority_run_and_walls(self):
        ours, ref = _structures(60)
        for copy in ("c01a", "c01b"):
            ours[copy] = {pos: xyz for pos, xyz in ours[copy].items()
                          if not (OFF + 20 * BIN <= pos < OFF + 40 * BIN)}
        r3 = refeval.r3_fragments(ours, 0, "chr1", ref, 60)
        self.assertTrue(r3["applicable"])
        self.assertEqual(r3["n_fragments_applicable"], 2)
        self.assertEqual(r3["n_fragments_insufficient"], 1)
        self.assertEqual(r3["longest_run"], 1)
        self.assertEqual(r3["n_walls"], 0)

    def test_r3_tied_majority_uses_swap_invariant_longest_run(self):
        ours, ref = _structures(200)
        # 每个各有五个 local labels，但最长 run 不同（0：四个，1：三个）。
        labels = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1]
        stitched = {"c01a": {}, "c01b": {}}
        for bin_index in range(200):
            phase = labels[bin_index // 20]
            pos = OFF + bin_index * BIN
            stitched["c01a"][pos] = ref["chr1(mat)" if phase == 0 else "chr1(pat)"][pos]
            stitched["c01b"][pos] = ref["chr1(pat)" if phase == 0 else "chr1(mat)"][pos]
        first = refeval.r3_fragments(stitched, 0, "chr1", ref, 200)
        second = refeval.r3_fragments(_swapped(stitched), 0, "chr1", ref, 200)
        self.assertTrue(first["global_label_tied"])
        self.assertIsNone(first["global_label"])
        self.assertEqual(first["frac_consistent"], 0.5)
        self.assertEqual(first["longest_run"], 4)
        self.assertEqual(first["longest_run"], second["longest_run"])
        self.assertEqual(first["n_walls"], second["n_walls"])
        paired = reevaluate._paired_r3(first, second)
        self.assertEqual(paired["a"], 0.5)
        self.assertEqual(paired["b"], 0.5)
        self.assertEqual(paired["delta_b_minus_a"], 0.0)

    def test_out_of_grid_bins_are_counted_without_negative_indexing(self):
        ours, ref = _structures(24)
        b1 = np.array([-1, 0], dtype=np.int64)
        b2 = np.array([1, 2], dtype=np.int64)
        labels = np.array([1, 1], dtype=np.int8)
        result = refeval.r1_accuracy(ours, 0, "chr1", labels, b1, b2, 24, ref=ref)
        self.assertEqual(result["n_out_of_grid_records"], 1)
        self.assertEqual(result["n_out_of_grid_excluded"], 1)
        self.assertEqual(result["n_denominator"], 1)


class PairsSchemaTests(unittest.TestCase):
    def _write_gz(self, directory, name, text):
        path = os.path.join(directory, name)
        with gzip.open(path, "wt") as handle:
            handle.write(text)
        return path

    def test_phase_header_is_rejected_and_legal_seven_columns_load(self):
        header = "## pairs format v1.0\n#chromosome: chr1 10000000\n"
        with tempfile.TemporaryDirectory() as directory:
            leaked = self._write_gz(directory, "leaked.pairs.gz", header +
                "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\tphase0\n"
                ".\tchr1\t3000000\tchr1\t6000000\t+\t+\t0\n")
            with self.assertRaises(pairs7.PhaseLeakError):
                genome.load_all(leaked)

            legal = self._write_gz(directory, "legal.pairs.gz", header +
                "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n"
                ".\tchr1\t6000000\tchr1\t3000000\t+\t+\n")
            loaded = genome.load_all(legal)
            self.assertEqual(loaded["p1"].tolist(), [3000000])
            self.assertEqual(loaded["p2"].tolist(), [6000000])

            malformed = self._write_gz(directory, "malformed.pairs.gz", header +
                "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n"
                ".\tchr1\t3000000\tchr1\t6000000\t+\t+\textra\n")
            with self.assertRaises(pairs7.PhaseLeakError):
                genome.load_all(malformed)

    def test_phase_end_labels_follow_reverse_cis_canonicalization(self):
        header = "## pairs format v1.0\n#chromosome: chr1 10000000\n"
        with tempfile.TemporaryDirectory() as directory:
            raw = self._write_gz(directory, "raw.pairs.gz", header +
                "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\tphase0\tphase1\n"
                ".\tchr1\t6000000\tchr1\t3000000\t+\t+\t0\t1\n")
            contacts = {"ci": np.array([0], dtype=np.int32), "p1": np.array([3000000]),
                        "cj": np.array([0], dtype=np.int32), "p2": np.array([6000000]),
                        "names": ["chr1"]}
            a1, a2 = refeval.load_labels_two(OpenGate(), contacts, pairs_path=raw)
            self.assertEqual(a1.tolist(), [1])
            self.assertEqual(a2.tolist(), [0])


class SplitCommandTests(unittest.TestCase):
    def test_prepare_cli_dispatches_without_touching_inputs(self):
        with mock.patch.object(run, "cmd_prepare") as prepare, \
             mock.patch.object(sys, "argv", ["run.py", "prepare"]):
            run.main()
        prepare.assert_called_once()

    def test_split_uses_oracle_common_denominator_and_strict_json(self):
        contacts = {"ci": np.array([0], dtype=np.int32), "p1": np.array([3000000]),
                    "cj": np.array([0], dtype=np.int32), "p2": np.array([4000000]),
                    "cis": np.array([True]), "names": ["chr1"]}
        observed_r1_calls = []

        with tempfile.TemporaryDirectory() as directory:
            def fake_fit(name, _contacts, _lengths, _k1, _k2, _workdir, coordsdir, **_kwargs):
                path = os.path.join(coordsdir, name + ".3dg")
                with open(path, "w") as handle:
                    handle.write("c01a\t0\t0\t0\t0\n")
                return {"_name": name}, path, 0.0, 1

            def fake_r2(structs, *_args):
                if structs["_name"] == "consensus":
                    return {"applicable": False, "contrast": None, "gauge": None}
                return {"applicable": True, "contrast": 0.2,
                        "gauge": {"applicable": True, "orientation": "direct"}}

            def fake_r1(structs, *_args, **kwargs):
                observed_r1_calls.append((structs["_name"], kwargs))
                if structs["_name"] == "consensus":
                    return {"applicable": False, "accuracy": None, "reference_ceiling": None,
                            "oracle_fit_ceiling": None, "n_denominator": 0}
                return {"applicable": True, "accuracy": 0.55, "reference_ceiling": 0.80,
                        "oracle_fit_ceiling": 0.65, "n_denominator": 1}

            def fake_r3(structs, *_args):
                if structs["_name"] == "consensus":
                    return {"applicable": False, "frac_consistent": None, "n_walls": None}
                return {"applicable": True, "frac_consistent": 0.5, "n_walls": 1}

            args = SimpleNamespace(out=os.path.join(directory, "mock-split"), assignment_seed=0,
                                   native_seed=1, n_iter=1, bin_size="1m", genome=True,
                                   func=None)
            run.LOG.clear()
            with mock.patch.object(run.genome, "chrom_lengths", return_value=[("chr1", 5000000)]), \
                 mock.patch.object(run.genome, "load_all", return_value=contacts), \
                 mock.patch.object(run.genome, "n_bins_per_chrom", return_value=[2]), \
                 mock.patch.object(run.genome, "random_assignment", return_value=(np.array([0]), np.array([0]))), \
                 mock.patch.object(run.s0, "fit", side_effect=fake_fit), \
                 mock.patch.object(run.refeval, "load_labels_two", return_value=(np.array([0]), np.array([0]))), \
                 mock.patch.object(run.refeval, "load_reference", return_value={}), \
                 mock.patch.object(run.refeval, "r1_accuracy", side_effect=fake_r1), \
                 mock.patch.object(run.refeval, "r2_table", side_effect=fake_r2), \
                 mock.patch.object(run.refeval, "r3_fragments", side_effect=fake_r3):
                run.cmd_split(args)
            self.assertEqual(len(observed_r1_calls), 3)
            for _name, kwargs in observed_r1_calls:
                self.assertEqual(kwargs["oracle_structs"]["_name"], "oracle")
                self.assertEqual(kwargs["oracle_gauge"]["orientation"], "direct")
            with open(os.path.join(args.out, "results.json")) as handle:
                payload = json.load(handle)
            self.assertEqual(payload["per_chromosome"]["chr1"]["random"]["R1"]["n_denominator"], 1)
            self.assertEqual(payload["config"]["assignment_seed"], 0)
            self.assertEqual(payload["config"]["native_seed"], 1)

    def test_split_uses_full_snpfree_loader_and_arms_gate_after_blind_coordinates(self):
        expected_lengths = genome.chrom_lengths(SNPFREE)
        expected_names = [name for name, _length in expected_lengths]
        seen_names, seen_counts, events = [], [], []
        real_load_labels = refeval.load_labels_two
        real_load_reference = refeval.load_reference

        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "full-loader-check")

            def fake_fit(name, contacts, lengths, _k1, _k2, _workdir, coordsdir, **_kwargs):
                self.assertEqual(lengths, expected_lengths)
                seen_names.append(contacts["names"])
                seen_counts.append(len(contacts["ci"]))
                path = os.path.join(coordsdir, name + ".3dg")
                with open(path, "w") as handle:
                    handle.write("c01a\t0\t0\t0\t0\n")
                return {"_name": name}, path, 0.0, len(contacts["ci"])

            def checked_labels(gate, contacts):
                gate.require(refeval.STAGE)
                self.assertEqual([entry["tag"] for entry in gate.entries], ["consensus", "random"])
                events.append("labels")
                return real_load_labels(gate, contacts)

            def checked_reference(gate):
                gate.require(refeval.STAGE)
                self.assertEqual([entry["tag"] for entry in gate.entries],
                                 ["consensus", "random", "oracle"])
                events.append("reference")
                return real_load_reference(gate)

            def fake_r2(structs, *_args):
                if structs["_name"] == "consensus":
                    return {"applicable": False, "contrast": None, "gauge": None}
                return {"applicable": True, "contrast": 0.0,
                        "gauge": {"applicable": True, "orientation": "direct"}}

            def fake_r1(structs, *_args, **_kwargs):
                if structs["_name"] == "consensus":
                    return {"applicable": False, "accuracy": None, "reference_ceiling": None,
                            "oracle_fit_ceiling": None, "n_denominator": 0}
                return {"applicable": True, "accuracy": 0.5, "reference_ceiling": 0.8,
                        "oracle_fit_ceiling": 0.6, "n_denominator": 1}

            def fake_r3(structs, *_args):
                if structs["_name"] == "consensus":
                    return {"applicable": False, "frac_consistent": None, "n_walls": None}
                return {"applicable": True, "frac_consistent": 0.5, "n_walls": 1}

            args = SimpleNamespace(out=outdir, assignment_seed=0, native_seed=1, n_iter=1,
                                   bin_size="1m", genome=True, func=None)
            run.LOG.clear()
            with mock.patch.object(run.s0, "fit", side_effect=fake_fit), \
                 mock.patch.object(run.refeval, "load_labels_two", side_effect=checked_labels), \
                 mock.patch.object(run.refeval, "load_reference", side_effect=checked_reference), \
                 mock.patch.object(run.refeval, "r1_accuracy", side_effect=fake_r1), \
                 mock.patch.object(run.refeval, "r2_table", side_effect=fake_r2), \
                 mock.patch.object(run.refeval, "r3_fragments", side_effect=fake_r3):
                run.cmd_split(args)

        self.assertEqual(seen_names, [expected_names, expected_names, expected_names])
        self.assertEqual(seen_counts, [1_703_888, 1_703_888, 1_703_888])
        self.assertEqual(events, ["labels", "reference"])


class CommandGuardTests(unittest.TestCase):
    def test_existing_split_output_is_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "existing")
            os.mkdir(outdir)
            args = SimpleNamespace(out=outdir, assignment_seed=0, native_seed=1, n_iter=1,
                                   bin_size="1m")
            with mock.patch.object(run.genome, "chrom_lengths") as chrom_lengths:
                with self.assertRaises(FileExistsError):
                    run.cmd_split(args)
            chrom_lengths.assert_not_called()

    def test_existing_stage1_output_is_rejected_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "existing")
            os.mkdir(outdir)
            args = SimpleNamespace(out=outdir, chrom="chr1", relax_rounds=1)
            with mock.patch.object(run.pairs7, "load") as load:
                with self.assertRaises(FileExistsError):
                    run.cmd_stage1(args)
            load.assert_not_called()

    def test_split_rejects_non_1m_grid_before_output_or_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "new-output")
            args = SimpleNamespace(out=outdir, assignment_seed=0, native_seed=1, n_iter=1,
                                   bin_size="2m")
            with mock.patch.object(run.genome, "chrom_lengths") as chrom_lengths:
                with self.assertRaisesRegex(ValueError, "only --bin-size 1m"):
                    run.cmd_split(args)
            chrom_lengths.assert_not_called()
            self.assertFalse(os.path.exists(outdir))

    def test_stage1_rejects_multiple_relax_rounds_before_output_or_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = os.path.join(directory, "new-output")
            args = SimpleNamespace(out=outdir, chrom="chr1", relax_rounds=2)
            with mock.patch.object(run.pairs7, "load") as load:
                with self.assertRaisesRegex(ValueError, "only --relax-rounds 1"):
                    run.cmd_stage1(args)
            load.assert_not_called()
            self.assertFalse(os.path.exists(outdir))

    def test_split_accepts_documented_genome_flag(self):
        with mock.patch.object(run, "cmd_split") as split, \
             mock.patch.object(sys, "argv", ["run.py", "split", "--out", "new", "--genome"]):
            run.main()
        parsed = split.call_args.args[0]
        self.assertTrue(parsed.genome)
        self.assertEqual(parsed.assignment_seed, 0)
        self.assertEqual(parsed.native_seed, 1)

    def test_cli_rejects_non_1m_final_grid(self):
        with mock.patch.object(sys, "argv", ["run.py", "split", "--out", "new", "--bin-size", "2m"]):
            with self.assertRaises(SystemExit) as caught:
                run.main()
        self.assertEqual(caught.exception.code, 2)


class NativeWrapperTests(unittest.TestCase):
    def test_single_chromosome_options_precede_first_bin(self):
        completed = SimpleNamespace(returncode=0, stderr="")
        with mock.patch.object(fdg.subprocess, "run", return_value=completed) as invoke:
            fdg.run_fdg("input.pairs.gz", "output.3dg", seed=7, max_iter=3)
        cmd = invoke.call_args.args[0]
        self.assertLess(cmd.index("-P1"), cmd.index("-i"))
        self.assertLess(cmd.index("-s"), cmd.index("-b1m"))
        self.assertLess(cmd.index("-n"), cmd.index("-b1m"))
        self.assertEqual(cmd[cmd.index("-s") + 1], "7")
        self.assertEqual(cmd[cmd.index("-n") + 1], "3")

    def test_genome_wide_options_and_seed_precede_first_bin(self):
        completed = SimpleNamespace(returncode=0, stderr="")
        with mock.patch.object(gwfdg.subprocess, "run", return_value=completed) as invoke:
            gwfdg.run("input.pairs.gz", "output.3dg", n_iter=4, bin_size="2m", seed=11)
        cmd = invoke.call_args.args[0]
        self.assertLess(cmd.index("-P1"), cmd.index("-i"))
        self.assertLess(cmd.index("-s"), cmd.index("-b2m"))
        self.assertLess(cmd.index("-n"), cmd.index("-b2m"))
        self.assertEqual(cmd[cmd.index("-s") + 1], "11")
        self.assertEqual(cmd[cmd.index("-n") + 1], "4")

    def test_s0_forwards_native_seed_to_genome_wrapper(self):
        contacts = {"ci": np.array([0], dtype=np.int32), "p1": np.array([3000000]),
                    "cj": np.array([0], dtype=np.int32), "p2": np.array([4000000]),
                    "cis": np.array([True]), "names": ["chr1"]}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(s0.gwfdg, "write_pairs"), \
                 mock.patch.object(s0.gwfdg, "run", return_value=("ignored", 0.0)) as fit, \
                 mock.patch.object(s0.gwfdg, "read_3dg", return_value={}):
                s0.fit("consensus", contacts, [("chr1", 5000000)], None, None, directory,
                       directory, n_iter=4, single=True, seed=23)
        self.assertEqual(fit.call_args.kwargs["seed"], 23)

    def test_native_prebin_seeds_change_one_iteration_fixture(self):
        p1 = np.array([3000000, 3000000, 4000000, 4000000, 5000000, 6000000], dtype=np.int64)
        p2 = np.array([4000000, 5000000, 5000000, 6000000, 7000000, 7000000], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            pairs = os.path.join(directory, "fixture.pairs.gz")
            first = os.path.join(directory, "seed1.3dg")
            second = os.path.join(directory, "seed7.3dg")
            fdg.write_pairs(pairs, "chr1", p1, p2)
            with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}, clear=False):
                fdg.run_fdg(pairs, first, seed=1, max_iter=1)
                fdg.run_fdg(pairs, second, seed=7, max_iter=1)
            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()
        self.assertNotEqual(first_bytes, second_bytes)


if __name__ == "__main__":
    unittest.main()
