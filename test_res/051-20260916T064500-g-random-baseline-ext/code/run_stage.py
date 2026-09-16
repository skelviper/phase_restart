"""051 训练运行器：baseline 复现 / A extra-levels / B no-bend / C reference-beads。

每个进程只跑一个 (condition, stage)，读上一阶段的 endpoint npz 作为下一阶段起点，
写出本 run 的 stages/<fit_id>/<stage>.json 与 coords/<fit_id>/<stage>.{npz,3dg}。

条件语义：

* ``B``：权重 bend 从 0.01 改为 0，其余（count/bond/repulsion/p_prior/e/球域/算法）不变，
  从 5Mb 起点重跑 5->2->1。
* ``C``：同 5->2->1 与 bend=0.01，但使用逐 copy 珠子 mask 的 ``PerCopySupportObjective``。
* ``R``：baseline 复现，直接复用 046 已发布的 1Mb endpoint（不重跑，只做 hash 断言）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for _path in (str(HERE), str(S049)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import round_runner  # noqa: E402
from frozen_imports import contact_model, data_io, formal_controller, reconstruction_init  # noqa: E402
from masked_objective import PerCopySupportObjective  # noqa: E402
from support_data import build_filtered  # noqa: E402
from round_paths import AGGREGATE_1MB, AGGREGATE_1MB_SHA256  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402

STAGE_BIN = {"20Mb": 20_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000,
             "2Mb": 2_000_000, "1Mb": 1_000_000}
CONDITIONS = {
    "A": {"stages": ("20Mb", "10Mb", "5Mb", "2Mb", "1Mb"), "bend": 0.01,
          "caps": {"20Mb": 200, "10Mb": 200, "5Mb": 612, "2Mb": 404, "1Mb": 486}},
    "B": {"stages": ("5Mb", "2Mb", "1Mb"), "bend": 0.0,
          "caps": {"5Mb": 612, "2Mb": 404, "1Mb": 486}},
    "C": {"stages": ("5Mb", "2Mb", "1Mb"), "bend": 0.01,
          "caps": {"5Mb": 612, "2Mb": 404, "1Mb": 486}},
}
BASELINE_NPZ = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz"
BASELINE_3DG = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg"
BASELINE_NPZ_SHA = "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752"
BASELINE_3DG_SHA = "ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b"
SUPPORT_DIR = RUN / "inputs"


def load_stage_data(stage: str, condition: str):
    bin_size = STAGE_BIN[stage]
    if bin_size == 1_000_000:
        data = data_io.load_aggregate(AGGREGATE_1MB)
    else:
        data = data_io.load_aggregate(RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size))
    if int(data.bin_size) != bin_size:
        raise RuntimeError("stage data bin_size mismatch")
    mask = None
    if condition == "C":
        with np.load(SUPPORT_DIR / ("reference_support_%s.npz" % stage), allow_pickle=False) as payload:
            if int(payload["bin_size"]) != bin_size:
                raise RuntimeError("support layer bin_size mismatch")
            mask = np.asarray(payload["mask"], dtype=bool)
    return data, mask


def load_start(condition: str, stage: str, previous: Path | None, data):
    """返回 (coordinates, raw_y, p, q)；跨层时按冻结 prolongation 规则升采样。"""
    bin_size = STAGE_BIN[stage]
    if previous is not None:
        with np.load(previous, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            p_init = float(np.asarray(payload["p"]).item())
            q_init = float(np.asarray(payload["q"]).item())
        if coordinates.shape[1] != int(data.n_loci):
            previous_bin = STAGE_BIN[Path(previous).stem]
            previous_data = load_stage_data(Path(previous).stem, "A")[0]
            if condition == "C":
                # C 跨层：只用上一层 mask 的有效点构造 40 条 source track（真实 bp），
                # 走与 045 相同的 zero-optimization prolongation（expand + 同一 noise/clip）。
                with np.load(SUPPORT_DIR / ("reference_support_%s.npz" % Path(previous).stem),
                             allow_pickle=False) as payload:
                    previous_mask = np.asarray(payload["mask"], dtype=bool)
                names = tuple(previous_data.chromosome_names)
                source_tracks = {}
                for chromosome_index in range(len(names)):
                    slc = previous_data.chromosome_slice(chromosome_index)
                    positions = (np.asarray(previous_data.locus_bin[slc])
                                 * previous_bin).astype(np.int64)
                    for copy in (0, 1):
                        keep = previous_mask[copy, slc]
                        track = "c%02d%s" % (chromosome_index + 1, "ab"[copy])
                        source_tracks[track] = (positions[keep].copy(),
                                                coordinates[copy, slc][keep].copy())
                expanded = reconstruction_init.expand_tracks_to_full_grid(
                    source_tracks, tuple(data.chromosome_names),
                    tuple(int(v) for v in data.chromosome_lengths), bin_size, "random")
                # 冻结扰动规则（pr/reconstruction_init.py:592-602 同 seed / 同 shape /
                # 同 scale / 同 clip）：expanded 已经落在目标网格上，不能再调
                # warm_start_from_layer（那会把全部点当 exact 旧点，perturb_mask 全 False）。
                coordinates = np.asarray(expanded["coords"], dtype=np.float64)
                perturb_mask = np.asarray(expanded["perturb_mask"], dtype=bool)
                l0 = float((2 * int(data.n_loci)) ** (-1.0 / 3.0))
                perturb_scale = 0.025 * l0
                seed = 3301 + bin_size // 1_000_000 + int(2207)
                rng = np.random.default_rng(seed)
                noise = rng.normal(scale=perturb_scale, size=coordinates.shape)
                coordinates[perturb_mask] += noise[perturb_mask]
                limit = 1.0 - 1e-6
                radii = np.linalg.norm(coordinates, axis=2)
                clipped = radii >= limit
                if clipped.any():
                    coordinates[clipped] *= (limit / radii[clipped])[:, None]
            else:
                positions = np.broadcast_to(
                    (np.asarray(previous_data.locus_bin) * previous_bin).astype(np.int64),
                    (2, coordinates.shape[1])).copy()
                chromosomes = np.broadcast_to(np.asarray(previous_data.locus_chromosome,
                                                         dtype=np.int32),
                                              (2, coordinates.shape[1])).copy()
                warm = reconstruction_init.warm_start_from_layer(
                    coordinates, positions, chromosomes, tuple(data.chromosome_names),
                    tuple(int(v) for v in data.chromosome_lengths), bin_size, 2207)
                coordinates = np.asarray(warm["coords"], dtype=np.float64)
        contact_model.assert_inside_unit_ball(coordinates)
        raw_y = contact_model.sphere_inverse(coordinates)
        return coordinates, raw_y, p_init, q_init
    path = RUN / "coords" / "initial" / ("random_%s.npz" % stage)
    if condition in ("B", "C") and stage == "5Mb":
        path = RUN / "coords" / "initial" / "base_start_5Mb.npz"
    coordinates, raw_y, p_init, _meta = data_io.load_start(path)
    return coordinates, raw_y, float(p_init), float(contact_model.q_from_p(float(p_init)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("A", "B", "C"))
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_BIN))
    parser.add_argument("--previous", default=None,
                        help="上一阶段 endpoint npz；缺省时使用本条件的零优化起点")
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()
    spec = CONDITIONS[args.condition]
    if args.stage not in spec["stages"]:
        raise SystemExit("stage %s is not part of condition %s" % (args.stage, args.condition))
    previous = Path(args.previous) if args.previous else None
    out_root = Path(args.out_root) if args.out_root else (RUN / "coords")
    fit_id = "%s-%s" % (args.condition, "reference-beads" if args.condition == "C" else
                        ("no-bend" if args.condition == "B" else "extra-levels"))
    fg_cap = int(spec["caps"][args.stage])
    weights = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": float(spec["bend"]),
               "p_prior": 1.0}

    round_runner.install(RUN)
    round_runner.set_active("A", "raw")
    aggregate_sha = round_runner.sha256_file(AGGREGATE_1MB)
    if aggregate_sha != AGGREGATE_1MB_SHA256:
        raise RuntimeError("frozen 1Mb aggregate SHA mismatch")
    data, mask = load_stage_data(args.stage, args.condition)
    coordinates, raw_y, p_init, q_init = load_start(args.condition, args.stage, previous, data)
    sphere = contact_model.sphere_forward(raw_y)
    sphere_error = float(np.max(np.abs(sphere - coordinates)))
    if sphere_error > 1e-10:
        raise RuntimeError("start raw_y does not map back to coordinates")
    mode = "V0-fixed-production-e" if data.count_mode == "raw_integer" else "V0-known-generating-e"
    objective_state: dict[str, object] = {"objective": None, "audit": None}

    def factory(d, model_id, w, known_e):
        if objective_state["objective"] is None:
            if args.condition == "C":
                filtered, exposure, audit = build_filtered(d, mask)
                objective_state["audit"] = audit
                obj = PerCopySupportObjective(
                    d, model_id, weights=w, mask=mask, filtered=filtered, exposure_valid=exposure,
                    mode=mode, device="cuda", pair_block=262_144, inner_cap=80, cg_cap=80,
                    profile_warm_start=True, known_e=None)
            else:
                obj = SharedCaptureObjective(
                    d, model_id, weights=w, mode=mode, device="cuda", pair_block=262_144,
                    inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=None)
            objective_state["objective"] = obj
        return objective_state["objective"]

    formal_controller._objective = factory
    formal_controller.RUN = RUN
    row = {
        "fit_id": fit_id, "model_id": "G", "kind": "real", "fixture": None,
        "candidate": "random", "start_name": "051-%s" % args.condition,
        "objective_variant": "G/full-J/raw/%s" % args.condition,
        "weights": weights,
        "stages": [{"stage": args.stage, "bin_size_bp": STAGE_BIN[args.stage], "fg_cap": fg_cap}],
    }
    stage_spec = row["stages"][0]
    formal_controller._event({"event": "stage_start", "fit_id": fit_id, "stage": args.stage,
                              "condition": args.condition, "fg_cap": fg_cap, "bend": weights["bend"],
                              "reference_opened": False, "phase_opened": False})
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    record = formal_controller.stage_fit(row, stage_spec, data, coordinates, p_init, q_init, None,
                                        initial_raw_y=raw_y)
    wall = time.perf_counter() - t0
    stage_path = RUN / "stages" / fit_id / ("%s.json" % args.stage)
    terminal_path = RUN / "logs" / ("%s-%s.terminal.json" % (fit_id, args.stage))
    terminal = {
        "schema": "p9016-round051-stage-terminal-v1",
        "fit_id": fit_id, "condition": args.condition, "stage": args.stage,
        "bin_size_bp": STAGE_BIN[args.stage], "fg_cap": fg_cap, "weights": weights,
        "status": record.get("status"), "terminal_reason": record.get("terminal_reason"),
        "outer_fg_actual": record.get("outer_fg_actual"),
        "last_accepted_endpoint": record.get("last_accepted_endpoint"),
        "wall_seconds": float(wall), "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs),
        "raw_records": float(data.raw_records),
        "raw_cis_offdiag": float(data.raw_cis_offdiag), "raw_inter": float(data.raw_inter),
        "start_coordinates_sha256": round_runner.array_sha256(coordinates),
        "start_raw_y_sha256": round_runner.array_sha256(raw_y),
        "start_sphere_mapping_max_abs_error": sphere_error,
        "endpoint": (record.get("endpoint") or {}),
        "artifact_hashes": record.get("artifact_hashes"),
        "stage_record_path": str(stage_path.relative_to(ROOT)),
        "stage_record_sha256": round_runner.sha256_file(stage_path) if stage_path.is_file() else None,
        "reference_opened": False, "phase_opened": False,
        "termination_semantics": "exit code 0 only means the runner finished; scientific status is "
                                 "status/terminal_reason and the canonical raw-y/q gradient",
    }
    if args.condition == "C" and record.get("status") != "failure":
        # 导出只写真实存在的珠子：缺失珠子不写行，原基因组位置（bp）保留，不压缩间隔。
        export_path = out_root / fit_id / ("%s.3dg" % args.stage)
        dropped = filter_export_beads(export_path, mask, tuple(data.chromosome_lengths))
        terminal["exported_beads"] = dropped
        terminal["artifact_hashes"] = dict(terminal.get("artifact_hashes") or {})
        terminal["artifact_hashes"]["coordinate_3dg_sha256_exported"] = round_runner.sha256_file(export_path)
        obj = objective_state["objective"]
        terminal["support_audit"] = obj.support_audit()  # type: ignore[attr-defined]
        terminal["support_data"] = objective_state.get("audit")
    round_runner.write_json(terminal_path, terminal)
    formal_controller._event({"event": "stage_end", "fit_id": fit_id, "stage": args.stage,
                              "status": terminal["status"], "outer_fg_actual": terminal["outer_fg_actual"]})
    print(json.dumps({key: terminal.get(key) for key in (
        "fit_id", "stage", "status", "terminal_reason", "outer_fg_actual",
        "last_accepted_endpoint", "wall_seconds")}, indent=2))
    return 0 if record.get("status") != "failure" else 2


def filter_export_beads(path: Path, mask, chromosome_lengths) -> dict:
    """把 C 的 3DG 导出裁成只含真实存在的珠子（保留原始 bp 位置）。"""
    import re
    names = ["chr%d" % (index + 1) for index in range(len(chromosome_lengths))]
    pattern = re.compile(r"^(chr\d+)\((mat|pat)\)$")
    kept = 0
    dropped = 0
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            lines.append(line)
            continue
        match = pattern.match(fields[0])
        if match is None:
            lines.append(line)
            continue
        chromosome = names.index(match.group(1))
        copy = 0 if match.group(2) == "mat" else 1
        bin_size = 1_000_000
        index = int(fields[1]) // bin_size
        if mask[copy, index]:
            lines.append(line)
            kept += 1
        else:
            dropped += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"kept_rows": kept, "dropped_rows": dropped,
            "rule": "missing beads are not exported; original bp positions kept, gaps not compressed"}


def baseline_check() -> int:
    """只做 baseline 复现断言：046 已发布 G-random 1Mb endpoint 的 hash。"""
    npz_sha = round_runner.sha256_file(BASELINE_NPZ)
    d3g_sha = round_runner.sha256_file(BASELINE_3DG)
    ok = npz_sha == BASELINE_NPZ_SHA and d3g_sha == BASELINE_3DG_SHA
    print(json.dumps({"baseline_npz": BASELINE_NPZ_SHA, "measured_npz": npz_sha,
                      "baseline_3dg": BASELINE_3DG_SHA, "measured_3dg": d3g_sha, "match": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--baseline-check" in sys.argv:
        raise SystemExit(baseline_check())
    raise SystemExit(main())
