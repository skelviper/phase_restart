#!/usr/bin/env python
"""049 评价 prep 的**隔离代码夹具**：用合成端点跑通 manifest/selection/target 适配、null 生成、
gate 写入与 baseline-null 复用核对。输出目录为 ``evaluation/codecheck/``，不写入正式 gate/nulls，
不代表任何科学结论。真实 gate 只能用训练侧真实产物运行。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_lib as lib  # noqa: E402

HERE = lib.EVAL
CHECK = HERE / "codecheck"
COORDS = CHECK / "coords"
FITS_JSON = CHECK / "results/fits"
STAGES = CHECK / "stages"
NULLS = CHECK / "nulls"
PYTHON = "/mnt/ssd/zliu/miniforge3/envs/analysis/bin/python"

# 设计好的 count 关系：A-raw 内为 1e-10 tie → 取 consensus；其余按最小 count 选
COUNTS = {
    "A": {"raw": {"consensus": 100.0, "random": 100.0 + 1e-10}, "ms": {"consensus": 101.0, "random": 99.0}},
    "B": {"raw": {"consensus": 95.0, "random": 97.0}, "ms": {"consensus": 94.0, "random": 96.0}},
    "C": {"raw": {"consensus": 90.0, "random": 88.0}, "ms": {"consensus": 89.0, "random": 87.0}},
}
SELECTED = {"A-raw": "consensus", "A-ms": "random", "B-raw": "consensus", "B-ms": "consensus",
            "C-raw": "random", "C-ms": "random"}


def write_3dg(path: Path, coords: np.ndarray, data: lib.Aggregate) -> None:
    lines = []
    for ci in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(ci)
        positions = data.locus_bin[slc] * int(data.bin_size)
        for copy, suffix in enumerate(("a", "b")):
            track = "c%02d%s" % (ci + 1, suffix)
            for local, position in enumerate(positions):
                point = coords[copy, slc.start + local]
                lines.append("%s\t%d\t%.17g\t%.17g\t%.17g" % (track, int(position), point[0], point[1], point[2]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def initial_record(name: str, data: lib.Aggregate) -> dict:
    """为 initial 生成 049 侧配套 3DG（candidate 命名），用于验证新增的 readback 路径。"""
    payload = lib.load_coords_npz(lib.INITIAL_NPZ[name])
    tdg_path = CHECK / ("initial-%s.3dg" % name)
    write_3dg(tdg_path, payload["coordinates"], data)
    return {"fit_id": "initial-%s" % name, "source": name,
            "coords_npz_path": str(lib.INITIAL_NPZ[name]),
            "coords_npz_sha256": lib.INITIAL_NPZ_SHA256[name],
            "three_dg_path": str(tdg_path), "three_dg_sha256": lib.sha256_file(tdg_path),
            "terminal": "initial_zero_optimization",
            # 控制记录也带完整元数据：parser 必须保留而不是丢掉
            "p": 0.75, "q": 0.0, "q_source": "q_from_p(p_init=0.75)", "fullJ": 12.0 + len(name),
            "count_nll_by_loss": {loss: 1.0 for loss in lib.LOSES}, "outerFG": 0, "fit_wall_seconds": 0.0}


def main() -> int:
    CHECK.mkdir(parents=True, exist_ok=True)
    # 只清理本夹具自己的子目录，避免删掉 codecheck/plots 与 codecheck/report 等其他自检产物
    for directory in (COORDS, FITS_JSON, STAGES, NULLS):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    data = lib.Aggregate()
    fits = []
    for fit_id in lib.EXPECTED_FIT_IDS:
        loss, solver, source = fit_id.split("-")
        seed = 900 + lib.EXPECTED_FIT_IDS.index(fit_id)
        rng = np.random.default_rng(seed)
        coords = rng.normal(size=(2, lib.N_LOCI, 3)) * 0.03
        lib.assert_inside_unit_ball(coords)
        npz_path = COORDS / fit_id / "1Mb.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        p_value, q_value = 0.75, 0.0
        np.savez_compressed(npz_path, coordinates=coords, raw_y=coords.copy(), theta=np.zeros(6 * lib.N_LOCI + 1),
                            p=np.asarray(p_value), q=np.asarray(q_value))
        tdg_path = npz_path.with_name("1Mb.3dg")
        write_3dg(tdg_path, coords, data)
        terminal = "budget_not_converged" if seed % 2 else "converged"
        record = {
            # 故意使用父侧给定的别名字段，验证 adapter 不会静默丢字段
            "fit_id": fit_id, "loss_id": loss, "solver_id": solver, "source_id": source,
            "coords_npz_path": str(npz_path), "coords_npz_sha256": lib.sha256_file(npz_path),
            "three_dg_path": str(tdg_path), "three_dg_sha256": lib.sha256_file(tdg_path),
            "status": terminal, "terminal": terminal, "outer_fg_actual": 1502,
            "count_nll_by_loss": {name: COUNTS[name][solver][source] for name in lib.LOSES},
            "fullJ_value": 10.0 + seed, "p": p_value, "q": q_value,
            "fit_wall_seconds": 100.0 + seed, "canonical_gradient_max_abs": 1e-4 * seed,
            "coordinate_array_sha256": lib.hash_array(coords), "kind": "code_fixture",
        }
        fits.append(record)
        lib.write_json(FITS_JSON / ("%s.json" % fit_id), record)
        lib.write_json(STAGES / fit_id / "1Mb.json", record)

    manifest = {"schema": "p9016-049-endpoint-manifest-code-fixture", "run_id": lib.RUN.name,
                "kind": "code_fixture", "fits": fits,
                "initials": [initial_record(name, data) for name in lib.SOURCES],
                "baseline": {"fit_id": "baseline-046-real-extension-G-full-J",
                             "coords_npz_path": str(lib.BASELINE_NPZ),
                             "coords_npz_sha256": lib.BASELINE_NPZ_SHA256,
                             "three_dg_path": str(lib.BASELINE_3DG),
                             "three_dg_sha256": lib.BASELINE_3DG_SHA256,
                             "terminal": "budget_not_converged", "p": 0.75, "q": 0.0,
                             "q_source": "q_from_p(p_init=0.75)", "fullJ": 9.0,
                             "count_nll_by_loss": {loss: 2.0 for loss in lib.LOSES}, "outerFG": 1988}}
    manifest_path = CHECK / "endpoint_manifest_pre_reference.json"
    lib.write_json(manifest_path, manifest)

    per_loss = {}
    for loss in lib.LOSES:
        ss = {}
        for solver in lib.SOLVERS:
            counts = COUNTS[loss][solver]
            selected = SELECTED["%s-%s" % (loss, solver)]
            ss[solver] = {"selected_source": selected, "rule": "min own count; consensus on <=1e-9 tie",
                          "candidates": counts,
                          "margin": abs(counts["consensus"] - counts["random"])}
        display_fit = min(("%s-%s-%s" % (loss, s, v) for s in lib.SOLVERS for v in lib.SOURCES),
                          key=lambda fid: COUNTS[loss][fid.split("-")[1]][fid.split("-")[2]])
        display_solver, display_source = display_fit.split("-")[1], display_fit.split("-")[2]
        display_coords = next(row["coordinate_array_sha256"] for row in fits if row["fit_id"] == display_fit)
        per_loss[loss] = {"source_selection": ss,
                          "display_endpoint": {"solver": display_solver, "source": display_source,
                                               "count": COUNTS[loss][display_solver][display_source],
                                               "coordsSHA": display_coords}}
    selection = {"schema": "p9016-049-selection-pre-reference-code-fixture", "kind": "code_fixture",
                 "reference_opened": False, "per_loss": per_loss}
    selection_path = CHECK / "selection_pre_reference.json"
    lib.write_json(selection_path, selection)

    gate = CHECK / "gate.json"
    command = [PYTHON, str(HERE / "source/eval049_prep.py"),
               "--manifest", str(manifest_path), "--selection", str(selection_path),
               "--fits-dir", str(FITS_JSON), "--stages-dir", str(STAGES),
               "--nulls-dir", str(NULLS), "--gate-path", str(gate),
               "--index-path", str(CHECK / "index.json"),
               "--reg-expected-path", str(CHECK / "baseline_regression_expected.json"),
               "--code-fixture"]
    completed = subprocess.run(command, capture_output=True, text=True)
    (CHECK / "prep_codecheck.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (CHECK / "prep_codecheck.stderr.log").write_text(completed.stderr, encoding="utf-8")
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr.strip()[-4000:], file=sys.stderr)
        print("CODECHECK FAIL exit=%d" % completed.returncode, file=sys.stderr)
        return completed.returncode
    payload = json.loads(gate.read_text(encoding="utf-8"))
    null_files = sorted(NULLS.glob("*.npz"))
    assert payload["status"] == "PASS" and payload["code_fixture"] is True
    assert len(payload["nulls_new"]) == 136, len(payload["nulls_new"])
    assert len(null_files) == 136, len(null_files)
    assert payload["reference_first_opened_utc"] is None and payload["reference_opened"] is False
    assert len(payload["nulls_reused_baseline"]) == 17
    # parser 字段保留：控制记录（initial/baseline）不得丢掉 p/q/fullJ/count_nll_by_loss/outerFG
    import eval049_inputs as inputs_mod
    normalized = inputs_mod.load_manifest(Path(manifest_path))
    for row in normalized["initials"]:
        for field in ("p", "q", "q_source", "full_j", "count_nll_by_loss", "outer_fg", "wall_s"):
            assert row.get(field) is not None, (row["fit_id"], field)
    base_row = normalized["baseline"]
    for field in ("p", "q", "q_source", "full_j", "count_nll_by_loss", "outer_fg"):
        assert base_row.get(field) is not None, ("baseline", field)
    for record in payload["initials"]:
        meta = record.get("manifest_metadata") or {}
        assert meta.get("p") is not None and meta.get("full_j") is not None, record["initial"]
    assert (payload["baseline"].get("manifest_metadata") or {}).get("outer_fg") == 1988
    selected = {row["source_candidate_id"] for row in payload["nulls_new"]}
    assert len(selected) == 8, selected
    # 路径解析：用训练侧已完成的真实端点（只读）验证四种相对/绝对写法
    real = lib.resolve_path("coords/A-raw-consensus/1Mb.npz")
    assert real.is_file() and real.suffix == ".npz" and real.parent.name == "A-raw-consensus", real
    for text in (str(real),
                 "test_res/%s/coords/A-raw-consensus/1Mb.npz" % lib.RUN.name,
                 "%s/coords/A-raw-consensus/1Mb.npz" % lib.RUN.name,
                 "coords/A-raw-consensus/1Mb.npz"):
        assert lib.resolve_path(text) == real, text
    assert lib.resolve_path("stages/A-raw-consensus/1Mb.json") == (lib.RUN / "stages/A-raw-consensus/1Mb.json")
    assert lib.resolve_path("results/fits/A-raw-consensus.json") == (lib.RUN / "results/fits/A-raw-consensus.json")

    # 元数据传播 + common G（不重算，纯消费字段）
    import eval049_evaluate as ev_mod
    sample = {"fit_id": "A-raw-consensus", "loss_id": "A", "solver_id": "raw", "source_id": "consensus",
              "status": "budget_not_converged", "coords_npz_path": "coords/A-raw-consensus/1Mb.npz",
              "coords_npz_sha256": "x", "three_dg_path": "coords/A-raw-consensus/1Mb.3dg", "three_dg_sha256": "y",
              "outer_fg_actual": 1502, "fullJ_value": 9.6, "p": 0.93, "q": 2.66, "fit_wall_seconds": 545.5,
              "canonical_gradient_max_abs": 5.3e-4, "count_nll_by_loss": {"A": 9.5722, "B": 9.6, "C": 9.7},
              "fullJ_A": 9.585, "fullJ_B": 9.61, "fullJ_C": 9.72}
    normalized_sample = inputs_mod.normalize_record(sample)
    meta = ev_mod.metadata_of(normalized_sample)
    assert meta["wall_s"] == 545.5 and meta["canonical_grad"] == 5.3e-4
    assert meta["p"] == 0.93 and meta["q"] == 2.66 and meta["outer_fg"] == 1502
    assert abs(meta["own_count_nll"] - 9.5722) < 1e-12 and abs(meta["common_G_count"] - 9.5722) < 1e-12
    assert abs(meta["common_G_fullJ"] - 9.585) < 1e-12 and meta["common_G_fields_present"] is True
    # 控制记录（initial/baseline）同样保留元数据，不被 None 覆盖
    control = inputs_mod.normalize_record({"fit_id": "initial-consensus", "source": "consensus",
                                           "terminal": "initial_zero_optimization", "p": 0.75, "q": 1.0989,
                                           "fullJ_value": 12.5, "count_nll_by_loss": {"A": 11.0},
                                           "outer_fg_actual": 0, "coords_npz_path": "coords/x.npz",
                                           "coords_npz_sha256": "z"}, require_tdg=False,
                                          required_fields=inputs_mod.INITIAL_BASELINE_REQUIRED_FIELDS)
    control_meta = ev_mod.metadata_of(control)
    assert control_meta["p"] == 0.75 and control_meta["own_full_j"] == 12.5
    assert abs(control_meta["common_G_count"] - 11.0) < 1e-12

    print("CODECHECK PASS: prep adapter/null/gate path validated on a synthetic fixture "
          "(no reference, no formal gate written); path resolver + metadata/common-G checked against a real endpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
