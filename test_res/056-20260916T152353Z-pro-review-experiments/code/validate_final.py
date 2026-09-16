"""056终态只读核验并在所有写入结束后生成最终artifact hash清单。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def main():
    checks = []
    def check(name, passed, detail=None):
        checks.append({"check": name, "pass": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError("validation failed: %s: %r" % (name, detail))

    locks = {}
    for lock_name in ("protocol_config.sha256", "protocol_clarifications.sha256"):
        for line in (RUN / lock_name).read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(None, 1)
            relative = relative.strip()
            path = ROOT / relative if relative.startswith("test_res/") else RUN / Path(relative).name
            actual = sha(path)
            locks[str(path.relative_to(ROOT))] = actual
            check("lock:" + relative, actual == expected, {"expected": expected, "actual": actual})
    exp1 = load_json(RUN / "results/experiment1.json")
    exp2 = load_json(RUN / "results/experiment2.json")
    exp3 = load_json(RUN / "results/experiment3_nonreference.json")
    training = load_json(RUN / "logs/training_terminal.json")
    reference = load_json(RUN / "reference_eval/metrics.json")
    final = load_json(RUN / "results/final_status.json")
    check("experiment1_hard_gate", exp1["status"] == "PASS" and not exp1["hard_failures"])
    check("experiment2_hard_gate", exp2["status"] == "PASS" and not exp2["hard_failures"])
    check("training_budget", training["status"] == "complete" and training["outer_fg_actual"] == 6008
          and training["fine_1Mb_fg_actual"] == 1944)
    check("all_stages_explicit_status", all(row["status"] == "budget_not_converged"
          for fit in training["fits"] for row in fit["stages"]))
    check("auxiliary_sweep_exact", exp3["auxiliary_total_full_pair_sweeps"] == 40
          and exp3["auxiliary_regularizer_full_pair_sweeps"] == 0)
    gate = load_json(RUN / "results/pre_reference_hash_gate.json")
    check("pre_reference_gate_inventory", gate["artifact_count"] == 90
          and gate["all_actual_artifact_hashes_matched"] and not gate["reference_opened"])
    for item in gate["artifacts"]:
        check("artifact_gate:" + item["path"], sha(ROOT / item["path"]) == item["sha256"])
    ref_gate = load_json(RUN / "reference_eval/pre_reference_hash_gate.json")
    check("reference_gate", ref_gate["status"] == "PASS" and not ref_gate["reference_opened"])
    check("reference_support", reference["candidate_count"] == 82
          and reference["support"]["pairs_total"] == 157529
          and reference["support"]["shared_across_all_candidates"])
    check("reference_baseline_regression", reference["baseline_055_max_abs_regression_error"] < 1e-10)
    check("final_adoption_decision", final["method_status"] == "offdiag_e_not_adopted_prespecified_gates_failed"
          and final["adoption_recommended"] is False)
    check("final_claim_scope", final["new_l1_claim"] is False
          and final["l2_supported_by_this_round"] is False
          and final["l3_claimed"] is False
          and final["global_method_falsification_claimed"] is False)
    check("optimization_alarm_complete_rule", all(
        row["optimization_imbalance_alarm"]
        == (row["one_converged_other_not"] or row["canonical_gradient_imbalance"])
        for row in final["comparisons"]))
    tie_audit = final["reference_tie_branch_audit"]
    check("reference_tie_branch_no_affected_exp3_rows",
          tie_audit["best_orientation_tie_rows"] == 20
          and tie_audit["experiment3_endpoint_or_splice_rows_affected"] == 0
          and tie_audit["existing_reference_metrics_changed"] is False)
    with (RUN / "results/summary.tsv").open("r", encoding="utf-8") as handle:
        endpoint_rows = list(__import__("csv").DictReader(handle, delimiter="\t"))
    required_endpoint_fields = {"heldout_test_nll_nat_per_offdiag", "pearson_same_macro",
                                "pearson_cross_macro", "mean_margin_A", "mean_margin_B",
                                "mean_min_margin", "canonical_gradient_max_abs_1Mb",
                                "terminal_1Mb", "terminal_reason_1Mb"}
    check("four_endpoint_absolute_summary",
          len(endpoint_rows) == 4 and required_endpoint_fields.issubset(endpoint_rows[0])
          and all(row["terminal_1Mb"] == "budget_not_converged" for row in endpoint_rows))
    with np.load(RUN / "results/experiment2_gradients.npz", allow_pickle=False) as payload:
        check("experiment2_gradient_shapes", len(payload.files) == 12
              and all(payload[key].shape == (2, 2645, 3) for key in payload.files))
    images = {}
    expected = {"experiment1_data_regularizer.png": (1800, 900),
                "experiment2_gradient_response.png": (1800, 900),
                "experiment3_paired_outcomes.png": (2700, 900)}
    for name, size in expected.items():
        path = RUN / "plots" / name
        with Image.open(path) as image:
            extrema = ImageStat.Stat(image.convert("RGB")).extrema
            nonblank = any(high > low for low, high in extrema)
            images[name] = {"size": list(image.size), "mode": image.mode, "nonblank": nonblank}
            check("plot:" + name, image.size == size and nonblank, images[name])
    # All JSON must be strict finite JSON before final hashes are minted.
    json_files = sorted(path for path in RUN.glob("**/*.json")
                        if path.name != "artifact_hashes.json")
    for path in json_files:
        load_json(path)
    check("strict_json_files", True, {"count": len(json_files)})
    validation = {"schema": "p9016-056-final-validation-v1", "status": "PASS",
                  "checks": checks, "plots": images, "locks": locks,
                  "strict_json_count": len(json_files)}
    write_json(RUN / "results/validation.json", validation)
    terminal_path = RUN / "logs/final_terminal.json"
    terminal = load_json(terminal_path)
    terminal["validation"] = {"status": "PASS", "path": "results/validation.json"}
    terminal["independent_parent_acceptance_pending"] = True
    terminal["github_pushed"] = False
    write_json(terminal_path, terminal)
    include_roots = ("code", "inputs", "states", "exports", "coords", "results",
                     "reference_eval", "plots", "logs")
    files = [RUN / "README.md", RUN / "PROTOCOL.md", RUN / "PROTOCOL_CLARIFICATIONS.md",
             RUN / "config.json", RUN / "protocol_config.sha256", RUN / "protocol_clarifications.sha256"]
    for directory in include_roots:
        files.extend(path for path in (RUN / directory).glob("**/*") if path.is_file())
    manifest_path = RUN / "results/artifact_hashes.json"
    files = sorted(set(path for path in files if path != manifest_path))
    artifacts = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                  "sha256": sha(path)} for path in files]
    write_json(manifest_path, {"schema": "p9016-056-final-artifact-hashes-v1",
                               "artifacts": artifacts, "artifact_count": len(artifacts),
                               "self_excluded": True})
    print(json.dumps({"status": "PASS", "checks": len(checks),
                      "artifact_count": len(artifacts), "strict_json_count": len(json_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
