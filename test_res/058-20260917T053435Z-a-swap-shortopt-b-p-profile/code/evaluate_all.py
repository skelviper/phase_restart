"""058 哈希门后的统一 dev、055 reference 与 phase provenance 评价。"""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S056 = ROOT / "test_res/056-20260916T152353Z-pro-review-experiments"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io
from pr.solver_state import load_solver_state, sha256_file
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective

MASK = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz"
REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
RAW_PAIRS = ROOT / "data/P9016.pairs.gz"
SNPFREE = ROOT / "inputs/P9016.snpfree.pairs.gz"
MASK_SHA = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_SHA = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
RAW_SHA = "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505"
SNPFREE_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
DOMAIN = b"P9016-contact-fold-v1\0"; SEED = 560301; THRESHOLD = 14757395258967642112
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def endpoints():
    A = json.loads((RUN / "logs/A_terminal.json").read_text())
    B = json.loads((RUN / "logs/B_terminal.json").read_text())
    rows = []
    for row in A["arms"]:
        rows.append({"branch": "A", "endpoint_id": row["arm_id"], "seed": row["seed"],
                     "role": row["arm"], "candidate": row["candidate"],
                     "state_path": row["artifacts"]["solver_state"]["path"]})
    for row in B["endpoints"]:
        rows.append({"branch": "B", "endpoint_id": row["endpoint_id"], "seed": row["seed"],
                     "role": row["role"], "candidate": None,
                     "state_path": row["artifacts"]["solver_state"]["path"]})
    if len(rows) != 12:
        raise RuntimeError("expected 12 finite endpoints")
    return rows


def state_path(row):
    path = Path(row["state_path"])
    return path if path.is_absolute() else ROOT / path


def make_dev_data():
    train = data_io.load_aggregate(S056 / "inputs/G-original_1000000_aggregate.npz")
    with np.load(S056 / "inputs/test_contacts.npz", allow_pickle=False) as payload:
        dev = __import__("pr.contact_model", fromlist=["aggregate_from_arrays"]).aggregate_from_arrays(
            train.chromosome_names, train.chromosome_lengths,
            payload["ci"], payload["p1"], payload["cj"], payload["p2"], 1_000_000)
    dev = replace(dev, exposure=np.asarray(train.exposure).copy(), exposure_mode="train_fixed_G-original")
    dev.assert_consistent()
    if int(dev.raw_cis_offdiag + dev.raw_inter) != 253067:
        raise RuntimeError("development Noff mismatch")
    return train, dev


def dev_score(dev, coords, p):
    obj = SharedCaptureObjective(dev, "G", weights=PenaltyWeights(),
                                 mode="V0-fixed-production-e", device="cuda",
                                 pair_block=262_144, inner_cap=80, cg_cap=80,
                                 profile_warm_start=True, known_e=None)
    raw = torch.as_tensor(__import__("pr.contact_model", fromlist=["sphere_inverse"]).sphere_inverse(coords),
                          dtype=torch.float64, device="cuda")
    x = obj._physics._map_raw(raw); pt = torch.as_tensor(float(p), dtype=torch.float64, device="cuda")
    z = [torch.zeros((), dtype=torch.float64, device="cuda") for _ in range(4)]
    obs = [torch.zeros((), dtype=torch.float64, device="cuda") for _ in range(4)]
    eligible = [0, 0, 0, 0]
    pair_i = np.asarray(dev.pair_i); pair_j = np.asarray(dev.pair_j)
    cis_np = np.asarray(dev.cis_pair)
    separation = np.abs(np.asarray(dev.locus_bin)[pair_j] - np.asarray(dev.locus_bin)[pair_i]) * int(dev.bin_size)
    group_np = np.full(dev.n_pairs, 3, dtype=np.int8)
    group_np[cis_np & (separation < 5_000_000)] = 0
    group_np[cis_np & (separation >= 5_000_000) & (separation < 20_000_000)] = 1
    group_np[cis_np & (separation >= 20_000_000)] = 2
    counts_np = np.asarray(dev.counts, dtype=np.float64)
    record_counts = [float(counts_np[group_np == g].sum()) for g in range(4)]
    eligible = [int(np.count_nonzero(group_np == g)) for g in range(4)]
    for start in range(0, dev.n_pairs, obj.pair_block):
        stop = min(start + obj.pair_block, dev.n_pairs)
        i = obj._pair_i[start:stop]; j = obj._pair_j[start:stop]
        kernels, _ = obj._kernel_block(x, i, j, pt, with_gradient=False)
        kaa, kab, kba, kbb = kernels[:4]
        local_cis = obj._cis[start:stop]
        eprod = obj._fixed_e[i] * obj._fixed_e[j]
        rate = eprod * (torch.where(local_cis, 0.5 * pt, torch.as_tensor(0.25, device="cuda")) * (kaa + kbb)
                        + torch.where(local_cis, 0.5 * (1.0 - pt), torch.as_tensor(0.25, device="cuda")) * (kab + kba))
        counts = obj._counts[start:stop]
        groups = group_np[start:stop]
        for g in range(4):
            mask_np = groups == g
            if not np.any(mask_np):
                continue
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device="cuda")
            z[g] = z[g] + rate[mask].sum()
            positive = mask & (counts > 0)
            if bool(torch.any(positive)):
                obs[g] = obs[g] + (counts[positive] * torch.log(rate[positive])).sum()
    obj.synchronize()
    zg = [float(v.item()) for v in z]; og = [float(v.item()) for v in obs]
    n_off = float(sum(record_counts)); zall = float(sum(zg)); observed = float(sum(og))
    global_nll = (n_off * math.log(zall) - observed) / n_off
    labels = ("cis_1_5Mb", "cis_5_20Mb", "cis_20Mb_inf", "inter")
    groups = {}
    contributions = []
    for name, ng, zz, oo, ne in zip(labels, record_counts, zg, og, eligible):
        contribution = (-oo + ng * math.log(zall)) / n_off
        conditional = (-oo + ng * math.log(zz)) / ng if ng > 0 else None
        groups[name] = {"records": int(ng), "eligible_pairs": ne, "Z": zz,
                        "observed_C_log_rate": oo, "global_contribution": contribution,
                        "within_group_conditional_nll": conditional}
        contributions.append(contribution)
    cis_records = sum(record_counts[:3]); cis_eligible = sum(eligible[:3])
    if cis_records != float(dev.raw_cis_offdiag) or cis_eligible != int(dev.cis_pair.sum()):
        raise RuntimeError("cis distance groups do not partition full cis offdiag")
    reconstruction_error = abs(sum(contributions) - global_nll)
    if reconstruction_error > 1e-12:
        raise RuntimeError("group contributions do not reconstruct global NLL")
    return {"primary_no_K0_nll_per_offdiag": global_nll, "Noff": int(n_off), "Zall": zall,
            "groups": groups, "cis_partition": {"records": int(cis_records),
                                                  "eligible_pairs": cis_eligible},
            "contribution_reconstruction_abs_error": reconstruction_error,
            "equivalents": 1}


def parse_reference():
    tracks = {}
    with gzip.open(REFERENCE, "rt") as handle:
        for line in handle:
            f = line.split()
            if len(f) == 5:
                tracks.setdefault(f[0], {})[int(f[1])] = np.asarray(f[2:5], dtype=float)
    return tracks


def pearson(a, b):
    a = np.asarray(a); b = np.asarray(b); a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def dists(x, i, j):
    d = x[i] - x[j]; return np.sqrt(np.sum(d * d, axis=1))


def structural_score(data, coords, reference, legacy):
    rows = []
    for ci, chrom in enumerate(CHROMS):
        positions = np.asarray(legacy[f"chr{ci}_positions"], dtype=np.int64)
        pi = np.asarray(legacy[f"chr{ci}_pair_i"], dtype=np.int64)
        pj = np.asarray(legacy[f"chr{ci}_pair_j"], dtype=np.int64)
        common = np.asarray(legacy[f"chr{ci}_common"], dtype=bool)
        slc = data.chromosome_slice(ci)
        candidate = coords[:, slc.start + positions // data.bin_size]
        ref = []
        for suffix in ("mat", "pat"):
            table = reference[f"{chrom}({suffix})"]
            arr = np.asarray([table.get(int(pos), [np.nan]*3) for pos in positions], dtype=float)
            ref.append(arr)
        finite = common.copy()
        for arr in (candidate[0], candidate[1], ref[0], ref[1]):
            finite &= np.isfinite(arr[pi]).all(1) & np.isfinite(arr[pj]).all(1)
        if not np.array_equal(finite, common):
            raise RuntimeError(f"055 support dropped {chrom}")
        i = pi[common]; j = pj[common]
        da, db, dm, dp = dists(candidate[0], i, j), dists(candidate[1], i, j), dists(ref[0], i, j), dists(ref[1], i, j)
        rho = {"A_mat": pearson(da, dm), "A_pat": pearson(da, dp),
               "B_mat": pearson(db, dm), "B_pat": pearson(db, dp)}
        direct = (rho["A_mat"] + rho["B_pat"]) / 2
        swapped = (rho["A_pat"] + rho["B_mat"]) / 2
        orientation = "unresolved_tie" if abs(direct-swapped) <= 1e-12 else ("direct" if direct > swapped else "swapped")
        if orientation == "direct":
            ma, mb, same, cross = rho["A_mat"]-rho["A_pat"], rho["B_pat"]-rho["B_mat"], direct, swapped
        elif orientation == "swapped":
            ma, mb, same, cross = rho["A_pat"]-rho["A_mat"], rho["B_mat"]-rho["B_pat"], swapped, direct
        else:
            ma, mb = rho["A_mat"]-rho["A_pat"], rho["B_pat"]-rho["B_mat"]
            same = cross = (direct+swapped)/2
        rows.append({"chromosome": chrom, "n_pairs": int(common.sum()), **rho,
                     "direct": direct, "swapped": swapped, "orientation": orientation,
                     "same": same, "cross": cross, "contrast": same-cross,
                     "margin_A": ma, "margin_B": mb, "min_margin": min(ma, mb)})
    if sum(row["n_pairs"] for row in rows) != 157529:
        raise RuntimeError("055 pair total mismatch")
    fields = ("same", "cross", "contrast", "margin_A", "margin_B", "min_margin")
    return {"per_chromosome": rows,
            "macro": {field: float(np.mean([row[field] for row in rows])) for field in fields}}


def return_to_control(data, control, swap, chromosome_index):
    slc = data.chromosome_slice(chromosome_index)
    c = control[:, slc]; s = swap[:, slc]
    options = []
    for orientation, candidate in (("direct", s), ("swapped", s[::-1])):
        displacement = np.linalg.norm(candidate - c, axis=2)
        options.append({"orientation": orientation,
                        "RMSD_over_l0": float(np.sqrt(np.mean(displacement**2)) / data.l0),
                        "max_bead_displacement_over_l0": float(displacement.max() / data.l0)})
    best = min(options, key=lambda row: (row["RMSD_over_l0"], row["orientation"]))
    best["near_control"] = bool(best["RMSD_over_l0"] <= 0.1
                                and best["max_bead_displacement_over_l0"] <= 0.5)
    best["no_alignment"] = True
    return best


def strand_code(value):
    if value == "+": return 1
    if value == "-": return 2
    return 128 + value.encode("ascii")[0]


def endpoint_bytes(ci, pos, strand):
    return struct.pack(">HQB", int(ci), int(pos), strand_code(strand))


def fold_hash(a, b):
    left, right = endpoint_bytes(*a), endpoint_bytes(*b)
    if right < left: left, right = right, left
    return int.from_bytes(hashlib.blake2b(DOMAIN + struct.pack(">Q", SEED) + left + right,
                                         digest_size=8).digest(), "big")


def phase_audit():
    # Source/docs establish hard phase fields and mapping, but no independent direct-vs-imputed origin.
    provenance = {"label_type": "hard_phase0_phase1_fields_origin_unqualified",
                  "direct_origin_established": False,
                  "reason": "no independent provenance distinguishes direct SNP calls from inferred/imputed hard labels"}
    names = []
    with gzip.open(SNPFREE, "rt") as handle:
        for line in handle:
            if line.startswith("#chromosome:"):
                names.append(line.split()[1])
    index = {name: i for i, name in enumerate(names)}
    counts = {"raw_rows": 0, "seven_column_identity_matches": 0, "dev_rows": 0,
              "dev_cis": 0, "dev_cis_offdiag": 0, "dev_inter": 0,
              "dev_cis_offdiag_same_copy_known": 0, "dev_cis_offdiag_cross_copy_known": 0,
              "dev_cis_offdiag_unknown_label": 0,
              "dev_same_copy_distance_groups": {"cis_1_5Mb": 0, "cis_5_20Mb": 0, "cis_20Mb_inf": 0},
              "dev_far_same_copy_distinct_blocks": 0}
    far_blocks = set(); raw_columns = None; snp_columns = None
    with gzip.open(RAW_PAIRS, "rt") as raw, gzip.open(SNPFREE, "rt") as snp:
        def records(handle, raw_mode):
            nonlocal raw_columns, snp_columns
            for line in handle:
                if line.startswith("#columns:"):
                    cols = line.rstrip().split(":",1)[1].split("\t")
                    if raw_mode: raw_columns = cols
                    else: snp_columns = cols
                elif not line.startswith("#"):
                    yield line.rstrip("\n").split("\t")
        for row_index, pair in enumerate(zip(records(raw, True), records(snp, False))):
            rr, ss = pair
            counts["raw_rows"] += 1
            if rr[:7] != ss or len(ss) != 7 or len(rr) < 9:
                raise RuntimeError(f"raw/SNP-free row identity mismatch at {row_index}")
            counts["seven_column_identity_matches"] += 1
            ci, cj = index[ss[1]], index[ss[3]]; p1, p2 = int(ss[2]), int(ss[4])
            is_dev = fold_hash((ci, p1, ss[5]), (cj, p2, ss[6])) >= THRESHOLD
            ph1, ph2 = rr[7], rr[8]
            if ci == cj and p1 > p2:
                p1, p2, ph1, ph2 = p2, p1, ph2, ph1
            if not is_dev:
                continue
            counts["dev_rows"] += 1
            if ci != cj:
                counts["dev_inter"] += 1; continue
            counts["dev_cis"] += 1
            b1, b2 = p1 // 1_000_000, p2 // 1_000_000
            if b1 == b2:
                continue
            counts["dev_cis_offdiag"] += 1
            known = ph1 in ("0","1") and ph2 in ("0","1")
            if known and ph1 == ph2:
                counts["dev_cis_offdiag_same_copy_known"] += 1
                sep = abs(b2-b1) * 1_000_000
                group = "cis_1_5Mb" if sep < 5_000_000 else ("cis_5_20Mb" if sep < 20_000_000 else "cis_20Mb_inf")
                counts["dev_same_copy_distance_groups"][group] += 1
                if sep >= 20_000_000:
                    far_blocks.add((ci, min(b1,b2), max(b1,b2)))
            elif known:
                counts["dev_cis_offdiag_cross_copy_known"] += 1
            else:
                counts["dev_cis_offdiag_unknown_label"] += 1
    counts["dev_far_same_copy_distinct_blocks"] = len(far_blocks)
    if counts["raw_rows"] != 1_703_888 or counts["dev_rows"] != 340_715:
        raise RuntimeError("phase full-row/dev fold count mismatch")
    return {"provenance": provenance, "raw_columns": raw_columns,
            "snpfree_columns": snp_columns, "alignment": counts,
            "R1_direct": None, "R1_direct_reason": provenance["reason"],
            "identity_regression_veto": None,
            "identity_regression_veto_reason": "R1_direct unavailable; inferred/unqualified labels are not substituted",
            "random_copy_control_expectation": 0.5,
            "bootstrap_performed": False}


def main():
    started = time.perf_counter()
    gate = json.loads((RUN / "results/pre_evaluation_hash_gate.json").read_text())
    if gate.get("status") != "PASS" or not gate.get("all_pending_states_hashed"):
        raise RuntimeError("pre-evaluation hash gate missing")
    for path, expected in ((MASK, MASK_SHA), (REFERENCE, REFERENCE_SHA),
                           (RAW_PAIRS, RAW_SHA), (SNPFREE, SNPFREE_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"evaluation input hash mismatch {path}")
    inventory = endpoints(); train, dev = make_dev_data()
    dev_results = {}; states = {}
    auxiliary = 2  # B diagnostic cache builds already completed.
    for row in inventory:
        state = load_solver_state(state_path(row)); states[row["endpoint_id"]] = state
        score = dev_score(dev, state["coordinates"], state["p"])
        dev_results[row["endpoint_id"]] = score; auxiliary += 1
        print(json.dumps({"event": "dev_score", "endpoint": row["endpoint_id"],
                          "nll": score["primary_no_K0_nll_per_offdiag"]}), flush=True)
    if auxiliary != 14:
        raise RuntimeError("nonreserve auxiliary equivalent count mismatch")
    dump(RUN / "evaluation/dev_results.json", {"schema": "p9016-058-dev-v1",
                                                "results": dev_results,
                                                "auxiliary_equivalents_including_B_cache": auxiliary})

    reference = parse_reference()
    with np.load(MASK, allow_pickle=False) as payload:
        legacy = {key: payload[key] for key in payload.files}
    structural = {}
    for row in inventory:
        structural[row["endpoint_id"]] = structural_score(train, states[row["endpoint_id"]]["coordinates"],
                                                           reference, legacy)
    returns = {}
    for row in inventory:
        if row["branch"] != "A" or row["role"] != "Swap": continue
        control_id = row["endpoint_id"].replace("-Swap", "-Control")
        returns[row["endpoint_id"]] = return_to_control(
            train, states[control_id]["coordinates"], states[row["endpoint_id"]]["coordinates"],
            int(row["candidate"]["chromosome_index"]))
    dump(RUN / "evaluation/structural_results.json", {"schema": "p9016-058-structural-v1",
                                                       "results": structural,
                                                       "return_to_control": returns})
    phase = phase_audit()
    dump(RUN / "evaluation/phase_traceability.json", phase)

    comparisons = []
    for row in inventory:
        if row["branch"] == "A" and row["role"] == "Swap":
            left = row["endpoint_id"].replace("-Swap", "-Control"); right = row["endpoint_id"]
        elif row["branch"] == "B" and row["role"] == "Profile":
            left = row["endpoint_id"].replace("-Profile", "-Original"); right = row["endpoint_id"]
        else:
            continue
        dleft, dright = dev_results[left], dev_results[right]
        sleft, sright = structural[left], structural[right]
        groups = {g: dleft["groups"][g]["global_contribution"] - dright["groups"][g]["global_contribution"]
                  for g in dleft["groups"]}
        comparisons.append({"branch": row["branch"], "seed": row["seed"],
                            "candidate": row["candidate"], "control": left, "candidate_endpoint": right,
                            "dev_gain": dleft["primary_no_K0_nll_per_offdiag"] - dright["primary_no_K0_nll_per_offdiag"],
                            "dev_gain_contributions": groups,
                            "same_delta": sright["macro"]["same"] - sleft["macro"]["same"],
                            "cross_delta": sright["macro"]["cross"] - sleft["macro"]["cross"],
                            "contrast_delta": sright["macro"]["contrast"] - sleft["macro"]["contrast"],
                            "min_margin_delta": sright["macro"]["min_margin"] - sleft["macro"]["min_margin"],
                            "return_to_control": returns.get(right),
                            "R1_direct": None, "identity_status": "identity_unverified"})
    dump(RUN / "evaluation/comparisons.json", {"schema": "p9016-058-comparisons-v1",
                                                "comparisons": comparisons})
    terminal = {"schema": "p9016-058-evaluation-terminal-v1", "status": "complete",
                "endpoint_count": len(inventory), "development_equivalents": 12,
                "B_cache_equivalents": 2, "nonreserve_auxiliary_equivalents": auxiliary,
                "cap": 64, "remaining_validation_reserve": 64-auxiliary,
                "reference_opened_after_hash_gate": True,
                "phase_body_opened_after_hash_gate": True,
                "elapsed_seconds": time.perf_counter() - started}
    dump(RUN / "logs/evaluation_terminal.json", terminal)
    print(json.dumps(terminal, sort_keys=True))


if __name__ == "__main__":
    main()
