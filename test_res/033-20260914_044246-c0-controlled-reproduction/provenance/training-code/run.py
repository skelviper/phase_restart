#!/usr/bin/env python
"""Thin driver for the P9016 SNP-free diploid reconstruction project.

    run.py prepare                 strip the phase columns into inputs/
    run.py stage1 [options]        Stage 1: the fragment-swap experiment
    run.py split --genome [options]  Stage S0 genome-wide reference points
    run.py reevaluate [options]    hash-verified metrics/figures on existing S0 coordinates
    run.py reconstruct [options]   formal phase-free Reconstruction V1 training

Stage 1 asks whether the label-free held-out score can tell a chromosome-wide
consistent split apart from a locally correct but globally mis-wired one.
Every candidate is refitted on the training fold only and scored on the same
held-out fold. This is an oracle-instrument check, not a blind recovery.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import sys
import time

import numpy as np

from pr import (fdg, folds, genome, gwfdg, pairs7, ref3dg, refeval, s0, score,
                splice, splits)
from pr.gate import EvalGate
from pr.paths import (ALPHAS, BIN, FRAG_BINS, OFF, ROOT, SNPFREE, bin_of,
                      safe_name)

LOG = []


def log(msg):
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    print(line)
    sys.stdout.flush()
    LOG.append(line)


def json_ready(value):
    """Make formal result files strict JSON (no IEEE NaN tokens)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def create_fresh_output_dir(outdir):
    """Create a formal-run directory only when its output path is new."""
    if os.path.lexists(outdir):
        raise FileExistsError("refusing to reuse existing output directory: %s" % outdir)
    os.makedirs(outdir)
    for sub in ("logs", "coords", "work", "plots"):
        os.makedirs(os.path.join(outdir, sub))


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------
def cmd_prepare(args):
    from pr.gate import sha256_file
    info = pairs7.write_snpfree()
    man = {
        "source": "data/P9016.pairs.gz",
        "source_sha256": sha256_file("data/P9016.pairs.gz"),
        "output": os.path.relpath(SNPFREE, ROOT),
        "output_sha256": sha256_file(SNPFREE),
        "columns": info["columns"],
        "records": info["records"],
        "note": "phase0/phase1 (and any phase_prob*) removed; seven columns remain",
    }
    os.makedirs(os.path.join(ROOT, "inputs"), exist_ok=True)
    with open(os.path.join(ROOT, "inputs", "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("prepare: %d records -> %s" % (info["records"], man["output"]))
    log("prepare: sha256 %s" % man["output_sha256"])
    print(json.dumps(man, indent=2))


# --------------------------------------------------------------------------
# Stage 1 machinery
# --------------------------------------------------------------------------
FINITE_FLOOR = 1e-3


class Candidate:
    def __init__(self, cid, family, tracks, assign=None, structs=None, tb=None,
                 rule=None, note="", rec=None):
        self.id = cid
        self.family = family        # "blind" | "null" | "oracle"
        self.tracks = tracks        # list of track names (1 or 2)
        self.assign = assign        # label array over `rec`, or None
        self.structs = structs
        self.tb = tb                # per-bin flip mask, or None
        self.rule = rule
        self.note = note
        self.rec = rec              # (p1, p2, truth_label) this split is defined on


def fit_candidate(cid, chrom, p1, p2, assign, workdir, coordsdir, seed=1):
    """One FDG fit. Returns the parsed structures."""
    pp = os.path.join(workdir, cid + ".pairs.gz")
    o3 = os.path.join(coordsdir, cid + ".3dg")
    fdg.write_pairs(pp, chrom, p1, p2, assign)
    fdg.run_fdg(pp, o3, seed=seed)
    return fdg.read_3dg(o3), o3


def dist_pair(structs, chrom, b1, b2, two_track):
    if two_track:
        return (fdg.pair_distances(structs, safe_name(chrom, 0),
                                   OFF + b1 * BIN, OFF + b2 * BIN),
                fdg.pair_distances(structs, safe_name(chrom, 1),
                                   OFF + b1 * BIN, OFF + b2 * BIN))
    return (fdg.pair_distances(structs, chrom, OFF + b1 * BIN, OFF + b2 * BIN),)


def make_testset(b1, b2, C, chrom, n_bins):
    return score.TestSet(b1, b2, C, splits.frag_of(b1, n_bins), splits.frag_of(b2, n_bins))


def evaluate(cand, chrom, vs, ts, n_bins):
    """Validation-selected exponent + held-out readout, per candidate."""
    vparts = dist_pair(cand.structs, chrom, vs.b1, vs.b2, len(cand.tracks) == 2)
    okv = np.ones(vs.n, dtype=bool)
    for p in vparts:
        okv &= np.isfinite(p)
    a, _ = score.select_alpha([p[okv] for p in vparts], vs.C[okv])
    tparts = dist_pair(cand.structs, chrom, ts.b1, ts.b2, len(cand.tracks) == 2)
    pred = np.zeros(ts.n, dtype=float)
    okt = np.ones(ts.n, dtype=bool)
    for p in tparts:
        okt &= np.isfinite(p)
        pred = pred + score.predictor(p, a)
    pred[~okt] = np.nan
    cand.alpha = a
    cand.pred = pred
    cand.ok = okt
    return pred


# --------------------------------------------------------------------------
# stage1
# --------------------------------------------------------------------------
def cmd_stage1(args):
    if args.relax_rounds != 1:
        raise ValueError("stage1 supports only --relax-rounds 1; multi-round refitting is not implemented")
    t_start = time.time()
    chrom = args.chrom
    outdir = args.out
    create_fresh_output_dir(outdir)
    rng = np.random.default_rng(args.seed)

    folds.sanity_check()

    # ---- 1. SNP-free inputs, folds ------------------------------------
    p1, p2 = pairs7.load(SNPFREE, chrom=chrom)
    log("%s: %d cis contact records (SNP-free, 7 columns)" % (chrom, len(p1)))
    samebin_records = int((bin_of(p1) == bin_of(p2)).sum())
    log("%s: same-bin records %d (%.2f%%) -- excluded from all structural metrics"
        % (chrom, samebin_records, 100.0 * samebin_records / len(p1)))
    I1, I2 = bin_of(p1), bin_of(p2)
    n_bins = int(max(I1.max(), I2.max())) + 1
    nfrag = splits.n_fragments(n_bins)
    f = folds.fold_of_array(I1, I2)
    tr, va, te = folds.masks(f)
    log("%s: bins %d, fragments %d x %d bins | train/val/test records %d/%d/%d"
        % (chrom, n_bins, nfrag, FRAG_BINS, tr.sum(), va.sum(), te.sum()))

    def set_of(mask):
        b1, b2, C = score.bin_pair_counts(I1, I2, mask & (I1 != I2))
        return b1, b2, C

    vb1, vb2, vC = set_of(va)
    tb1, tb2, tC = set_of(te)
    vs = make_testset(vb1, vb2, vC, chrom, n_bins)
    ts = make_testset(tb1, tb2, tC, chrom, n_bins)
    log("%s: held-out bin pairs  val %d  test %d (same-bin excluded)"
        % (chrom, vs.n, ts.n))
    for k, m in ts.masks.items():
        log("   stratum %-10s n=%5d  contacts=%6d" % (k, int(m.sum()), int(tC[m].sum())))

    gate = EvalGate(os.path.join(outdir, "gate.json"))
    cands = []

    # ---- 2. BLIND candidates (SNP-free only) --------------------------
    log("--- blind candidates (no label access) ---")
    trp1, trp2 = p1[tr], p2[tr]
    s, o3 = fit_candidate("consensus", chrom, trp1, trp2, None,
                          os.path.join(outdir, "work"), os.path.join(outdir, "coords"))
    gate.register("blind", "consensus", o3)
    cands.append(Candidate("consensus", "blind", [chrom], None, s, rec=(trp1, trp2, None)))
    log("consensus fitted  (%s)" % os.path.basename(o3))

    rand_assign = splits.random_split(int(tr.sum()), args.seed)
    s, o3 = fit_candidate("random", chrom, trp1, trp2, rand_assign,
                          os.path.join(outdir, "work"), os.path.join(outdir, "coords"))
    gate.register("blind", "random", o3)
    cands.append(Candidate("random", "blind", [safe_name(chrom, 0), safe_name(chrom, 1)],
                           rand_assign, s, rec=(trp1, trp2, None)))
    log("random split fitted, %d/%d contacts to copy a" % (int(rand_assign.sum()), len(rand_assign)))

    # ---- 3. arm the gate; only now read labels ------------------------
    gate.arm(ref3dg.STAGE)
    from pr import labels as lab
    lp1, lp2, lb = lab.load_phase_labels(gate, chrom)
    assert len(lp1) == len(p1) and np.array_equal(lp1, p1) and np.array_equal(lp2, p2), \
        "label records do not line up with the SNP-free records"
    log("labels read under gate: %s" % lab.label_counts(lb))
    # Reporting rule 2: an oracle is built only from FULLY LABELLED contacts.
    # Every label-based candidate therefore shares one record set (the phased
    # train records) and differs only in its labels. The blind controls keep the
    # larger all-record train set, which makes them stronger controls.
    ph_tr = tr & (lb >= 0)
    n_ph_tr = int(ph_tr.sum())
    log("phased train records %d of %d train records (blind family uses all %d)"
        % (n_ph_tr, int(tr.sum()), int(tr.sum())))
    o_p1, o_p2, o_lb = p1[ph_tr], p2[ph_tr], lb[ph_tr]
    o_b1, o_b2 = I1[ph_tr], I2[ph_tr]        # bins, not fragment indices
    if args.ph_subsample and args.ph_subsample < n_ph_tr:
        # Depth control: hold everything fixed except the number of fully phased
        # training records, so the chrX result can be tested against chr1 at the
        # same depth instead of against a different chromosome.
        sub = np.random.default_rng(args.seed + 77).choice(n_ph_tr, args.ph_subsample,
                                                           replace=False)
        o_p1, o_p2, o_lb = o_p1[sub], o_p2[sub], o_lb[sub]
        o_b1, o_b2 = o_b1[sub], o_b2[sub]
        n_ph_tr = int(args.ph_subsample)
        log("DEPTH CONTROL: phased train records subsampled to %d (chrX has 5209)" % n_ph_tr)

    # ---- 4. oracle instruments ---------------------------------------
    log("--- oracle instruments (labels; refit on the phased train fold only) ---")

    # The null: several iid 50/50 splits of the SAME phased record set. All of
    # them are label-free and structurally equivalent, so the spread of rho
    # among them is exactly the "repartitioning wobble" a swap effect must beat.
    for k in range(args.n_null):
        cid = "random_phased" if k == 0 else "random_phased_%d" % k
        rp_assign = splits.random_split(n_ph_tr, args.seed + 5 + k)
        s, o3 = fit_candidate(cid, chrom, o_p1, o_p2, rp_assign,
                              os.path.join(outdir, "work"), os.path.join(outdir, "coords"))
        gate.register("eval", cid, o3)
        c = Candidate(cid, "null", [safe_name(chrom, 0), safe_name(chrom, 1)],
                      rp_assign, s, tb=None, rule=None,
                      note="null control: random labels on the oracle record set",
                      rec=(o_p1, o_p2, o_lb))
        c.flipped_mb = 0
        cands.append(c)
    null_ids = [c.id for c in cands if c.family == "null"]
    log("null family: %d independent random 50/50 splits of the same %d records: %s"
        % (len(null_ids), n_ph_tr, ", ".join(null_ids)))

    patterns = [("oracle", "relabel"), ("gauge_all", "relabel")]
    patterns += [("micro%d" % k, "relabel") for k in args.micro]
    patterns += [("wall%d" % m, "relabel") for m in range(1, 6)]
    patterns += [("island%d" % m, "relabel") for m in range(1, 4)]
    patterns += [("alt", "relabel")]
    patterns += [("wall%d" % m, "rewire") for m in range(1, 6)]
    patterns += [("alt", "rewire")]

    def add_pattern(cid, tb, rule, tag=""):
        assign = splits.apply_flip(o_lb, o_b1, o_b2, tb, rule=rule)
        st, o3 = fit_candidate(cid, chrom, o_p1, o_p2, assign,
                               os.path.join(outdir, "work"), os.path.join(outdir, "coords"))
        gate.register("eval", cid, o3)
        cc = Candidate(cid, "oracle", [safe_name(chrom, 0), safe_name(chrom, 1)],
                       assign, st, tb=tb, rule=rule, rec=(o_p1, o_p2, o_lb), note=tag)
        cc.flipped_mb = splits.flipped_mb(tb)
        cc.n_walls = splits.n_walls(tb)
        cands.append(cc)
        log("%-12s flipped %3d Mb | %d wall(s) | copy-a gets %d/%d train contacts%s"
            % (cid, cc.flipped_mb, cc.n_walls, int(assign.sum()), len(assign), tag))
        return cc

    seen = set()
    for kind, rule in patterns:
        cid = kind if rule == "relabel" else "%s_%s" % (kind, rule)
        if cid in seen:
            continue
        seen.add(cid)
        tb = splits.bin_sign(kind, n_bins)
        add_pattern(cid, tb, rule)
        # The gauge complement (every remaining bin's label flipped) is the same
        # hypothesis with copyA/copyB exchanged. Under `rewire` it is literally
        # the same assignment, so only `relabel` needs the extra fit.
        if rule == "relabel" and 0 < tb.sum() < n_bins:
            add_pattern(cid + "_gc", ~tb, rule, tag="   [gauge complement]")

    # Independent-oracle null: the FDG output is bit-identical across `-s` seeds
    # (verified), so the only available source of fit-to-fit variation is the
    # data. Two disjoint random halves of the phased train records give the
    # fit-noise floor against which every swap difference must be read.
    t0 = splits.bin_sign("oracle", n_bins)
    a0 = splits.apply_flip(o_lb, o_b1, o_b2, t0, rule="relabel")
    order = rng.permutation(n_ph_tr)
    halves = {"oracle_half1": order[:n_ph_tr // 2], "oracle_half2": order[n_ph_tr // 2:]}
    for cid, sub_idx in halves.items():
        s, o3 = fit_candidate(cid, chrom, o_p1[sub_idx], o_p2[sub_idx], a0[sub_idx],
                              os.path.join(outdir, "work"), os.path.join(outdir, "coords"))
        gate.register("eval", cid, o3)
        c = Candidate(cid, "oracle", [safe_name(chrom, 0), safe_name(chrom, 1)],
                      a0[sub_idx], s, tb=t0, rule="relabel",
                      note="independent oracle fit on %d of %d phased train records"
                           % (len(sub_idx), n_ph_tr),
                      rec=(o_p1[sub_idx], o_p2[sub_idx], o_lb[sub_idx]))
        c.flipped_mb = 0
        cands.append(c)

    # ---- 5. score -----------------------------------------------------
    # NaN policy (declared): a candidate's FDG output has no bead for a bin that
    # carries no contact in that candidate's own fitting record set, so the
    # distance is genuinely undefined there. Each comparison therefore uses the
    # MAXIMAL COMMON set of the two candidates involved, and every row reports
    # its own n. There is no global intersection of all candidates: that would
    # let one auxiliary diagnostic fit silently shrink every other number.
    log("--- scoring: alpha on validation, rho on test ---")
    for c in cands:
        evaluate(c, chrom, vs, ts, n_bins)

    global TS, STRATUM_IDX
    TS = ts
    # STRATUM_IDX holds INDEX ARRAYS, not boolean masks. Never combine one with a
    # boolean mask using `&`: numpy casts the bool to 0/1 and `arange(n) & mask`
    # silently becomes `arange(n) & 1`, i.e. parity of the row number. That is the
    # same defect class as the retired `(i+j) % 2` fold hash. Use `common(*cands)`
    # to build boolean masks instead.
    STRATUM_IDX = {m: np.where(mm)[0] for m, mm in ts.masks.items()}
    log("scoring universe: %d test bin pairs (same-bin excluded); "
        "each row reports the finite subset it actually used" % ts.n)

    # mechanistic strata for each pattern
    mechan = {}
    for c in cands:
        if c.tb is None:
            mechan[c.id] = (ts.masks["all"], ts.masks["all"])
            continue
        a, b = c.tb[ts.b1], c.tb[ts.b2]
        mechan[c.id] = (a == b, a != b)

    ref_c = [c for c in cands if c.id == "oracle"][0]
    ref_pred = ref_c.pred
    ok_ref = ref_c.ok

    def row_masks(c):
        """{stratum: index array finite for this candidate}."""
        out = {}
        for m, idx in STRATUM_IDX.items():
            out[m] = idx[c.ok[idx]]
        una, aff = mechan[c.id]
        for nm, mm in (("unaffected", una), ("affected", aff)):
            idx = np.where(mm)[0]
            out[nm] = idx[c.ok[idx]]
        return out

    table = {}
    for c in cands:
        rm = row_masks(c)
        row = {"family": c.family, "alpha": c.alpha,
               "flipped_mb": getattr(c, "flipped_mb", None),
               "n_walls": getattr(c, "n_walls", None), "rule": c.rule,
               "n_finite": int(c.ok.sum()), "n_nan": int((~c.ok).sum())}
        row["strata"] = {m: {"n": int(len(idx)),
                             "rho": score.spearman(c.pred[idx], ts.C[idx])
                             if len(idx) >= 20 else float("nan")}
                         for m, idx in rm.items()}
        row["rho_pooled"] = row["strata"]["all"]["rho"]
        table[c.id] = row
        log("%-12s alpha %.2f | pooled rho %.4f (n=%d, nan=%d) | intra %.4f xshort %.4f "
            "xlong %.4f | unaffected %.4f affected %.4f"
            % (c.id, c.alpha, row["rho_pooled"], row["n_finite"], row["n_nan"],
               row["strata"]["intra_frag"]["rho"], row["strata"]["xshort"]["rho"],
               row["strata"]["xlong"]["rho"], row["strata"]["unaffected"]["rho"],
               row["strata"]["affected"]["rho"]))

    def common(*cs):
        """BOOLEAN mask: test-fold pairs finite for every one of these candidates."""
        m = np.ones(ts.n, dtype=bool)
        for c in cs:
            assert c.ok.dtype == bool, "candidate masks must be boolean"
            m &= c.ok
        return m

    # ---- 6. paired bootstrap against the oracle -----------------------
    log("--- paired bootstrap vs oracle (%d resamples) ---" % args.nboot)
    boot = {}
    for c in cands:
        if c.id == "oracle":
            boot[c.id] = {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": int(common(ref_c, c).sum()),
                          "excludes_zero": False, "d": np.zeros(0)}
            continue
        m = common(ref_c, c)
        idx = np.where(m)[0]
        lo, hi, d = score.paired_bootstrap(ts.C[idx], ref_pred[idx], c.pred[idx],
                                           n_boot=args.nboot, seed=args.seed)
        # delta and its CI must live on the SAME universe, or the point estimate
        # and the interval answer different questions.
        boot[c.id] = {"delta": (score.spearman(ref_pred[idx], ts.C[idx])
                                - score.spearman(c.pred[idx], ts.C[idx])),
                      "delta_own_sets": (table["oracle"]["rho_pooled"]
                                         - table[c.id]["rho_pooled"]),
                      "lo": lo, "hi": hi, "n": int(len(idx)),
                      "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                      "d": d}
        log("  oracle - %-16s = %+.4f (own sets %+.4f) [%+.4f, %+.4f] n=%d  %s"
            % (c.id, boot[c.id]["delta"], boot[c.id]["delta_own_sets"], lo, hi, len(idx),
               "DISTINGUISHABLE" if boot[c.id]["excludes_zero"] else "indistinguishable"))
    for k in null_ids + ["oracle_half1", "oracle_half2"]:
        v = boot[k]
        tag = "null" if k in null_ids else "half-data oracle"
        log("  [%s] oracle - %-16s = %+.4f [%+.4f, %+.4f] n=%d"
            % (tag, k, v["delta"], v["lo"], v["hi"], v["n"]))

    null_rhos = np.array([table[k]["rho_pooled"] for k in null_ids])
    null_spread = {"ids": null_ids, "rhos": null_rhos.tolist(),
                   "sd": float(null_rhos.std(ddof=1)) if len(null_rhos) > 1 else float("nan"),
                   "range": float(null_rhos.max() - null_rhos.min()) if len(null_rhos) > 1
                   else float("nan")}
    log("  null spread over %d random splits: sd %.4f  range %.4f"
        % (len(null_ids), null_spread["sd"], null_spread["range"]))

    # ---- 6b. common-alpha readout (removes the exponent as an explanation) ---
    a_common = table["oracle"]["alpha"]
    common_alpha = {}
    for c in cands:
        parts = dist_pair(c.structs, chrom, ts.b1, ts.b2, len(c.tracks) == 2)
        p = np.zeros(ts.n)
        ok = np.ones(ts.n, dtype=bool)
        for q in parts:
            ok &= np.isfinite(q)
            p = p + score.predictor(q, a_common)
        p[~ok] = np.nan
        c.ok &= ok
        idx = np.where(common(ref_c, c))[0]
        if len(idx) < 20:
            continue
        rho = score.spearman(p[idx], ts.C[idx])
        lo, hi, _ = score.paired_bootstrap(ts.C[idx], ref_pred[idx], p[idx],
                                           n_boot=args.nboot, seed=args.seed)
        common_alpha[c.id] = {"rho": rho, "delta": score.spearman(ref_pred[idx], ts.C[idx]) - rho,
                              "lo": lo, "hi": hi, "n": int(len(idx)),
                              "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
    log("--- common-alpha readout at a = %.2f (the oracle's validation choice) ---" % a_common)
    for cid in ["consensus", "random"] + null_ids + [
            "gauge_all", "micro5", "micro10", "wall1", "wall2", "wall3", "wall4", "wall5",
            "island1", "island2", "island3", "alt",
            "wall1_rewire", "wall5_rewire", "alt_rewire"]:
        if cid not in common_alpha:
            continue
        v = common_alpha[cid]
        log("  %-15s rho %.4f  delta %+.4f [%+.4f, %+.4f] n=%d %s"
            % (cid, v["rho"], v["delta"], v["lo"], v["hi"], v["n"],
               "DISTINGUISHABLE" if v["excludes_zero"] else "indistinguishable"))

    # ---- 6c. gauge-invariant readout ----------------------------------
    gauge = {}
    by_id = {c.id: c for c in cands}
    ref_gc = by_id.get("gauge_all")
    ref_gi = (table["oracle"]["rho_pooled"] + table["gauge_all"]["rho_pooled"]) / 2.0
    gauge_asym = abs(table["oracle"]["rho_pooled"] - table["gauge_all"]["rho_pooled"])
    log("--- gauge-invariant readout ---")
    log("  engine gauge asymmetry |rho(oracle) - rho(gauge_all)| = %.4f ; oracle rho_avg = %.4f"
        % (gauge_asym, ref_gi))
    gauge["_reference"] = {"rho_oracle_gauge_avg": ref_gi, "gauge_asymmetry": gauge_asym,
                           "rho_oracle": table["oracle"]["rho_pooled"],
                           "rho_gauge_all": table["gauge_all"]["rho_pooled"]}
    pat_ids = [k for k in by_id
               if k.startswith(("micro", "wall", "island", "alt")) and not k.endswith("_gc")]
    for cid in pat_ids:
        c = by_id[cid]
        gc = by_id.get(cid + "_gc")
        if gc is not None:
            r_b = (table[cid]["rho_pooled"] + table[gc.id]["rho_pooled"]) / 2.0
            idx = np.where(common(ref_c, ref_gc, c, gc))[0]
            lo, hi, _ = score.paired_bootstrap_gaugeavg(
                ts.C[idx], ref_pred[idx], ref_gc.pred[idx], c.pred[idx], gc.pred[idx],
                n_boot=args.nboot, seed=args.seed)
            asym = abs(table[cid]["rho_pooled"] - table[gc.id]["rho_pooled"])
        else:
            r_b = table[cid]["rho_pooled"]
            idx = np.where(common(ref_c, ref_gc, c))[0]
            lo, hi, _ = score.paired_bootstrap_gaugeavg(
                ts.C[idx], ref_pred[idx], ref_gc.pred[idx], c.pred[idx], c.pred[idx],
                n_boot=args.nboot, seed=args.seed)
            asym = 0.0
        gauge[cid] = {"rho_gauge_avg": r_b, "gauge_asymmetry": asym, "n": int(len(idx)),
                      "delta": ref_gi - r_b, "lo": lo, "hi": hi,
                      "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
        log("  %-14s rho_avg %.4f | own gauge asym %.4f | delta %+.4f [%+.4f, %+.4f] n=%d %s"
            % (cid, r_b, asym, ref_gi - r_b, lo, hi, len(idx),
               "DISTINGUISHABLE" if gauge[cid]["excludes_zero"] else "indistinguishable"))

    # ---- 6d. masked comparison: rho(oracle) vs rho(candidate) on the pairs the
    #          candidate itself damaged, with a matched null on the same pairs. --
    masked = {}
    log("--- restricted to each candidate's own AFFECTED bin pairs ---")
    h1, h2 = by_id["oracle_half1"], by_id["oracle_half2"]
    for c in cands:
        una, aff = mechan[c.id]
        if c.tb is None or c.id == "oracle":
            continue
        idx = np.where(aff & common(ref_c, c))[0]
        if len(idx) < 30:
            continue
        idn = np.where(aff & common(ref_c, c, h1, h2))[0]
        r_alt = score.spearman(c.pred[idx], ts.C[idx])
        r_ref = score.spearman(ref_pred[idx], ts.C[idx])
        lo, hi, _ = score.paired_bootstrap(ts.C[idx], ref_pred[idx], c.pred[idx],
                                           n_boot=args.nboot, seed=args.seed)
        nlo, nhi = (float("nan"), float("nan"))
        nd = float("nan")
        if len(idn) >= 30:
            nlo, nhi, _ = score.paired_bootstrap(ts.C[idn], h1.pred[idn], h2.pred[idn],
                                                 n_boot=args.nboot, seed=args.seed)
            nd = float(score.spearman(h1.pred[idn], ts.C[idn])
                       - score.spearman(h2.pred[idn], ts.C[idn]))
        masked[c.id] = {"n": int(len(idx)), "n_common_with_null": int(len(idn)),
                        "frac_of_test": float(len(idx) / ts.n),
                        "rho_oracle": r_ref, "rho_cand": r_alt,
                        "delta": r_ref - r_alt, "delta_common": r_ref - r_alt,
                        "lo": lo, "hi": hi,
                        "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                        "null_delta_half_oracles": nd, "null_lo": nlo, "null_hi": nhi}
        log("  %-15s n=%4d (%.0f%%) | oracle %.4f cand %.4f | delta %+.4f [%+.4f, %+.4f] %s"
            " | null(half-oracle) %+.4f (n=%d)"
            % (c.id, len(idx), 100 * len(idx) / ts.n, r_ref, r_alt,
               r_ref - r_alt, lo, hi,
               "DISTINGUISHABLE" if masked[c.id]["excludes_zero"] else "indistinguishable",
               nd, len(idn)))

    # ---- 7. boundary relaxation, equal budget for every candidate -----
    relax = {}
    if args.relax:
        log("--- boundary relaxation: %d round(s), +/-%d bins around each 20 Mb grid line ---"
            % (args.relax_rounds, args.relax_band))
        band_bins = np.zeros(n_bins, dtype=bool)
        grid = [FRAG_BINS * k for k in range(1, nfrag)]
        for g in grid:
            band_bins[max(0, g - args.relax_band):min(n_bins, g + args.relax_band + 1)] = True
        for c in cands:
            if len(c.tracks) != 2:
                relax[c.id] = {"skipped": "single-structure control"}
                continue
            rp1, rp2, rtruth = c.rec
            inband = band_bins[bin_of(rp1)] | band_bins[bin_of(rp2)]
            r = relax_candidate(c, chrom, inband, outdir, args.relax_rounds, vs, ts, gate)
            relax[c.id] = r
            log("  %-12s relaxed rho %.4f (was %.4f) | E-step agreement %.4f | changed %5d | band %d rec"
                % (c.id, r["rho_pooled"], table[c.id]["rho_pooled"],
                   r["estep_agreement"], r["changed"], r["n_in_band"]))

    # ---- 8. reference (evaluator side, after hashing) -----------------
    log("--- evaluator-side reference readout ---")
    refc = ref3dg.load_reference(gate)
    acc = ref3dg.reference_accuracy(refc, chrom, I1, I2, lb)
    log("reference structures classify %s cis contacts: acc %.4f (orientation %d), "
        "n %d, same-bin excluded %d, tie rate %.4f"
        % (chrom, acc["acc"], acc["orientation"], acc["n"],
           acc["n_samebin_excluded"], acc["tie_rate"]))

    # structure-vs-reference rank correlation, best swap chosen by geometry
    refcorr = {}
    for c in cands:
        row = {}
        for ti, trk in enumerate(c.tracks):
            d_ours = fdg.pair_distances(c.structs, trk, OFF + ts.b1 * BIN, OFF + ts.b2 * BIN)
            for suffix in ("mat", "pat"):
                d_ref = ref3dg.distance(refc, "%s(%s)" % (chrom, suffix), ts.b1, ts.b2)
                ok = np.isfinite(d_ours) & np.isfinite(d_ref)
                row["%s_vs_%s" % (trk, suffix)] = {
                    "rho": score.spearman(d_ours[ok], d_ref[ok]), "n": int(ok.sum())}
        refcorr[c.id] = row

    # ---- 9. splice instrument (structure level, no relabelling) --------
    splice_res = {}
    if args.splice:
        log("--- structure-level splice (chimera, no contact relabelling) ---")
        orc = [c for c in cands if c.id == "oracle"][0]
        ka, kb = safe_name(chrom, 0), safe_name(chrom, 1)
        for m in ["micro%d" % k for k in args.micro] + ["wall%d" % m for m in range(1, 6)]:
            tb = splits.bin_sign(m, n_bins)
            blk = [OFF + b * BIN for b in range(n_bins) if tb[b]]
            A2, rms_a = splice.splice(orc.structs, ka, kb, blk)
            B2, rms_b = splice.splice(orc.structs, kb, ka, blk)
            if A2 is None:
                continue
            st = {ka: A2, kb: B2}
            pred = None
            ok = np.ones(ts.n, dtype=bool)
            for trk in (ka, kb):
                d = fdg.pair_distances(st, trk, OFF + ts.b1 * BIN, OFF + ts.b2 * BIN)
                ok &= np.isfinite(d)
                pred = score.predictor(d, orc.alpha) if pred is None else pred + score.predictor(d, orc.alpha)
            pred[~ok] = np.nan
            idx = np.where(common(orc) & ok)[0]
            rho = score.spearman(pred[idx], ts.C[idx])
            lo, hi, _ = score.paired_bootstrap(ts.C[idx], orc.pred[idx], pred[idx],
                                               n_boot=args.nboot, seed=args.seed)
            splice_res["splice_%s" % m] = {
                "rho_pooled": rho,
                "delta_vs_oracle": score.spearman(orc.pred[idx], ts.C[idx]) - rho,
                "lo": lo, "hi": hi, "n": int(len(idx)),
                "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                "flipped_mb": int(tb.sum()), "procrustes_rmsd": rms_a}
            log("  splice_%-8s (%3d Mb flipped) rho %.4f  delta %+.4f [%+.4f, %+.4f] n=%d %s"
                % (m, int(tb.sum()), rho, splice_res["splice_%s" % m]["delta_vs_oracle"], lo, hi,
                   len(idx),
                   "DISTINGUISHABLE" if splice_res["splice_%s" % m]["excludes_zero"]
                   else "indistinguishable"))

    # ---- 9b. selection test: does the label-free criterion rank the
    #          chromosome-wide consistent split first? ----------------------
    selection = {}
    log("--- selection test ---")
    families = {
        "relabel (gauge-invariant)": [("oracle", ref_gi)] + [
            (cid, gauge[cid]["rho_gauge_avg"]) for cid in gauge if cid != "_reference"],
        "rewire": [("oracle", ref_gi)] + [(c.id, table[c.id]["rho_pooled"])
                                          for c in cands if (c.rule == "rewire")],
        "splice (label-free chimera)": [("oracle", table["oracle"]["rho_pooled"])] + [
            (k, v["rho_pooled"]) for k, v in splice_res.items()],
    }
    for fam, rows in families.items():
        if len(rows) < 2:
            continue
        best = max(rows, key=lambda r: r[1])
        selection[fam] = {"best": best[0], "rho": best[1],
                          "oracle_wins": best[0] == "oracle",
                          "ranking": sorted(rows, key=lambda r: -r[1])}
        log("  %-28s argmax = %-14s (rho %.4f) | oracle first: %s"
            % (fam, best[0], best[1], "YES" if best[0] == "oracle" else "NO"))

    # ---- 9c. pre-registered decision rule -----------------------------
    null_tol = max(null_spread["range"], gauge_asym)
    micro_ids = ["micro%d" % k for k in args.micro]
    scale_rows = []
    for cid in micro_ids + ["wall%d" % m for m in range(1, 6)] + ["alt"]:
        if cid in gauge:
            scale_rows.append((by_id[cid].flipped_mb, cid, gauge[cid]["delta"],
                               gauge[cid]["lo"], gauge[cid]["hi"], gauge[cid]["excludes_zero"]))
    scale_rows.sort()
    detectable = [r for r in scale_rows if r[5]]
    min_detect = min((r[0] for r in detectable), default=None)
    decision = {
        "null_tolerance": null_tol,
        "null_spread_range": null_spread["range"],
        "gauge_asymmetry": gauge_asym,
        "min_detectable_flipped_mb": min_detect,
        "n_scales_tested": len(scale_rows),
        "half_data_oracle_delta_max": float(max(
            abs(boot[k]["delta"]) for k in ("oracle_half1", "oracle_half2"))),
        "note": ("null_tolerance is max(null spread over independent random splits, engine "
                 "gauge asymmetry). The half-data oracle refit is a different candidate, "
                 "not a null; its effect size is reported separately as a caveat."),
        "scales": [{"flipped_mb": r[0], "id": r[1], "delta": r[2], "lo": r[3], "hi": r[4],
                    "distinguishable": r[5]} for r in scale_rows],
    }
    log("--- pre-registered decision rule ---")
    log("  null tolerance = max(null spread %.4f, gauge asymmetry %.4f) = %.4f"
        % (null_spread["range"], gauge_asym, null_tol))
    for r in scale_rows:
        log("  %5d Mb  %-8s delta %+.4f [%+.4f, %+.4f]  %s"
            % (r[0], r[1], r[2], r[3], r[4],
               "detected" if r[5] else "NOT detected"))
    log("  smallest flipped region the pooled gauge-invariant score detects: %s Mb"
        % decision["min_detectable_flipped_mb"])

    # ---- 10. write results --------------------------------------------
    res = {
        "chrom": chrom, "bin_bp": BIN, "bin_offset": OFF, "frag_bins": FRAG_BINS,
        "n_bins": n_bins, "n_fragments": nfrag,
        "n_records": int(len(p1)), "samebin_records": samebin_records,
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "n_ph_train": int(n_ph_tr),
        "pairs_val": int(vs.n), "pairs_test": int(ts.n),
        "nan_policy": ("per-comparison maximal common finite subset; every row reports "
                       "its own n; no global intersection across candidates"),
        "finite_counts": {c.id: int(c.ok.sum()) for c in cands},
        "stratum_sizes": {m: int(len(i)) for m, i in STRATUM_IDX.items()},
        "label_counts": lab.label_counts(lb),
        "reference_accuracy": acc,
        "table": table, "bootstrap": {k: {kk: vv for kk, vv in v.items() if kk != "d"}
                                      for k, v in boot.items()},
        "bootstrap_draws": {k: v["d"].tolist() for k, v in boot.items()},
        "null_spread": null_spread, "common_alpha": common_alpha,
        "masked_compare": masked, "gauge_invariant": gauge,
        "relaxation": relax, "splice": splice_res, "reference_corr": refcorr,
        "selection": selection, "decision": decision,
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "runtime_sec": time.time() - t_start,
    }
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(json_ready(res), fh, indent=2, allow_nan=False)

    if args.plots:
        make_plots(res, outdir, cands, ts, chrom)
    with open(os.path.join(outdir, "logs", "run.log"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump({k: v for k, v in vars(args).items() if k != "func"}, fh, indent=2)
    log("wrote %s/results.json (%.1f s)" % (outdir, res["runtime_sec"]))
    return res


def make_plots(res, outdir, cands, ts, chrom):
    """Three-inch panels, 300 dpi: dose response, strata, masked effect, maps."""
    from pr import figs
    from pr.paths import pos_of
    by_id = {c.id: c for c in cands}
    plots = os.path.join(outdir, "plots")
    os.makedirs(plots, exist_ok=True)
    g = res["gauge_invariant"]
    sp = res["splice"]
    series = {}
    for key, col in (("relabel", "C0"), ("rewire", "C1"), ("splice", "C2")):
        xs, ds, los, his = [], [], [], []
        for k, v in g.items():
            if k == "_reference":
                continue
            if key == "relabel" and (k.endswith("_gc") or k.endswith("_rewire")):
                continue
            if key == "rewire" and not k.endswith("_rewire"):
                continue
            if key == "splice":
                continue
            c = by_id[k]
            xs.append(c.flipped_mb); ds.append(v["delta"])
            los.append(v["lo"]); his.append(v["hi"])
        if key == "splice":
            for k, v in sp.items():
                xs.append(v["flipped_mb"]); ds.append(v["delta_vs_oracle"])
                los.append(v["lo"]); his.append(v["hi"])
        series[key] = (np.array(xs), np.array(ds), np.array(los), np.array(his), col)
    figs.dose_response(os.path.join(plots, "fig1_dose_response.png"), series,
                       null_band=res["null_spread"]["range"],
                       gauge=res["decision"]["gauge_asymmetry"])

    ids = ["oracle", "gauge_all"] + ["wall%d" % m for m in range(1, 6)] + ["alt"]
    ids = [i for i in ids if i in res["table"]]
    strata = {i: res["table"][i]["strata"] for i in ids}
    figs.strata_bars(os.path.join(plots, "fig2_strata.png"), ids, strata,
                     ["intra_frag", "xshort", "xlong"],
                     title="%s: held-out rho by contact class" % chrom)

    mids = [i for i in ids[2:] if i in res["masked_compare"]]
    figs.masked_effect(os.path.join(plots, "fig3_masked.png"), mids,
                       [res["masked_compare"][i]["delta"] for i in mids],
                       [res["masked_compare"][i]["lo"] for i in mids],
                       [res["masked_compare"][i]["hi"] for i in mids],
                       [res["masked_compare"][i]["null_delta_half_oracles"] for i in mids])

    from pr.paths import safe_name
    o = by_id["oracle"]
    w = by_id.get("wall2")
    bins = list(range(0, res["n_bins"], 4))
    panel = []
    Mo = figs.dist_matrix(o.structs, safe_name(chrom, 0), bins, pos_of)
    vmax = np.nanpercentile(Mo, 95)
    panel.append(("oracle copyA", Mo, 0, vmax))
    if w is not None:
        Mw = figs.dist_matrix(w.structs, safe_name(chrom, 0), bins, pos_of)
        panel.append(("wall2 copyA (mis-wired)", Mw, 0, vmax))
        panel.append(("difference", Mw - Mo, -vmax / 2, vmax / 2))
    figs.distance_maps(os.path.join(plots, "fig4_maps.png"), panel,
                       title="%s copy A distance maps (4 Mb bins)" % chrom)
    log("plots written to %s" % plots)


def relax_candidate(c, chrom, inband, outdir, rounds, vs, ts, gate=None):
    """Run the supported one-round boundary E-step and FDG refit."""
    if rounds != 1:
        raise ValueError("relax_candidate supports exactly one round")
    trp1, trp2, ltr = c.rec
    assign = c.assign.copy()
    structs = c.structs
    changed = 0
    agree = float("nan")
    for r in range(rounds):
        d0 = fdg.pair_distances(structs, safe_name(chrom, 0),
                                trp1, trp2)
        d1 = fdg.pair_distances(structs, safe_name(chrom, 1),
                                trp1, trp2)
        ok = np.isfinite(d0) & np.isfinite(d1)
        sel = inband & ok
        new = (d1[sel] < d0[sel]).astype(assign.dtype)
        changed = int((new != assign[sel]).sum())
        agree = (float((new == ltr[sel]).mean())
                 if (ltr is not None and sel.sum()) else float("nan"))
        assign = assign.copy()
        assign[sel] = new
    pp = os.path.join(outdir, "work", c.id + "_relax.pairs.gz")
    o3 = os.path.join(outdir, "coords", c.id + "_relax.3dg")
    fdg.write_pairs(pp, chrom, trp1, trp2, assign)
    fdg.run_fdg(pp, o3, seed=1)
    if gate is not None:
        gate.register("eval", c.id + "_relax", o3)
    st = fdg.read_3dg(o3)
    sub = Candidate(c.id + "_relax", c.family, c.tracks, assign, st, c.tb, c.rule,
                    rec=c.rec)
    sub.pred = None
    evaluate(sub, chrom, vs, ts, None)
    okr = np.where(np.isfinite(sub.pred))[0]
    return {"rho_pooled": score.spearman(sub.pred[okr], ts.C[okr]),
            "estep_agreement": agree, "changed": changed,
            "n_in_band": int(inband.sum()), "rounds": rounds,
            "structure": o3}




# --------------------------------------------------------------------------
# split --genome  (Stage S0)
# --------------------------------------------------------------------------
def cmd_split(args):
    if args.bin_size != "1m":
        raise ValueError("split final evaluation supports only --bin-size 1m; use gwfdg directly for warm starts")
    t0 = time.time()
    outdir = args.out
    create_fresh_output_dir(outdir)
    workdir = os.path.join(outdir, "work")
    coordsdir = os.path.join(outdir, "coords")
    lengths = genome.chrom_lengths(SNPFREE)
    contacts = genome.load_all(SNPFREE)
    header_names = [name for name, _length in lengths]
    if contacts["names"] != header_names:
        raise RuntimeError("SNP-free chromosome headers do not match loaded contact names")
    assignment_seed = args.assignment_seed
    native_seed = args.native_seed
    n_cis = int(contacts["cis"].sum())
    log("genome-wide: %d contacts (%d cis, %d inter) over %d chromosomes"
        % (len(contacts["ci"]), n_cis, len(contacts["ci"]) - n_cis, len(lengths)))
    nbins = genome.n_bins_per_chrom(lengths)
    log("bins per copy: %d (%d chromosomes, 1 Mb)" % (sum(nbins), len(nbins)))
    log("seed provenance: assignment_seed=%d; native_seed=%d (initialization only, not a data replicate)"
        % (assignment_seed, native_seed))

    gate = EvalGate(os.path.join(outdir, "gate.json"))
    cands = {}

    # ---- blind candidates first; only then may labels be read -----------
    log("--- S0 blind candidates (no label access) ---")
    for name, single in (("consensus", True), ("random", False)):
        if name == "random":
            k1, k2 = genome.random_assignment(contacts, seed=assignment_seed)
        else:
            k1 = k2 = None
        st, o3, dt, n = s0.fit(name, contacts, lengths, k1, k2, workdir, coordsdir,
                               n_iter=args.n_iter, bin_size=args.bin_size,
                               single=single, seed=native_seed)
        gate.register("blind", name, o3)
        cands[name] = {"structs": st, "path": o3, "sec": dt, "records": n, "family": "blind"}
        log("%-10s fitted in %6.1f s from %d records (%d tracks)"
            % (name, dt, n, 20 if single else 40))

    gate.arm(refeval.STAGE)
    a1, a2 = refeval.load_labels_two(gate, contacts)
    lab = refeval.single_label(a1, a2)          # R1 only: ends must agree
    keep = s0.oracle_keep(a1, a2)
    k1, k2 = s0.oracle_copies(a1, a2)
    log("--- S0 oracle (labels; records with BOTH ends labelled) ---")
    log("oracle records: %d of %d (cis %d, inter %d) | of the inter ones, %d have ends on "
        "DIFFERENT copies"
        % (int(keep.sum()), len(keep), int((keep & contacts["cis"]).sum()),
           int((keep & ~contacts["cis"]).sum()),
           int((keep & ~contacts["cis"] & (a1 != a2)).sum())))
    st, o3, dt, n = s0.fit("oracle", contacts, lengths, k1, k2, workdir, coordsdir,
                            n_iter=args.n_iter, bin_size=args.bin_size,
                            keep=keep, seed=native_seed)
    gate.register("eval", "oracle", o3)
    cands["oracle"] = {"structs": st, "path": o3, "sec": dt, "records": n, "family": "oracle"}
    log("%-10s fitted in %6.1f s from %d records" % ("oracle", dt, n))

    # ---- evaluator side -------------------------------------------------
    ref = refeval.load_reference(gate)
    log("--- R1/R2/R3 vs the reference ---")
    res = {"n_contacts": int(len(contacts["ci"])), "n_cis": n_cis,
           "n_inter": int(len(contacts["ci"]) - n_cis),
           "bins_per_copy": int(sum(nbins)), "n_chromosomes": len(lengths),
           "n_iter": args.n_iter, "bin_size": args.bin_size,
           "per_chromosome": {}, "candidates": {}}
    for nm, c in cands.items():
        res["candidates"][nm] = {"sec": c["sec"], "records": c["records"],
                                 "tracks": 20 if nm == "consensus" else 40,
                                 "path": os.path.relpath(c["path"])}
    refeval.clear_cache()
    for ci, (cname, _L) in enumerate(lengths):
        sel = np.where(contacts["cis"] & (contacts["ci"] == ci))[0]
        b1 = genome.bin_of(contacts["p1"][sel])
        b2 = genome.bin_of(contacts["p2"][sel])
        row = {"n_cis_records": int(len(sel)),
               "n_metric_grid_records": int(((b1 >= 0) & (b2 >= 0)
                                                & (b1 < nbins[ci]) & (b2 < nbins[ci])).sum()),
               "n_out_of_grid_records": int(((b1 < 0) | (b2 < 0)
                                               | (b1 >= nbins[ci]) | (b2 >= nbins[ci])).sum())}
        r2_by_candidate = {
            nm: refeval.r2_table(c["structs"], ci, cname, ref, nbins[ci])
            for nm, c in cands.items()}
        oracle_gauge = r2_by_candidate["oracle"].get("gauge")
        for nm, c in cands.items():
            r1 = refeval.r1_accuracy(
                c["structs"], ci, cname, lab[sel], b1, b2, nbins[ci], ref=ref,
                gauge=r2_by_candidate[nm].get("gauge"),
                oracle_structs=cands["oracle"]["structs"], oracle_gauge=oracle_gauge)
            row[nm] = {
                "R1": r1,
                "R2": r2_by_candidate[nm],
                "R3": refeval.r3_fragments(c["structs"], ci, cname, ref, nbins[ci]),
            }
        row["oracle_fit_ceiling"] = {
            "accuracy": row["oracle"]["R1"].get("accuracy"),
            "reference_ceiling": row["oracle"]["R1"].get("reference_ceiling"),
            "n_denominator": row["oracle"]["R1"].get("n_denominator"),
        }
        res["per_chromosome"][cname] = row
        fmt = lambda value, digits=3: ("n/a" if value is None or not np.isfinite(value)
                                       else ("%%.%df" % digits) % value)
        log("%-6s cis_records=%6d grid=%6d out_grid=%d | R1 random %s (den=%d, ref=%s, oracle=%s) "
            "oracle %s (den=%d) | R2 random/oracle=%s/%s | R3 walls random/oracle=%s/%s"
            % (cname, row["n_cis_records"], row["n_metric_grid_records"], row["n_out_of_grid_records"],
               fmt(row["random"]["R1"].get("accuracy")),
               row["random"]["R1"].get("n_denominator", 0),
               fmt(row["random"]["R1"].get("reference_ceiling")),
               fmt(row["random"]["R1"].get("oracle_fit_ceiling")),
               fmt(row["oracle"]["R1"].get("accuracy")),
               row["oracle"]["R1"].get("n_denominator", 0),
               fmt(row["random"]["R2"].get("contrast")),
               fmt(row["oracle"]["R2"].get("contrast")),
               row["random"]["R3"].get("n_walls"), row["oracle"]["R3"].get("n_walls")))

    def summ(nm, key, sub=None):
        v = []
        for cname in res["per_chromosome"]:
            x = res["per_chromosome"][cname][nm][key]
            if isinstance(x, dict) and x.get("applicable") is False:
                continue
            if sub:
                x = x.get(sub, float("nan"))
            if isinstance(x, (int, float)) and np.isfinite(x):
                v.append(x)
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

    log("--- summary (mean over chromosomes with a usable value) ---")
    for nm in cands:
        a, na = summ(nm, "R1", "accuracy")
        cr, _ = summ(nm, "R1", "reference_ceiling")
        oc, _ = summ(nm, "R1", "oracle_fit_ceiling")
        ct, _ = summ(nm, "R2", "contrast")
        fc, nf = summ(nm, "R3", "frac_consistent")
        wl, _ = summ(nm, "R3", "n_walls")
        res["candidates"][nm].update({"R1_mean": a, "R1_reference_ceiling_mean": cr,
                                       "R1_oracle_fit_ceiling_mean": oc,
                                       "R2_contrast_mean": ct,
                                      "R3_frac_consistent_mean": fc,
                                      "R3_n_walls_mean": wl})
        log("  %-10s R1 %s (ref ceiling %s, oracle ceiling %s, chromosomes=%d) | R2 contrast %s | R3 frac %s walls %s"
            % (nm, fmt(a, 4), fmt(cr, 4), fmt(oc, 4), na, fmt(ct, 4), fmt(fc, 3), fmt(wl, 2)))

    res["runtime_sec"] = time.time() - t0
    res["config"] = {k: v for k, v in vars(args).items() if k != "func"}
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(json_ready(res), fh, indent=2, allow_nan=False)
    with open(os.path.join(outdir, "logs", "run.log"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump({k: v for k, v in vars(args).items() if k != "func"}, fh, indent=2)
    log("wrote %s/results.json (%.1f s)" % (outdir, res["runtime_sec"]))
    return res


# --------------------------------------------------------------------------
# reevaluate -- frozen existing coordinates only
# --------------------------------------------------------------------------
def cmd_reevaluate(args):
    """Re-score verified S0 coordinates without starting an FDG fit."""
    from pr import reevaluate
    outdir, _ = reevaluate.run(args)
    print("evaluation-only artifacts: %s" % outdir)


# --------------------------------------------------------------------------
# reconstruct -- formal V1 training only
# --------------------------------------------------------------------------
def cmd_reconstruct(args):
    """Run the released phase-free V1 training path in a fresh formal directory."""
    target_threads = str(args.threads)
    marker = "_P9016_RECONSTRUCT_THREADS"
    # `run.py` imports NumPy for older commands. Re-exec once so the selected
    # thread cap is present before NumPy/BLAS import in the actual training process.
    if (os.environ.get(marker) != target_threads
            or any(os.environ.get(name) != target_threads for name in
                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))):
        environment = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = target_threads
        environment[marker] = target_threads
        os.execvpe(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]], environment)
    from pr import reconstruct
    result = reconstruct.run_production(args.out, workers=args.workers, threads=args.threads)
    print("formal training artifacts: %s" % result["outdir"])
    print("label-free selected candidate: %s" % result["selected_id"])
    return result


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    prepare = sub.add_parser("prepare")
    prepare.set_defaults(func=cmd_prepare)
    s1 = sub.add_parser("stage1")
    s1.add_argument("--chrom", default="chr1")
    s1.add_argument("--out", required=True)
    s1.add_argument("--seed", type=int, default=0)
    s1.add_argument("--nboot", type=int, default=2000)
    s1.add_argument("--relax", action="store_true", default=True)
    s1.add_argument("--no-relax", dest="relax", action="store_false")
    s1.add_argument("--relax-rounds", type=int, choices=(1,), default=1,
                    help="only one E-step/refit round is implemented")
    s1.add_argument("--relax-band", type=int, default=2)
    s1.add_argument("--ph-subsample", type=int, default=None,
                    help="keep only this many phased train records (depth control)")
    s1.add_argument("--micro", type=int, nargs="*", default=[5, 10],
                    help="sub-fragment flip widths in bins (Mb), for the detection limit")
    s1.add_argument("--n-null", type=int, default=3,
                    help="independent random 50/50 splits used as the null family")
    s1.add_argument("--plots", action="store_true", default=True)
    s1.add_argument("--no-plots", dest="plots", action="store_false")
    s1.add_argument("--splice", action="store_true", default=True)
    s1.add_argument("--no-splice", dest="splice", action="store_false")
    s1.set_defaults(func=cmd_stage1)
    sp = sub.add_parser("split", help="genome-wide Stage S0 reference points; not an S2 solver")
    sp.add_argument("--out", required=True)
    sp.add_argument("--genome", action="store_true",
                    help="explicitly select the already genome-wide S0 command")
    sp.add_argument("--seed", "--assignment-seed", dest="assignment_seed", type=int, default=0,
                    help="random contact-assignment seed")
    sp.add_argument("--native-seed", type=int, default=1,
                    help="native FDG initialization seed; an optimization repeat, not a data replicate")
    sp.add_argument("--n-iter", type=int, default=1000)
    sp.add_argument("--bin-size", choices=("1m",), default="1m",
                    help="final evaluated grid; lower-level gwfdg warm starts may use 2m or 5m")
    sp.set_defaults(func=cmd_split)
    ev = sub.add_parser("reevaluate", help="evaluate hash-verified existing S0 coordinates; never refit")
    ev.add_argument("--source", required=True,
                    help="source formal run containing gate.json and consensus/random/oracle coordinates")
    ev.add_argument("--out", default=None,
                    help="fresh output directory; default is next NNN-timestamp-s0-reevaluated")
    ev.add_argument("--regression-log", default=None,
                    help="copy a completed regression-test stdout/stderr log into final logs/")
    ev.set_defaults(func=cmd_reevaluate)
    rec = sub.add_parser("reconstruct", help="formal V1 phase-free training; requires a fresh output path")
    rec.add_argument("--out", required=True,
                     help="new formal output directory; it must not already exist")
    rec.add_argument("--workers", choices=(1, 2), type=int, default=1,
                     help="independent pre-registered candidate worker processes")
    rec.add_argument("--threads", choices=(1, 2), type=int, default=1,
                     help="OPENBLAS/OpenMP thread cap per worker")
    rec.set_defaults(func=cmd_reconstruct)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
