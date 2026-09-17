"""057实验B候选、精确G增量缓存与冻结物理项。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[1]
SOURCE = ROOT / "test_res/056-20260916T152353Z-pro-review-experiments"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for p in (ROOT, SOURCE / "code", S045):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nonref_core import fixed_state_fullgrid  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402
from pr import contact_model  # noqa: E402

WEIGHTS = PenaltyWeights(count=1.0, bond=1.0, repulsion=1.0, bend=0.01, p_prior=1.0)
TYPE_PRIORITY = {"suffix": 0, "5M": 1, "10M": 2, "20M": 3}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, order="C").tobytes(order="C")).hexdigest()


def candidate_list(data) -> list[dict[str, Any]]:
    out = []
    lengths = {"5M": 5_000_000, "10M": 10_000_000, "20M": 20_000_000}
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        pos = np.asarray(data.locus_bin[slc], dtype=np.int64) * int(data.bin_size)
        chr_len = int(data.chromosome_lengths[ci])
        raw = []
        for start in range(0, chr_len, 5_000_000):
            raw.append(("suffix", start, chr_len))
            for typ, length in lengths.items():
                if start + length <= chr_len:
                    raw.append((typ, start, start + length))
        dedup = {}
        for typ, start, end in raw:
            a = int(np.searchsorted(pos, start, side="left"))
            b = int(np.searchsorted(pos, end, side="left"))
            if a >= b or (a == 0 and b == len(pos)):
                continue
            key = (a, b)
            row = {"chromosome_index": ci, "chromosome": str(name), "type": typ,
                   "start_bp": start, "end_bp": end, "bead_start": a, "bead_end": b,
                   "bead_count": b - a}
            if key not in dedup or TYPE_PRIORITY[typ] < TYPE_PRIORITY[dedup[key]["type"]]:
                dedup[key] = row
        rows = sorted(dedup.values(), key=lambda r: (r["start_bp"], r["end_bp"], TYPE_PRIORITY[r["type"]]))
        out.extend(rows)
    for i, row in enumerate(out):
        row["candidate_order"] = i
        row["candidate_id"] = "%s:%d-%d:%s" % (row["chromosome"], row["bead_start"], row["bead_end"], row["type"])
    return out


def sampled_candidates(rows, seed: int, cap: int = 2000):
    if len(rows) <= cap:
        return [dict(r) for r in rows], {"sampled": False, "original_count": len(rows), "selected_count": len(rows)}
    strata = {}
    for row in rows:
        strata.setdefault((row["chromosome_index"], row["type"]), []).append(row)
    for key, values in strata.items():
        def h(row):
            payload = (f"P9016-057-B\0{seed}\0{row['candidate_id']}").encode("ascii")
            return hashlib.blake2b(payload, digest_size=16).digest()
        values.sort(key=lambda r: (h(r), r["candidate_order"]))
    keys = sorted(strata, key=lambda k: (k[0], TYPE_PRIORITY[k[1]]))
    selected = []; cursor = {k: 0 for k in keys}
    while len(selected) < cap:
        progressed = False
        for key in keys:
            idx = cursor[key]
            if idx < len(strata[key]) and len(selected) < cap:
                selected.append(strata[key][idx]); cursor[key] += 1; progressed = True
        if not progressed:
            break
    selected.sort(key=lambda r: r["candidate_order"])
    return [dict(r) for r in selected], {"sampled": True, "seed": seed, "original_count": len(rows), "selected_count": len(selected)}


def swap_interval(coords: np.ndarray, raw_y: np.ndarray | None, data, candidate):
    slc = data.chromosome_slice(candidate["chromosome_index"])
    a = slc.start + candidate["bead_start"]; b = slc.start + candidate["bead_end"]
    x = np.asarray(coords, dtype=np.float64).copy(); x[:, a:b] = x[::-1, a:b]
    y = None
    if raw_y is not None:
        y = np.asarray(raw_y, dtype=np.float64).copy(); y[:, a:b] = y[::-1, a:b]
    return x, y


def physics_values(coords: np.ndarray, data):
    x = np.asarray(coords, dtype=np.float64)
    l0 = float(data.l0)
    bond_num = 0.0; bond_den = 0; bend_num = 0.0; bend_den = 0
    per_chr = {}
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci); v = x[:, slc]
        d = np.linalg.norm(v[:, 1:] - v[:, :-1], axis=2) / l0
        bnum = float((np.maximum(0.75-d, 0.0)**2 + np.maximum(d-1.25, 0.0)**2).sum())
        sec = v[:, 2:] - 2.0*v[:, 1:-1] + v[:, :-2]
        benum = float((sec*sec).sum() / (l0*l0))
        bden = int(2 * max(0, v.shape[1]-1)); beden = int(2 * max(0, v.shape[1]-2))
        per_chr[ci] = (bnum, benum)
        bond_num += bnum; bond_den += bden; bend_num += benum; bend_den += beden
    return {"bond": bond_num/bond_den, "bend": bend_num/bend_den,
            "bond_den": bond_den, "bend_den": bend_den, "per_chr": per_chr}


def _prefix(matrix):
    return np.pad(np.cumsum(np.cumsum(matrix, axis=0), axis=1), ((1,0),(1,0)))


def _rect(prefix, r0, r1, c0, c1):
    return float(prefix[r1,c1] - prefix[r0,c1] - prefix[r1,c0] + prefix[r0,c0])


@dataclass
class DeltaCache:
    data: Any
    coords: np.ndarray
    p: float
    base: dict[str, Any]
    prefixes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    physics: dict[str, Any]
    partial_pairs: int

    def score(self, candidate):
        ci = candidate["chromosome_index"]; a = candidate["bead_start"]; b = candidate["bead_end"]
        pz, pl, pn = self.prefixes[ci]; n = int(self.data.n_bins[ci])
        dz = _rect(pz, 0,a,a,b) + _rect(pz,a,b,b,n)
        dlog = _rect(pl,0,a,a,b) + _rect(pl,a,b,b,n)
        pairs = int(round(_rect(pn,0,a,a,b) + _rect(pn,a,b,b,n)))
        noff = float(self.data.raw_cis_offdiag + self.data.raw_inter)
        dcount = (noff * math.log1p(dz / float(self.base["Zall"])) - dlog) / float(self.data.raw_records)
        changed, _ = swap_interval(self.coords, None, self.data, candidate)
        slc = self.data.chromosome_slice(ci); v = changed[:, slc]; l0 = float(self.data.l0)
        d = np.linalg.norm(v[:,1:]-v[:,:-1],axis=2)/l0
        bnum = float((np.maximum(0.75-d,0)**2+np.maximum(d-1.25,0)**2).sum())
        sec=v[:,2:]-2*v[:,1:-1]+v[:,:-2]
        benum=float((sec*sec).sum()/(l0*l0))
        oldb, oldbe=self.physics["per_chr"][ci]
        dbond=(bnum-oldb)/self.physics["bond_den"]
        dbend=(benum-oldbe)/self.physics["bend_den"]
        return {**candidate, "delta_Zall": dz, "delta_sum_C_log_rate": dlog,
                "updated_cis_pairs": pairs, "delta_count_Nraw": dcount,
                "delta_bond": dbond, "delta_bend": dbend, "delta_repulsion": 0.0,
                "delta_p_prior": 0.0, "delta_full_J": dcount+dbond+0.01*dbend}


def build_cache(data, coords, p: float, *, device="cuda", base_record=None):
    coords=np.asarray(coords,dtype=np.float64)
    obj=SharedCaptureObjective(data,"G",weights=WEIGHTS,mode="V0-fixed-production-e",device=device,
                               pair_block=262_144,inner_cap=80,cg_cap=80,profile_warm_start=True,known_e=None)
    raw=torch.as_tensor(contact_model.sphere_inverse(coords),dtype=torch.float64,device=device)
    x=obj._physics._map_raw(raw); pt=torch.as_tensor(float(p),dtype=torch.float64,device=device)
    prefixes={}; partial=0
    for ci in range(len(data.chromosome_names)):
        slc=data.chromosome_slice(ci); n=slc.stop-slc.start
        li,lj=np.triu_indices(n,1); gi=li+slc.start; gj=lj+slc.start
        ti=torch.as_tensor(gi,dtype=torch.long,device=device); tj=torch.as_tensor(gj,dtype=torch.long,device=device)
        kernels,_=obj._kernel_block(x,ti,tj,pt,with_gradient=False)
        kaa,kab,kba,kbb=kernels[:4]
        ep=obj._fixed_e[ti]*obj._fixed_e[tj]
        r=ep*(0.5*pt*(kaa+kbb)+0.5*(1-pt)*(kab+kba))
        rs=ep*(0.5*pt*(kab+kba)+0.5*(1-pt)*(kaa+kbb))
        rv=r.detach().cpu().numpy(); rsv=rs.detach().cpu().numpy()
        # Full aggregate is upper triangular in the same global i,j order.
        mask=(np.asarray(data.pair_i)==gi[:,None]) if False else None
        flat=gi*int(data.n_loci) - gi*(gi+1)//2 + (gj-gi-1)
        counts=np.asarray(data.counts,dtype=np.float64)[flat]
        dz=np.zeros((n,n)); dl=np.zeros((n,n)); ones=np.zeros((n,n))
        dz[li,lj]=rsv-rv; dl[li,lj]=counts*np.log(rsv/rv); ones[li,lj]=1.0
        prefixes[ci]=(_prefix(dz),_prefix(dl),_prefix(ones)); partial += len(li)
    if device=="cuda": torch.cuda.synchronize()
    if base_record is None:
        base_record,_=fixed_state_fullgrid(data,coords,p,keep_inter_arrays=False,include_penalties=True)
    return DeltaCache(data,coords,float(p),base_record,prefixes,physics_values(coords,data),partial)


def full_objective_cpu(data, coords, p):
    obj=SharedCaptureObjective(data,"G",weights=WEIGHTS,mode="V0-fixed-production-e",device="cpu",
                               pair_block=262_144,inner_cap=80,cg_cap=80,profile_warm_start=True,known_e=None)
    raw=contact_model.sphere_inverse(np.asarray(coords,dtype=np.float64)); q=float(contact_model.q_from_p(float(p)))
    theta=np.concatenate((raw.ravel(),np.asarray([q])))
    value,_,comp=obj.evaluate(theta,need_gradient=False)
    return float(value),comp
