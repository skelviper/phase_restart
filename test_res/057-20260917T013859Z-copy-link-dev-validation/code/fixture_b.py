"""057 B的CPU fixture与正式候选清单冻结。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent; RUN=HERE.parent; ROOT=RUN.parents[1]
S045=ROOT/"test_res/045-20260915T073310Z-shared-capture-round/source"
for p in (HERE,ROOT,S045):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import data_io
from b_core import (WEIGHTS, array_hash, build_cache, candidate_list, full_objective_cpu,
                    json_dump, sampled_candidates, swap_interval)
from shared_capture_objective import SharedCaptureObjective
from pr import contact_model
from pr.solver_state import (export_present_3dg, load_solver_state, sha256_file,
                             write_presence_mask, write_solver_state)


def inter_rates(data, coords, p):
    """直接调用原 SharedCaptureObjective kernel 的 inter 1/4 mixture。"""
    obj=SharedCaptureObjective(data,"G",weights=WEIGHTS,mode="V0-fixed-production-e",device="cpu",
                               pair_block=262_144,inner_cap=80,cg_cap=80,profile_warm_start=True,known_e=None)
    raw=torch.as_tensor(contact_model.sphere_inverse(np.asarray(coords,dtype=np.float64)),dtype=torch.float64)
    x=obj._physics._map_raw(raw)
    all_i=np.asarray(data.pair_i,dtype=np.int64); all_j=np.asarray(data.pair_j,dtype=np.int64)
    mask=np.asarray(data.locus_chromosome)[all_i]!=np.asarray(data.locus_chromosome)[all_j]
    i=torch.as_tensor(all_i[mask],dtype=torch.long); j=torch.as_tensor(all_j[mask],dtype=torch.long)
    kernels,_=obj._kernel_block(x,i,j,torch.as_tensor(float(p),dtype=torch.float64),with_gradient=False)
    rate=(obj._fixed_e[i]*obj._fixed_e[j])*0.25*sum(kernels[:4])
    return rate.detach().cpu().numpy().astype(np.float64)


def main():
    config=json.loads((RUN/"config_B.json").read_text())
    frozen={
      ROOT/"test_res/056-20260916T152353Z-pro-review-experiments/inputs/G-original_1000000_aggregate.npz":config["frozen_hashes"]["train_aggregate"],
      ROOT/"test_res/028-20260913_151456-020-gpu-independent/source/gpu_backend.py":config["frozen_hashes"]["gpu_backend_028"],
      ROOT/"test_res/045-20260915T073310Z-shared-capture-round/source/shared_capture_objective.py":config["frozen_hashes"]["shared_capture_objective"],
    }
    for path,expected in frozen.items():
        if sha256_file(path)!=expected: raise RuntimeError(f"frozen SHA mismatch: {path}")
    train=data_io.load_aggregate(next(iter(frozen)))
    rows=candidate_list(train)
    manifests={}
    for seed,sampling_seed in ((560101,570201),(560102,570202)):
        selected,meta=sampled_candidates(rows,sampling_seed,2000)
        payload={"schema":"p9016-057-B-candidates-v1","seed":seed,"sampling_seed":sampling_seed,
                 "generation_count":len(rows),"selection":meta,"candidates":selected}
        path=RUN/f"candidates_seed{seed}.json"; json_dump(path,payload)
        manifests[str(seed)]={"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),
                              "count":len(selected),"sampling":meta}
    json_dump(RUN/"candidate_manifest.json",{"status":"frozen_before_scan","by_seed":manifests})

    names=("chrA","chrB"); lengths=(4_000_000,3_000_000)
    ci=np.asarray([0,0,0,0,0,1,1,1,0,0,1,0],dtype=np.int64)
    p1=np.asarray([0,0,1e6,2e6,3e6,0,0,1e6,1e6,2e6,2e6,3e6],dtype=np.int64)
    cj=np.asarray([0,0,0,0,0,1,1,1,1,1,0,1],dtype=np.int64)
    p2=np.asarray([0,1e6,2e6,3e6,3e6,0,1e6,2e6,0,1e6,1e6,2e6],dtype=np.int64)
    base=contact_model.aggregate_from_arrays(names,lengths,ci,p1,cj,p2,1_000_000)
    e=np.asarray([0.65,0.8,1.1,1.45,0.7,1.0,1.3]); e=e/e.mean()
    data=replace(base,exposure=e,exposure_mode="fixture_nonuniform_fixed"); data.assert_consistent()
    rng=np.random.default_rng(570001); coords=rng.normal(size=(2,data.n_loci,3))*0.12
    raw=contact_model.sphere_inverse(coords); coords=contact_model.sphere_forward(raw)
    candidate={"chromosome_index":0,"chromosome":"chrA","type":"fixture","start_bp":1_000_000,
               "end_bp":3_000_000,"bead_start":1,"bead_end":3,"candidate_order":0,"candidate_id":"fixture"}
    checks=[]
    for p in (0.24,0.8):
        old,comp=full_objective_cpu(data,coords,p)
        cache=build_cache(data,coords,p,device="cpu",base_record={"Zall":comp["sum_rate_offdiag"]})
        inc=cache.score(candidate); changed,_=swap_interval(coords,None,data,candidate)
        new,newcomp=full_objective_cpu(data,changed,p)
        err=abs((new-old)-inc["delta_full_J"])
        if err>1e-9: raise RuntimeError(f"fixture incremental mismatch p={p}: {err}")
        whole={**candidate,"bead_start":0,"bead_end":4,"candidate_id":"whole"}
        whole_coords,_=swap_interval(coords,None,data,whole)
        whole_value,_=full_objective_cpu(data,whole_coords,p)
        whole_err=abs(whole_value-old)
        inter_err=float(np.max(np.abs(inter_rates(data,coords,p)-inter_rates(data,changed,p))))
        direct=np.all(changed[0]==coords[0],axis=1)&np.all(changed[1]==coords[1],axis=1)
        swapped=np.all(changed[0]==coords[1],axis=1)&np.all(changed[1]==coords[0],axis=1)
        if whole_err>1e-12 or inter_err>1e-12 or not np.all(direct|swapped):
            raise RuntimeError("fixture invariance failed")
        checks.append({"p":p,"incremental_full_G_abs_error":err,"whole_chromosome_full_J_abs_error":whole_err,
                       "inter_rate_max_abs_error":inter_err,"unordered_pointsets":True,
                       "delta_full_direct":new-old,"delta_full_cached":inc["delta_full_J"]})
    q=float(contact_model.q_from_p(0.24)); theta=np.concatenate((raw.ravel(),np.asarray([q])))
    fdir=RUN/"fixtures/state"; state_path=fdir/"solver_state.npz"; mask_path=fdir/"presence_mask.npz"; export_path=fdir/"export.3dg"
    if state_path.exists():
        loaded=load_solver_state(state_path)
        if not (np.array_equal(loaded["coordinates"],coords) and np.array_equal(loaded["raw_y"],raw)
                and np.array_equal(loaded["theta"],theta) and loaded["p"]==0.24 and loaded["q"]==q):
            raise RuntimeError("existing fixture solver state identity mismatch")
        state={"path":str(state_path),"sha256":loaded["sha256"],**loaded["audit"],"readback":loaded["audit"]}
        with np.load(mask_path,allow_pickle=False) as payload: existing_mask=np.asarray(payload["presence"])
        if not np.array_equal(existing_mask,np.ones((2,data.n_loci),dtype=bool)): raise RuntimeError("existing fixture mask mismatch")
        mask={"path":str(mask_path),"sha256":sha256_file(mask_path),"shape":list(existing_mask.shape),"present":int(existing_mask.sum()),"absent":0}
        export={"path":str(export_path),"sha256":sha256_file(export_path),"rows":int(existing_mask.sum()),"bin_size":int(data.bin_size),"track_count":len(data.track_specs)}
    else:
        state=write_solver_state(state_path,coordinates=coords,raw_y=raw,theta=theta,p=0.24,q=q)
        mask=write_presence_mask(mask_path,np.ones((2,data.n_loci),dtype=bool),data.n_loci)
        export=export_present_3dg(export_path,data,coords,np.ones((2,data.n_loci),dtype=bool))
    before=sha256_file(state_path); after=load_solver_state(state_path)["sha256"]
    if before!=after: raise RuntimeError("fixture export changed solver state")
    result={"schema":"p9016-057-B-fixture-v1","status":"PASS","checks":checks,
            "candidate_generation_count":len(rows),"candidate_manifests":manifests,
            "state":state,"mask":mask,"export":export,"solver_sha_unchanged_after_export":True,
            "sphere_max_radius":float(np.linalg.norm(coords,axis=2).max()),"reference_opened":False,"development_opened":False}
    json_dump(RUN/"fixture_results.json",result)
    print(json.dumps({"status":"PASS","candidate_count":len(rows),"checks":checks},sort_keys=True))

if __name__=="__main__": main()
