"""057实验B初始完整核对与首轮精确缓存扫描（训练侧专用）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent; RUN=HERE.parent; ROOT=RUN.parents[1]
SOURCE=ROOT/"test_res/056-20260916T152353Z-pro-review-experiments"
S045=ROOT/"test_res/045-20260915T073310Z-shared-capture-round/source"
for p in (HERE,ROOT,S045,SOURCE/"code"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import data_io
from b_core import build_cache, json_dump, swap_interval
from nonref_core import fixed_state_fullgrid
from pr.solver_state import load_solver_state, sha256_file


def timed_full(data,coords,p):
    a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    h=time.perf_counter(); a.record(); record,_=fixed_state_fullgrid(data,coords,p,keep_inter_arrays=False,include_penalties=True); b.record(); b.synchronize()
    return record,time.perf_counter()-h,float(a.elapsed_time(b))/1000.0


def main():
    config=json.loads((RUN/"config_B.json").read_text()); data=data_io.load_aggregate(SOURCE/"inputs/G-original_1000000_aggregate.npz")
    results=[]; ledger=[]
    for seed in config["seeds"]:
        candidates_payload=json.loads((RUN/f"candidates_seed{seed}.json").read_text())
        if sha256_file(RUN/f"candidates_seed{seed}.json") != json.loads((RUN/"candidate_manifest.json").read_text())["by_seed"][str(seed)]["sha256"]:
            raise RuntimeError("candidate manifest changed")
        fit=f"G-original-seed{seed}"; state=load_solver_state(SOURCE/"states/experiment3"/fit/"1Mb/solver_state.npz")
        coords=state["coordinates"]; p=state["p"]
        base,hw,gw=timed_full(data,coords,p); ledger.append({"seed":seed,"role":"round1_base_full_G","logical_fullgrid":1,"physical_sweeps":2,"host_wall_seconds":hw,"gpu_event_seconds":gw})
        stage=json.loads((SOURCE/"stages"/fit/"1Mb.json").read_text()); old=stage["endpoint"]
        component_errors={"total":abs(base["total_normalized"]-float(old["total"])),
                          "count":abs(base["count_nll_per_raw_record"]-float(old["components"]["count_nll_normalized"])),
                          "bond":abs(base["penalties_raw"]["bond"]-float(old["components"]["bond"])),
                          "bend":abs(base["penalties_raw"]["bend"]-float(old["components"]["bend"])),
                          "repulsion":abs(base["penalties_raw"]["repulsion"]-float(old["components"]["repulsion"])),
                          "p_prior":abs(base["penalties_raw"]["p_prior"]-float(old["components"]["p_prior"]))}
        if max(component_errors.values())>=1e-10: raise RuntimeError(f"056 endpoint component mismatch seed {seed}: {component_errors}")
        whole={"chromosome_index":0,"chromosome":"chr1","bead_start":0,"bead_end":int(data.n_bins[0])}
        whole_coords,_=swap_interval(coords,None,data,whole); whole_score,hw,gw=timed_full(data,whole_coords,p)
        ledger.append({"seed":seed,"role":"initial_whole_chr_full_G","logical_fullgrid":1,"physical_sweeps":2,"host_wall_seconds":hw,"gpu_event_seconds":gw})
        whole_errors={k:abs(float(whole_score[k])-float(base[k])) for k in ("Zall","count_nll_per_raw_record","total_normalized")}
        whole_errors.update({"penalty_"+k:abs(whole_score["penalties_raw"][k]-base["penalties_raw"][k]) for k in ("bond","bend","repulsion","p_prior")})
        if max(whole_errors.values())>=1e-10: raise RuntimeError(f"whole chromosome invariance failed seed {seed}: {whole_errors}")
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); h=time.perf_counter(); a.record()
        cache=build_cache(data,coords,p,device="cuda",base_record=base); b.record(); b.synchronize(); cache_host=time.perf_counter()-h; cache_gpu=float(a.elapsed_time(b))/1000
        ledger.append({"seed":seed,"role":"round1_partial_cis_cache","logical_fullgrid":0,"physical_sweeps":0,"partial_kernel_pairs":cache.partial_pairs,"host_wall_seconds":cache_host,"gpu_event_seconds":cache_gpu})
        h=time.perf_counter(); scored=[cache.score(row) for row in candidates_payload["candidates"]]; scan_wall=time.perf_counter()-h
        for rank,row in enumerate(sorted(scored,key=lambda x:(x["delta_full_J"],x["candidate_order"])),1): row["rank_full_J"]=rank
        best=min(scored,key=lambda x:(x["delta_full_J"],x["candidate_order"]))
        changed,_=swap_interval(coords,None,data,best); check,hw,gw=timed_full(data,changed,p)
        ledger.append({"seed":seed,"role":"round1_best_full_G","logical_fullgrid":1,"physical_sweeps":2,"host_wall_seconds":hw,"gpu_event_seconds":gw})
        actual={"delta_full_J":check["total_normalized"]-base["total_normalized"],
                "delta_count_Nraw":check["count_nll_per_raw_record"]-base["count_nll_per_raw_record"],
                "delta_bond":check["penalties_raw"]["bond"]-base["penalties_raw"]["bond"],
                "delta_bend":check["penalties_raw"]["bend"]-base["penalties_raw"]["bend"],
                "delta_repulsion":check["penalties_raw"]["repulsion"]-base["penalties_raw"]["repulsion"],
                "delta_p_prior":check["penalties_raw"]["p_prior"]-base["penalties_raw"]["p_prior"]}
        verify={k:abs(actual[k]-best[k]) for k in actual}
        if max(verify.values())>=1e-9: raise RuntimeError(f"best candidate incremental mismatch seed {seed}: {verify}")
        accepted=actual["delta_full_J"]<=-1e-8 and actual["delta_count_Nraw"]<=1e-10
        reason="accepted_both_gates" if accepted else ("no_op_best_full_J_gate" if actual["delta_full_J"]>-1e-8 else "no_op_best_count_gate")
        out=RUN/f"scan_round1_seed{seed}.tsv"
        fields=["rank_full_J","candidate_order","candidate_id","chromosome","type","start_bp","end_bp","bead_start","bead_end","bead_count","updated_cis_pairs","delta_Zall","delta_sum_C_log_rate","delta_count_Nraw","delta_bond","delta_bend","delta_repulsion","delta_p_prior","delta_full_J"]
        with out.open("x",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields,delimiter="\t",extrasaction="ignore"); w.writeheader(); w.writerows(sorted(scored,key=lambda x:x["candidate_order"]))
        results.append({"seed":seed,"candidate_count":len(scored),"candidate_manifest_sha256":sha256_file(RUN/f"candidates_seed{seed}.json"),
                        "base":base,"source_component_errors":component_errors,"whole_chr_errors":whole_errors,
                        "partial_kernel_pairs":cache.partial_pairs,"candidate_scan_cpu_wall_seconds":scan_wall,
                        "best":best,"best_full_verification_actual":actual,"best_full_verification_abs_errors":verify,
                        "accepted":accepted,"decision_reason":reason,"decision_basis":"exact_full_G_checked","scan_tsv":str(out.relative_to(ROOT)),"scan_tsv_sha256":sha256_file(out)})
    physical=sum(x.get("physical_sweeps",0) for x in ledger); logical=sum(x.get("logical_fullgrid",0) for x in ledger)
    payload={"schema":"p9016-057-B-round1-v1","status":"complete","results":results,
             "paired_training_triggered":any(r["accepted"] for r in results),"reference_opened":False,"development_opened":False}
    json_dump(RUN/"round1_results.json",payload)
    json_dump(RUN/"ledger_B_round1.json",{"status":"complete","logical_fullgrid_calls":logical,"physical_fullgrid_sweeps":physical,
                                         "physical_cap":32,"entries":ledger,"training_fg":0})
    print(json.dumps({"status":"complete","trigger":payload["paired_training_triggered"],"decisions":[{"seed":r["seed"],"accepted":r["accepted"],"reason":r["decision_reason"],"best":r["best"]} for r in results]},sort_keys=True))

if __name__=="__main__": main()
