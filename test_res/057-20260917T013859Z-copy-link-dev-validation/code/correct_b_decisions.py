"""用已保存 exact full-G best 核对值重新派生首轮门，不重算GPU。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; RUN=HERE.parent
sys.path.insert(0,str(HERE))
from b_core import json_dump

def main():
    path=RUN/"round1_results.json"; payload=json.loads(path.read_text())
    for row in payload["results"]:
        actual=row["best_full_verification_actual"]
        accepted=actual["delta_full_J"]<=-1e-8 and actual["delta_count_Nraw"]<=1e-10
        reason="accepted_both_gates" if accepted else ("no_op_best_full_J_gate" if actual["delta_full_J"]>-1e-8 else "no_op_best_count_gate")
        row["accepted"]=accepted; row["decision_reason"]=reason; row["decision_basis"]="exact_full_G_checked"
    payload["paired_training_triggered"]=any(r["accepted"] for r in payload["results"])
    payload["decision_logic_correction"]={"status":"applied_without_rescan","candidate_argmin_basis":"cached_exact_incremental_full_J",
      "acceptance_gate_basis":"stored exact full-G checked best deltas","additional_gpu_calls":0,
      "formal_scan_values_changed":False}
    json_dump(path,payload)
    print(json.dumps({"trigger":payload["paired_training_triggered"],"decisions":[{"seed":r["seed"],"accepted":r["accepted"],"basis":r["decision_basis"],"actual":r["best_full_verification_actual"]} for r in payload["results"]]},sort_keys=True))
if __name__=="__main__": main()
