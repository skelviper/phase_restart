# Restricted Swap Optimization and Fixed-x p Profile 058

This is the concise external report for round 058. It follows the round-057 no-op scan by testing four
preselected copy-swap starts under short, restricted optimization and by diagnosing whether the existing
mixture parameter `p` is a material fixed-coordinate bottleneck. It does not replace the registered
full-data baseline (`P9016-046-G-random-base-1Mb`).

## Scope and decision

**Experiment A failed its training and development gates.** Four frozen Swap candidates were paired
with four Controls. Each arm received exactly 100 full value-and-gradient (FG) calls. All eight finite
endpoints ended `budget_not_converged`; every Swap had worse training `J`, training count NLL and
development NLL than its paired Control. Neither seed selected a candidate (`selected=null`), so the
development fold was not used to choose one. All four fixed-frame return checks were
`near_control=false`.

**Experiment B stopped after the fixed-x diagnostic.** Profiling `p` while holding coordinates and
exposure fixed improved training `J` by only `1.7971419907780728e-9` and
`3.943964976826919e-8`, both below the frozen `1e-5` trigger. The planned 3 x 50 FG
Control/Profile strategy comparison therefore did not run and coordinate FG was zero. This says that
`p` is not a material bottleneck at these two fixed endpoints under the profiled objective; it does not
establish a mathematical global optimum or reject every profile strategy.

The bounded conclusion is **no candidate improvement in this round**. Stop this version of A and this
round's B extension; retain model G and baseline 046. These fixed-budget, nonconverged negative results
must not be generalized to arbitrarily long optimization or all SNP-free methods. L2 remains unproven,
and no L3 claim is made.

## Experiment A: restricted full-G optimization

The sources were the round-056 `G-original` 80%-training endpoints for optimization seeds `560101`
and `560102`. Candidate 1 was the round-057 training scan's minimum full-J candidate; candidate 2 was
the minimum count candidate after excluding candidate 1, with original candidate order breaking ties.

| Seed | Slot | Frozen interval | Selection role |
| ---: | ---: | --- | --- |
| 560101 | 1 | `chr5 [130000000,151834684)` | minimum full J |
| 560101 | 2 | `chr3 [160000000,160039680)` | minimum count after exclusion |
| 560102 | 1 | `chr2 [60000000,182113224)` | minimum full J |
| 560102 | 2 | `chr3 [160000000,160039680)` | minimum count after exclusion |

For each candidate, Swap exchanged both copies' complete xyz and corresponding `raw_y` across the
half-open interval. Optimization activated all `raw_y` coordinates on that chromosome and fixed all
other chromosomes, `p`, exposure and all other parameters. Every FG reassembled the complete theta
and called the original full-G `SharedCaptureObjective`; it included all intra- and inter-chromosomal
terms and the full normalizer. The round-057 fixed-point candidate cache was never substituted after
coordinates moved. Endpoint reporting used the last accepted state, never the last rejected trial.

| Seed / slot | Train J gain, Control - Swap | Dev gain, Control - Swap | Result |
| --- | ---: | ---: | --- |
| 560101 / 1 | `-0.005118326126122952` | `-0.006709928265145493` | fail |
| 560101 / 2 | `-0.002254213701426977` | `-0.0024357897915283644` | fail |
| 560102 / 1 | `-0.0006250771850524472` | `-0.0010445821337654593` | fail |
| 560102 / 2 | `-0.0023815715204875687` | `-0.0029844823542077847` | fail |

The development threshold was `0.001`, so all four fail by sign as well as magnitude. The complete
paired values, count components, gradients, same/cross/contrast and margins are in
[`paired_A.tsv`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/paired_A.tsv).
All arms used 100 FG and ended `budget_not_converged`, with finite last-accepted endpoints. Only
seed 560101 candidate 1 had an active-gradient ratio above 10 (`42.91079905491416`); this was a
preregistered fixed-budget imbalance warning, not an added veto, and the primary gates had already
failed.

## Experiment B: fixed-x p diagnostic

B started independently from each unaltered round-056 endpoint and never consumed A. With coordinates
and exposure fixed, its cache retained the original `p` dependence, `K0 + diag` count constant,
original `p` prior and full `Z` including inter-chromosomal terms. The search evaluated a 33-point
coarse grid plus a bounded scalar interval, always included the original `p`, and capped each profile
at 128 unique scalar calls.

| Seed | p before | p after | Train J gain | Dev gain | Scalar calls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 560101 | `0.24035967246787707` | `0.2403934067333627` | `1.7971419907780728e-9` | `1.2363274315418948e-7` | 47 |
| 560102 | `0.243843226710218` | `0.24368279562818268` | `3.943964976826919e-8` | `-5.344776958793318e-7` | 46 |

The development effects differ in direction and are far below `0.001`. The four reported B endpoint
summaries are descriptive fixed-x Original/Profile states, not a completed strategy comparison;
identical coordinates make their structural deltas exactly zero. `p < 0.5` is neither an error nor an accuracy
measure. The profile's bounded numerical stop is not a claim of global optimality.

## Development distance strata

This is the round-056 already viewed 20% record fold, not a fresh test. Exposure is fixed from
training. The primary development score excludes `K0` and regularizers and uses `Noff = 253,067` with
the full `3,496,690` eligible-pair normalizer, including zero-count pairs.

Distances use actual 1 Mb genomic bin anchors, not raw-read distances or compressed indices:

| Group | Records | Eligible pairs |
| --- | ---: | ---: |
| `[1,5) Mb` | 49,424 | 10,380 |
| `[5,20) Mb` | 31,007 | 36,075 |
| `[20, infinity) Mb` | 59,129 | 137,561 |
| inter-chromosomal | 113,507 | 3,312,674 |

For group `g`, let `S_g = sum_{ij in g} Cdev_ij * log(rate_ij)`. The within-group score is
`NLL_cond,g = log(Z_g) - S_g/N_g`. The separate contribution to the global score is
`global_contribution_g = (N_g * log(Z_all) - S_g)/Noff`; the four group contributions sum to the
global NLL. These are different quantities and their signs need not agree. All four A far-
distance global contributions are negative, but that is not evidence that every within-group
conditional gain is negative. For example, seed 560101 candidate 2 has far-distance conditional gain
`+0.0013013725019881406` while its far global contribution gain is
`-0.00028677825133271995`. The complete 6 comparisons x 4 groups are in
[`dev_strata_pairs.tsv`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/dev_strata_pairs.tsv).

## Structural and identity readouts

R2 uses the round-055 fixed support: 20 chromosomes and 157,529 pairs. For each endpoint and
chromosome, one whole-chromosome Pearson-distance mapping is chosen and reused for same, cross,
contrast and both copy margins; there is no window-level truth flip. Affected-chromosome minimum
margin decreased in three A comparisons and increased in one: seed 560102 candidate 2 changed by
`+0.10186136907797211`. Overall there was no candidate improvement.

Raw phase rows and the phase-free seven columns aligned exactly for all
`1,703,888 / 1,703,888` records before applying the original fold. The development cis off-diagonal
classification contains 40,939 hard same-copy records, 1,055 hard cross-copy records and 97,566
unknown-label records; the far same-copy subset has 16,542 records across 10,320 distinct bin-pair
blocks. These hard labels have **unverified origin**: no independent provenance distinguishes direct
SNP calls from inferred or imputed labels. Therefore `R1_direct` and the identity veto are `NA`, no
block bootstrap was performed, and no alternate truth label was substituted. The 40,939 same-copy
records must not be described as verified direct-SNP truth. There is no new identity-recovery claim.

## Freeze, validation, budget, and timing

All 12 finite endpoints and their 36 state/mask/export artifacts were hashed and read back before the
phase body or reference was opened. The preformal original-G fixture reproduced cached p-profile values
and counts with maximum error `0.0`, active-gradient finite differences with maximum error
`2.5809786252661837e-10`, and the real first budgeted FG reproduced its expected initial components
with maximum error `1.7763568394002505e-15`.

- Training FG: `800 / 1400` (A 800; B coordinate optimization 0).
- Scalar calls: 93 total (47 + 46).
- Extra full-grid equivalents: `14 / 64` (2 B caches + 12 development endpoint scores); 50 unused
  when this round stopped.
- A optimizer segments: `163.6937117157504 s` total.
- B cache builds: `0.5192612949758768 s` total.
- Unified evaluation: `10.605994581244886 s` for mixed development + reference + phase work.

Each arm's initial check was its first budgeted FG; there was no extra terminal training readback.
Independent reference and SNP CPU subtimings were not recorded, so the mixed evaluation wall time must
not be presented as either standalone CPU time. No timing was rerun to fill that gap.

## Limits and evidence map

- One cell was studied; the two seeds are optimization initializations, not biological replicates.
- Every `readID` is `.`, so molecule-level isolation is unavailable.
- The baseline remains `P9016-046-G-random-base-1Mb`; rounds 056 and 057 remain historical inputs.
- Raw or derived contacts, arrays, coordinates, checkpoints, fixture binaries and full logs are not
  distributed. This snapshot is not data-inclusive or one-click reproducible.

Published round-058 evidence:

- Internal run summary: [`README.md`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/README.md)
- Frozen method and preformal gate: [`config.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/config.json), [`preformal_freeze.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/preformal_freeze.json)
- Final machine result and budget: [`FINAL_RESULTS.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/FINAL_RESULTS.json), [`budget_ledger.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/budget_ledger.json)
- Pre-evaluation freeze: [`pre_evaluation_hash_gate.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/results/pre_evaluation_hash_gate.json)
- Evaluation: [`dev_results.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/evaluation/dev_results.json), [`structural_results.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/evaluation/structural_results.json), [`phase_traceability.json`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/evaluation/phase_traceability.json)
- Complete paired exports: [`paired_A.tsv`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/paired_A.tsv), [`p_profile_B.tsv`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/p_profile_B.tsv), [`dev_strata_pairs.tsv`](../test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/dev_strata_pairs.tsv)

For the source endpoints and scan history, read
[`PRO_REVIEW_RESULTS_056.md`](PRO_REVIEW_RESULTS_056.md) and
[`PRO_REVIEW_RESULTS_057.md`](PRO_REVIEW_RESULTS_057.md). The registered baseline is documented in
[`CURRENT_BASELINE.md`](CURRENT_BASELINE.md).

## Targeted questions for GPT Pro

1. Does `code/run_A.py` implement a fair paired restricted optimization: original full G on every FG,
   identical 100-FG budgets, the intended active/fixed coordinates, and correct last-accepted endpoint
   semantics? Identify any implementation detail that could bias Control versus Swap.
2. Does `code/run_B.py` and `code/frozen_core.py` preserve all p-dependent and p-independent terms
   needed for the original fixed-x objective? Given the measured gains and derivatives, is the bounded
   conclusion that current fixed-x `p` is not a material bottleneck justified, and what does it not
   rule out?
3. Is the report's distinction between `NLL_cond,g = log(Z_g) - S_g/N_g` and
   `global_contribution_g = (N_g * log(Z_all) - S_g)/Noff` correct? How should mixed signs,
   especially the seed-560101 candidate-2 far-distance example, be interpreted without substituting one metric for the other?
4. Does the missing independent direct/inferred provenance make `R1_direct=NA`, no identity veto and no
   bootstrap the correct choice? What can and cannot be inferred from the unverified hard-label counts?
5. Based on the existing 056-058 evidence, propose **one** minimal falsifiable next experiment. It must
   state exact cohort/unit, denominator, primary metric, comparator, budget/stopping rule and the
   outcome that would kill the hypothesis. Do not propose simply rerunning completed rounds 056, 057
   or 058, and do not treat L2 as established.
