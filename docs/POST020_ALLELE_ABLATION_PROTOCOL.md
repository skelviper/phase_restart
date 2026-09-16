# POST020 allele ablation protocol

**Protocol version:** `1.0`  
**Status:** `frozen_before_fit`  
**Scope:** P9016 SNP-free allele-signal ablation, synthetic calibration, and isolated real-data R2 evaluation  
**Authoritative machine file:** [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)

本文件把 020 之后的后续实验冻结为可执行 protocol。它不是结果报告，也不是 fit 授权本身：当前没有新 real fit、synthetic fit 或 real R2 evaluation；没有在本次冻结工作中读取真实 reference、真实 phase 或新的 coordinates。025 只到 prepare-only，不能写成 native fit 已完成。

## 1. Freeze and release gate

任何 native prepare 或 joint fit 之前，必须在实际 run 目录写入：

```text
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-{real|synthetic}/freeze/POST020_ALLELE_ABLATION_FREEZE.json
```

native prepare 的启动前置是本 machine JSON、输入/source hashes，以及该阶段已生成的 prepared graph hashes（若已到 x0 阶段则记录 locked x0 hash）。protocol MD/plan 的真实 mtime/hash 在文档可用时追加到 append-only registry；若它们在 native prepare 之后才完成，必须记录真实时间和 hash，不能回填成早于 native 的版本，也不阻塞 native 启动。任何后续 joint fit/evaluation 仍须保留完整的实际 hash/mtime 记录。

freeze JSON 与同一记录必须追加到：

```text
test_res/POST020_ALLELE_ABLATION_ARTIFACT_REGISTRY.jsonl
```

registry 是 append-only：旧记录不覆盖；实现 bug 的重试追加新记录并同时保留原/新 source hash。run ID 尚未分配时，只使用上述 path pattern，不猜已有 fixture/run 文件。协议文档 hash 使用：

```bash
sha256sum docs/PLAN-post020-allele-signal.md \
  docs/POST020_ALLELE_ABLATION_PROTOCOL.md \
  docs/POST020_ALLELE_ABLATION_PROTOCOL.json
```

当前 machine JSON 已先严格解析并标为 `frozen_before_fit`；本 MD 的补写不改变任何方法、seed、预算或执行状态。不能以文件 mtime 代替 freeze，也不能把后补的 MD/plan 说成 pre-fit。

## 2. Historical motivation and actual progress

原计划的动机保留：020 相比 Softall 的整体 geometry 相似度较低，但 allele contrast 较高；需要区分 bend、sphere 参数化/可行域和 exposure 的影响，优先验证 allele-specific signal，而不是只追求与 reference 的整体 Spearman。当前新增 protocol 将每项因素单独消融，并用独立 synthetic negative/positive fixture 识别数值失败模式。

### 2.1 022 continuation

022 是计划 B 的固定 warm continuation：同一 020 `random_joint` 1 Mb endpoint 增加 240 accepted steps，累计到 480；不是独立初始化重复。

| quantity | 020 original | 022 continuation |
|---|---:|---:|
| total J | 9.61001577125 | 9.60361804197 |
| count NLL | 9.59358593129 | 9.58710105238 |
| accepted steps | 240 | 480 cumulative |
| status | budget-limited | `budget_not_converged` |

FDG1000full 提案原 J 未通过 label-free 规则，保留为 `rejected_no_label_free_improvement`，不把 rejected endpoint 当作新成功条件。

### 2.2 023 fixed-endpoint diagnostics

结果见 [`test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)。023 没有启动新 fit、没有读取 reference/phase；最终 bash75 和 validator 均 exit 0。

| diagnostic | 020 random 1 Mb | 022 continuation 480 |
|---|---:|---:|
| max radius | 0.65161710 | 0.68574970 |
| beads `>=0.90` | 0 | 0 |
| beads `>=0.95` | 0 | 0 |
| beads `>=0.99` | 0 | 0 |
| minimum sphere radial attenuation | 0.43646488 | 0.38556995 |
| minimum sphere tangential attenuation | 0.75854806 | 0.72783744 |
| weighted bend y-gradient L2 | 0.01216713 | 0.01206299 |
| count y-gradient L2 | 0.03243707 | 0.03234493 |
| weighted bend/count y-gradient ratio | about 0.375 | about 0.373 |
| total gradient L2 | 0.00620367 | 0.00514436 |
| last-20 accepted total-J decrease | 0.00091717 | 0.0003898645 |

没有 bead 接近 0.90/0.95/0.99，只能排除这些 endpoint 的 active hard-wall occupancy；不能说 sphere 参数化无优化影响。radial/tangential attenuation 仍改变梯度传递，bend=.01 也有实际梯度作用。

实际 repulsion hinge threshold 是 `.7*l0 = 0.04017409936`。020 旧 config 的 `repulsion_radius_l0=2.0` 是元数据字段，native `d_r2.0` 是另一单位；三者不能混用。023 保存 source 与当前 `pr` 有差异时，历史重现以保存代码的数值结果为准。020 六层和 022 均为固定预算停止，不称充分收敛；本 protocol 保留 480 accepted 的固定有限预算，不自动加量。

### 2.3 024 R2-derived evidence and correction

结果见 [`test_res/024-20260913_133109-r2-allele-signal-derived/README.md`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md)。024 只从既有四 rho 派生 R2，没有新 fit，也没有打开真实 reference、phase 或 contacts。

- 020 original 相对 Softall：contrast `+0.044957`，95% CI `[-0.007420,+0.096044]` 跨 0；matched 变化 `-0.055800`。
- 020 original 与 continuation 都是 20/20 finite；两者均为 `both-positive=13/20`、`one-negative=7/20`；minmargin 分别为 `.049677/.047710`。
- continuation 相对 020 的 contrast 为 `-0.000222`，基本持平。
- 父侧发现 fixed-reference 列锚定 bug：swapped 时应交换 candidate 行，不应交换 reference mat/pat 列。024 v2 现已按此修复。已有 matched/cross/contrast/minmargin/counts 不受影响；旧 v1 的 absolute per-reference margin 或 contribution share 仍不引用，除非在 v2 下重新生成。

024 的 best-overall-swap 选择偏置必须保留在解释中；它不能被包装成无偏的 recovery score。

### 2.4 025 prepare-only preflight

实际目录为 [`test_res/025-20260913_135100-random-native-fullgrid-preflight/`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/)。父侧确认 3 native bundles、count conservation 和 63 graph-swap checks 已通过，但没有调用 `hk_fdg`。这是输入图和 canonicalization 的 prepare-only 证据，不是 native fit、coordinate endpoint 或 allele recovery 结果。

## 3. Frozen real cohort and observation unit

- Biological sample: P9016, one biological cell; biological replicate level = 1。
- Training input: `inputs/P9016.snpfree.pairs.gz`。
- Training SHA256: `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`。
- Training side never sees `phase0`, `phase1`, `phase_prob00..11`, `p_gen`, reference 3DG, or 014 coordinates.

### 3.1 Raw-record accounting

```text
raw total              = 1,703,888
cis total              = 1,135,454
inter                   =   568,434
same-bin diagonal       =   438,774
cis off-diagonal        =   696,680
structural              = 1,265,114
1,703,888 = 438,774 + 696,680 + 568,434
```

The likelihood observation unit is the aggregated unordered genomic bin-pair count `C_ij`. Repeated raw records at the same pair are summed. Same-bin records remain as an independent per-bin saturated nuisance layer; they are not geometry terms.

### 3.2 Full-grid convention

- Resolution: 1 Mb only; origin 0.
- For chromosome length `L`, use `ceil(L/B)` bins with the terminal partial bin retained, `bin = position // B`.
- 20 chromosomes, 40 tracks, 2,645 loci per copy, 5,290 physical beads。
- Complete unordered eligible set, including zero-count pairs: `3,496,690` pairs。
- Real zero eligible pairs: `3,009,436`。
- Structural raw groups: cis-offdiag `696,680`, inter `568,434`。
- Same-bin nuisance bins: `2,645`。
- No legacy 3 Mb offset in training; no record is dropped to match a legacy coordinate inventory.

All eligible pairs and all zero pairs remain in rate normalizers. The implementation must assert raw-record conservation and preserve all normalizers.

### 3.3 Fixed scales and count model

```text
l0 = (2*N_loci)^(-1/3) = 0.05739157051286577
r0 = 2*l0            = 0.11478314102573153
epsilon              = 1e-6
repulsion threshold  = .7*l0 = 0.04017409936
```

The finite contact kernel is:

```text
K(d) = epsilon + (1-epsilon)*(1 + d^2/r0^2)^(-2)
```

For cis off-diagonal pairs:

```text
rate_ij = e_i*e_j * [
    p/2       * (K(X_i,X_j) + K(Y_i,Y_j))
  + (1-p)/2  * (K(X_i,Y_j) + K(Y_i,X_j))
]
```

For inter pairs:

```text
rate_ij = e_i*e_j/4 * [
    K(X_i,X_j) + K(X_i,Y_j) + K(Y_i,X_j) + K(Y_i,Y_j)
]
```

`p` is one genome-wide cis nuisance, initialized at `.75` and parameterized as:

```text
p = p_floor + (1 - 2*p_floor)*logistic(q)
p_floor = 1e-4
p_prior_strength = 1e-4
q_init = q_from_p(.75)
```

The candidate-dependent conditional count NLL profiles the fixed total in each structural group. The diagonal layer is a separate per-bin saturated nuisance. Zero-count pairs remain in the group rate sums. Weights are fixed: count=1, bond=1, repulsion=1, p-prior=1, bend=.01 except C1.

## 4. Five frozen real variants

The real experiment is `5 variants x 3 paired starts = 15` 1 Mb-only joint fits. C0 is a new 1 Mb start using the V1 model; it is not a replay of the 020 5 Mb -> 2 Mb -> 1 Mb path.

| arm | only registered change | parameterization / exposure | status |
|---|---|---|---|
| C0 | none relative to new V1 1 Mb model | current sphere / observed-endpoint | planned, not run |
| C1 | bend `.01 -> 0`; bond retained | same sphere / observed-endpoint | planned, not run |
| C2-map | replace map only by identity-core `.90` smooth ball map | physical `R<1` / observed-endpoint | planned, not run |
| C2-free | direct physical x, finite-only, no hard ball, no clip/rescale | direct Cartesian / observed-endpoint | planned, not run |
| C3 | exposure only set to full-grid ones; endpoint audit and normalizers retained | same sphere / ones | planned, not run |

### 4.1 Exposure

For C0/C1/C2-map/C2-free, use the fixed observed-endpoint exposure:

```text
e_i = sqrt(endpoint_count_i + 10)
      / mean_full_grid(sqrt(endpoint_count + 10))
```

It uses all raw endpoints, including same-bin records, and is computed without phase or reference. It is not optimized. C3 uses `e_i=1` on every full-grid locus, while still retaining the endpoint audit and every normalizer.

### 4.2 C2-map and C2-free

For C2-map, with `r=||y||`:

```text
r <= .90:  s = r
r >  .90:  t = (r-.90)/.10
           s = .90 + .10*t/sqrt(1+t^2)
           x = s*y/r                 (r=0 -> x=0)
```

Thus `R=||x||=s<1`; there is no clip or rescale. C2-free uses direct physical `x` variables with finite-only objective evaluation, no hard sphere, no clip, and no rescale. The C0 -> C2-free contrast changes both parameterization and feasible domain, so it cannot be assigned to a pure boundary cause. C2-map helps separate map sensitivity, but neither contrast is a causal proof.

No `bend=0 + free` combination is run. No `r0` search, prior search, exposure search, or reference-driven parameter search is allowed. 020 consensus remains a historical control and is not rerun.

## 5. Frozen native initialization and physical x0

### 5.1 Bundles and assignment

| bundle | assignment seed | native seed |
|---|---:|---:|
| bundle1 | 250101 | 250201 |
| bundle2 | 250102 | 250202 |
| bundle3 | 250103 | 250203 |

Assignment uses `genome.random_assignment`: both endpoints of a cis record receive the same random copy; inter endpoints receive independent random copies. No phase, `p_gen`, or 014 seed substitution is used.

### 5.2 Native bridge

Use the explicit full-grid NULL-source native bridge exactly once per seed:

```text
n_iter=1000
CPU=1
source=NULL
target_radius=10
native bend=0
```

Native defaults are otherwise unchanged. Terminal and zero-contact beads are included. C1 changes the joint model bend only; it does not change native bend.

### 5.3 Canonicalization and x0 lock

For each chromosome, sort the integer AA/BB cis off-diagonal edge signatures numerically using fixed encoding. Select the smaller byte key as canonical copy A. If the primary keys tie, use an incidence signature sorted by other chromosome, self local bin, other local bin and count while ignoring partner-copy labels. If that tie remains, mark preflight failure, retain the seed, and do not silently swap or replace it.

Each bundle has 20 single-chromosome plus one all-chromosome canonical-input byte-invariance audit, 21 per bundle and 63 total. These are graph/blob checks only; the native engine itself is not called equivariant by this protocol.

Native is finalized once over all 40 tracks. Then apply one common all-track centering and uniform scaling to max radius `.8`; save canonical labels. Hash the three physical x0 files before any joint model. All five variants for one bundle share exactly the same physical x0; their latent inverse representation may differ. Do not call native once per variant.

## 6. Real optimization budget and selection

Every real fit uses SciPy L-BFGS-B with:

```text
maxiter = 480 accepted iterations
maxfun  = 1470
maxls   = 20
ftol    = 1e-10
gtol    = 1e-6
checkpoint = every 20 accepted iterations
```

Record actual `nfev`, `njev`, every objective evaluation, line-search probe, component values, checkpoint, source hash, input hash, x0 hash and terminal reason. Do not use R2 for early stop, budget extension or candidate choice. The fixed budget is explicitly non-sufficient-convergence-prone.

Within each variant only, select a representative by final `count_nll_normalized`; ties `<=1e-12` are broken by bundle ID. All 15 outputs remain in the evaluation set. Do not rank across exposure/prior changes by J and call one variant globally best.

Statistical failures, numerical failures, rejected endpoints and `not_converged` endpoints remain in the registry and final report. Do not redraw seeds or rerun because a result is unfavorable. An implementation bug may be retried only on the same frozen data, seed, budget and variant; record both source hashes and call it a retry, never an independent repeat.

## 7. Synthetic calibration (frozen, not run)

Synthetic calibration is a numerical and failure-mode check. It is not a claim that native real initialization is recovered or biologically matched.

### 7.1 Fixtures and seeds

All fixtures use the same full header, 1 Mb grid, 20 chromosomes, 40 tracks and 5,290 points. The four truth objects are independent and must not be swapped:

| fixture | class | truth seed | draw seed | exposure seed | initial-shape seed | generating exposure |
|---|---|---:|---:|---:|---:|---|
| N1 | same internal shape, spatially separated negative | 260101 | 260201 | 260301 | 260401 | ones |
| N2 | same internal shape, spatially separated negative | 260102 | 260202 | 260302 | 260402 | lognormal sigma=.4, mean=1, no dropout |
| P1 | different-shape positive | 260103 | 260203 | 260303 | 260403 | ones |
| P2 | different-shape positive | 260104 | 260204 | 260304 | 260404 | lognormal sigma=.4, mean=1, no dropout |

For N2/P2, follow the actual helper semantics: draw `np.exp(default_rng(seed).normal(0, sigma, N))` with `sigma=0.4`, then divide by the realized full-grid mean. Do not substitute a population log mean. N1/N2 and P1/P2 are not a paired exposure-only experiment; cross-fixture differences cannot be interpreted as pure exposure effects.

### 7.2 Generation and worker isolation

Generation uses the V1 kernel and `p_gen=.8`. Per fixture, fixed integer multinomial totals are:

```text
diagonal same-bin = 438,774
cis off-diagonal  = 696,680
inter             = 568,434
raw total         = 1,703,888
```

All eligible pairs are retained. Record each fixture's actual zero count; never substitute real `3,009,436`.

For C0/C1/C2-map/C2-free, recompute observed-endpoint exposure from synthetic observed endpoints. C3 uses ones. The generating exposure is never supplied to the worker. Initial shapes independently call `generate_truth(startseed,false)` once, then share one normalization to max radius `.8`; they do not use or perturb truth. The five arms within one fixture share the same x0 and initialize `p=.75`.

Prepare and hash all truth/observed-data artifacts before fitting. Fit workers receive observed data only. Hash and lock every fit coordinate endpoint before an isolated evaluator reads truth.

There are `4 x 5 = 20` synthetic fits. Use:

```text
maxiter = 80 accepted iterations
maxfun  = 270
maxls   = 20
ftol    = 1e-10
gtol    = 1e-6
checkpoint = every 20 accepted iterations
```

### 7.3 Synthetic gates and metrics

For N, the two truth distance matrices must agree per chromosome at rtol `1e-10` and truth centers must be separated by `>1e-6`. For P, normalized truth distance matrices must differ by `>1e-6`. An invalid fixture is retained as `invalid`; its seed is not replaced.

Finite, gradient, grid, budget, code and truth-coordinate hash checks are implementation gates. Truth R2 increase is not a release gate and truth cannot select parameters. N contrast near zero is an algebraic property of the same-shape construction, not proof that a method cannot fabricate a split.

For N, define common truth per chromosome as the mean of the two truth D matrices. For every chromosome, use the same finite unordered off-diagonal mask for both candidate and truth. The candidate scale `sC` and truth scale `sT` are each shared by both copies:

```text
sC = sqrt(mean of squared distances from candidate copies A and B over the common mask)
sT = sqrt(mean of squared distances from truth copies A and B over the common mask)
```

For N, use the common-truth D matrix for both truth-copy entries when calculating `sT`. For P, use its own truth copy matrices. Report each candidate copy's shape error as `RMS(Dc_copy/sC - Dt_copy/sT)`, with per-chromosome mean and max. This preserves relative copy scale; do not rescale copies separately. Also report the negative-only diagnostic `RMS((DcA-DcB)/sC)` with the same mask and shared `sC`; it is not a positive score or selection rule. For P, report own-truth four rho, one overall orientation, contrast, both fixed-reference margins and shape error.

## 8. Isolated real R2-only evaluation

This is planned and has not run. Only after all 15 endpoints, within-variant selections, coordinates and source hashes are locked may the evaluator read:

```text
data/P9016.1m.3dg.gz
SHA256 1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29
```

No real phase pairs are read. R1/R3 are not added in this round; the evaluation scope is R2-only. Existing controls are fixed: Softall seed124101, 020 random, 022 continuation, FDGfull rejected, random014 and consensus014. They are descriptive controls, not tuning targets.

### 8.1 Common mask

Use positions `range(3Mb,L,1Mb)` and, per chromosome, one common finite unordered off-diagonal mask. Expected totals are:

```text
common mask pairs       = 157,529
total non-diagonal pairs = 176,201
```

Actual coverage must be checked. A constant metric input, any nonfinite input, or fewer than 20 common pairs for a chromosome makes that metric n/a. If an arm fails or has no valid coordinates, keep it as n/a; do not silently remove it or change its denominator. This does not authorize any local copy swap.

### 8.2 Four rho and fixed-reference orientation

Always retain raw:

```text
rho_A_mat, rho_A_pat, rho_B_mat, rho_B_pat
```

Reference columns are fixed as `ref1=mat` and `ref2=pat`. Do not swap reference columns. The two allowed whole-chromosome mappings are:

```text
direct: a=A_mat, b=A_pat, c=B_mat, d=B_pat
cross/swapped: a=B_mat, b=B_pat, c=A_mat, d=A_pat
```

For the selected orientation:

```text
matched       = (a+d)/2
cross_matched = (b+c)/2
contrast      = matched - cross_matched
margin_ref1   = a-b
margin_ref2   = d-c
minmargin     = min(margin_ref1, margin_ref2)
```

Compare the original direct and cross values once per chromosome, then take the larger as the one overall best swap. This retains the best-swap positive selection bias and must be stated in interpretation. There is no local copy repair.

If `abs(direct-cross) <= 1e-12`, mark `unresolved_tie`, retain the raw four rho, set contrast to 0, do not count `both-positive`, and set fixed-reference margins/minmargin to n/a. `both-positive` and `one-negative` are descriptive patterns affected by best-swap selection. For non-ties, `both-negative=0` follows algebraically from contrast being nonnegative; it is not success evidence.

### 8.3 Comparisons, seeds and bootstrap

Primary paired comparisons are:

```text
C1-C0
C2-map-C0
C2-free-C0
C2-free-C2-map
C3-C0
```

Then compare descriptively with the fixed historical controls. First pair each seed on the same 20 chromosomes. Then average the effect for each chromosome over all three seeds. Seeds are optimization repeats, not biological replicates; never pool 60 chromosome rows as n=60 and never report only the best seed.

A primary three-seed summary requires all planned seeds valid for that comparison. If any seed fails or has no valid coordinates, report the main three-seed summary as n/a while retaining `planned_seed_count=3` and `valid_seed_count`; an available-seed summary may be listed separately only with that label. It must not silently become a two-seed primary result.

All metrics and comparisons use the same seed-9301 `10000 x 20` bootstrap index matrix. Intervals describe within-cell structural/technical variation, not biological replication and not a p-value.

## 9. Resources and artifact layout

Observed read-only snapshot: 192 logical CPUs, MemAvailable about 335 GB, no active fit at snapshot. This is not acceleration evidence. Schedule limits are:

```text
native preparation: at most 2 workers
synthetic fits:     at most 4 workers
real fits:          at most 6 workers
global single-thread process limit: 6
BLAS=1, OMP=1
```

Before launch, record current MemAvailable and single-worker RSS. If MemAvailable is below 32 GiB, do not start a new worker; wait for running workers and do not change the optimization budget. Based on 023, one real 480-step fit historically took about 45--47 minutes; 15 real fits are about 11.4 CPU hours in total and several hours wall time. No linear wall-time speedup is promised.

Future runs use fresh directories:

```text
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-real/
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-synthetic/
```

Each run must preserve `README.md`, `config.json`, `freeze/`, `logs/`, coordinates, plots, source/input manifests, selection, termination audit and isolated evaluation. No existing source or experiment directory is modified by this document work.

Release order is fixed:

```text
machine JSON/input/source/prepared-graph hash
-> run-specific freeze JSON and registry append
-> fixture/graph/native prepare
-> physical x0 hash lock
-> synthetic fits and calibration
-> real 15 fits
-> endpoint/selection/hash lock
-> isolated real R2 evaluation
```

If the protocol MD/plan are completed after native preparation, append their actual mtime/hash without backdating or blocking native. The old machine-JSON snapshot SHA256 `6fa2f833ac4a915614b04c85577d1b7ef5528105553c4c8be69c6327af2eec92` is retained as `prefit_superseded_snapshot`; the current JSON hash is recomputed after these corrections.

## 10. Claim boundary

This freeze establishes only that the next experiments, inputs, parameterizations, seeds, budgets and readout rules are explicit and machine-readable. It does not claim synthetic calibration complete, real ablation complete, native equivariance, sufficient numerical convergence, allele recovery or L2 whole-chromosome copy identity.

Historical L1 evidence, the unresolved L2 question, and the retraction of the broad L3 claim remain separate. A positive contrast, two separated lines, lower training loss, an exit code of zero, or a single high correlation cannot by itself be reported as allele recovery.
