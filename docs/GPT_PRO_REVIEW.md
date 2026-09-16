# GPT Pro review guide

Reviewer-facing companion to this snapshot. It states exactly what was measured, on which
denominator, under which controls, and what is *not* established — then gives a copy-paste prompt
(§8) for a code-grounded review.

- **Repository:** https://github.com/skelviper/phase_restart
- **Snapshot entry point:** [`../README.md`](../README.md)

Everything below is quoted from files in this snapshot. Where a number comes from a run readout, the
file is named. This guide does not introduce new claims.

---

## 1. Object of study

Reconstruct the **diploid** 3D structure of **one** mouse single cell (`P9016`) from contact data
with the phase/haplotype columns removed: 20 chromosomes, 40 copy trajectories (2 per chromosome),
fitted jointly in one shared nuclear volume, genome-wide (intra- and inter-chromosomal contacts).

- **Cohort / replicate level:** one cell. The 20 chromosomes are **associated measurements within
  that single cell**, not biological replicates. No result here supports a population claim.
- **Training uses the full contact set** (no train/test split in the current Reconstruction V1);
  hyperparameters are chosen by label-free count likelihood only.
- **Reference-structure usage is per-experiment, not repo-wide.** On the blind path the reference is
  used post-hoc only: not for source selection, fitting, regularisation, hyperparameter choice,
  stopping or candidate selection, and it is read only after candidate coordinates are written and
  hashed. That applies to the current baseline and to the 049 losses A/B/C. **It does not apply to the
  whole repository:** one condition, `051 C-reference-beads`, is deliberately *not* blind — see §4.1.
- **Inter-chromosomal contacts are used for spatial placement but never as haplotype accuracy
  evidence.** Their same-/different-haplotype split is `70,558 / 76,438`, i.e. they carry **no
  maternal/paternal label signal**.

### 1.1 Documentation tense — do not read stale "current" sentences as present state

This project freezes historical documents byte-for-byte, so several files still describe the state of
the round they were written for. **Their wording is not the current state of the work.**

| Document | Historical framing | Reality |
| --- | --- | --- |
| `docs/REVIEW-s2-observation-model.md` | Opens by stating that S2 does not exist and that the latest run is 017 | **Stale.** Later stages exist; rounds 045 and 049 implement the shared-capture and max-contact objectives. |
| `docs/PROJECT_CONTEXT.md` | Stage coverage runs mainly to round 022 | **Stale as an index.** Rounds 045 / 046 / 049 / 051 / 055 are not covered. |
| `docs/legacy_workspace_README.md` | The workspace README at snapshot time; names the `real-extension-G-full-J` endpoint as the working baseline | **Stale baseline.** The registered baseline is `P9016-046-G-random-base-1Mb` (`docs/CURRENT_BASELINE.md`). |
| `docs/legacy_workspace_AGENTS.md`, `docs/MEASURED_FACTS.md`, `docs/POST020_*`, `docs/PLAN-*`, `docs/RECONSTRUCTION_V1_*` | Round-anchored rules, evidence logs and plans | Historical context; where they conflict with a current round's `config.json`, the round wins. |

**The current focus is rounds 045 / 046 / 049** (fit and evaluation implementations, including the
loss variants A/B/C) **and 051 / 055** (baseline readouts and the latest frozen comparison). The
registered baseline is in `docs/CURRENT_BASELINE.md` and `docs/current_baseline.json`.

This snapshot is provided **for reading only**. Neither the reference structure nor any baseline
coordinate file is included, and none is needed to review the code, the objective definitions or the
recorded metrics.

## 2. Observation model and exact denominators

From `test_res/049-.../gates/gate8_lineage_and_denominators.json`:

| Quantity | Value | Meaning |
| --- | --- | --- |
| `N_raw` | `1,703,888` | raw contact records after the 7-column phase-free conversion |
| same-bin (zero distance) | `438,774` | independent saturated nuisance layer (`diag_nll`); carries no structural information; excluded from structural metrics |
| `N_off` | `1,265,114` | off-diagonal records entering the count objective |
| — `cis_offdiag` | `696,680` | |
| — `inter` | `568,434` | |
| full grid `n_pairs` | `3,496,690 = C(2645,2)` | all 1 Mb bin pairs, **including zero-count pairs** |
| `count_mode` | `raw_integer` | observed counts `C` are kept as integers, never rescaled by the max branch |

Conservation identity: `1,703,888 = 438,774 + 696,680 + 568,434`.

The objective is evaluated on the **full grid including zero-count pairs**; the observed term is
`C · log(r_obs)` with `C` unchanged. Within the 049 max-contact losses (B/C) there is **no posterior
threshold screening**: no contact is dropped or reweighted by its posterior probability, and no
per-contact max-posterior filter is applied. The max only changes which rate enters the likelihood —
counts and support stay as they are.

## 3. The three losses actually implemented (049 round)

Defined in `test_res/049-20260915T162917Z-max-contact-unified-multiscale/source/max_contact_objective.py`
(`LOSS_SPEC`, lines 28–32), with `r_sum = e·Σ_s w_s K_s` and `r_max = e·max_s w_s K_s` over the four
copy states:

| Loss | Normaliser `Z` | Observed term | Semantics |
| --- | --- | --- | --- |
| **A** `marginal_G` | four-state sum (`r_sum`) | `r_sum` | the 045 original `G`, used as the parity reference |
| **B** `hard_observed` | four-state **sum** (`r_sum`) | four-state **max** (`r_max`) | observed term scored by the highest-rate state |
| **C** `max_rate` | four-state **max** (`Zmax`) | four-state **max** | `max` enters both the normaliser and the observed term |

Precise wording matters here:

- In **B and C the observed term** is the max over the four copy-state rates of the *same* bin pair.
  When the maximum branch is unique this is numerically equivalent to a **hard-MAP plug-in
  likelihood** — but no per-contact assignment file is ever produced. The loss takes the max of the
  rate directly; counts and support are unchanged.
- **B keeps the four-state sum as the normaliser**; only **C** also puts `max` into the normaliser.
- Ties are handled by `_tie_weights` (lines 61–66): exactly tied states **split the subgradient
  equally**, recorded as
  `exact_tie_rule = "equal split subgradient over exactly tied states; no first-argmax"` (line 349).
- Diagnostics report `gamma_s = t_s / Σ_s t_s`, which is a **model-posterior attribution** over the
  `1,265,114` counts — **not** a true allele assignment, and it does not enter the loss.

## 4. Current baseline and the comparison of record

Registered in `docs/CURRENT_BASELINE.md`; machine record in `docs/current_baseline.json`.

`P9016-046-G-random-base-1Mb` — candidate `real-G-random` under `base_remaining`, model `G`,
objective `full-J`, gauge `fixed_e`, lineage `612 + 404 + 486 = 1502` FG from a label-free random
start (seed 2207), terminal state **`budget_not_converged`** (`fg_budget_exhausted`).
3DG SHA256 `ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b`.
The reference was not used for source selection, fitting or stopping; the choice of this endpoint as
baseline was a **user decision after evaluation**, not a blind source selection.

Metric definition (identical in all rows below) — from `055/eval/summary.json`:

- Support: the frozen legacy mask `046/evaluation_final/results/frozen_legacy_mask_snapshot.npz`
  (SHA256 `9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9`), intersected with the
  finite beads of all four datasets in both copies. **One common denominator for all groups**:
  `157,529` pairs, `dropped = 0`. Support per chromosome is reported in
  `055/eval/per_chromosome.tsv`.
- Distance: Euclidean distance in the 1 Mb coordinates, keyed on numeric mask positions (not
  compressed indices, no string comparison).
- Four correlations per chromosome: `A_mat`, `A_pat`, `B_mat`, `B_pat`
  (`A`/`B` = candidate copies; `mat`/`pat` = the reference's two true haplotypes).
- `direct = (A_mat + B_pat)/2`, `swapped = (A_pat + B_mat)/2`,
  `same = max(direct, swapped)`, `cross = min(direct, swapped)`, `contrast = same − cross`.
  Orientation is **one whole-chromosome A/B swap**, not per-bin relabelling; `|direct − swapped| ≤
  1e-12` is recorded as `unresolved_tie`.
- **`same`/`cross` are `matched`/`swapped`.** They are a selected/subleading pair and must always be
  reported together. `contrast` is the swap-invariant readout. `same` alone is not a result.
- The reference self-control has `same ≡ 1` by construction (self-correlation degeneracy) and is a
  structural reference row only — never fit-quality evidence.

Readouts (`055/eval/summary.json`, 20 chromosomes, all defined):

| Group | same mean / median | cross mean / median | contrast mean / median |
| --- | --- | --- | --- |
| Reference (self) | 1.000000 / 1.000000 | 0.261784 / 0.286217 | 0.738216 / 0.713783 |
| **Baseline `046-G-random`** | **0.602361 / 0.617525** | 0.365617 / 0.348783 | 0.236743 / 0.257475 |
| B `hard_observed` | 0.452346 / 0.426026 | 0.353771 / 0.350388 | 0.098575 / 0.066040 |
| C `max_rate` | 0.542721 / 0.518449 | 0.393174 / 0.414706 | 0.149547 / 0.155479 |

Signed Spearman means: Reference `1.000000 / 0.251047`; Baseline `0.604876 / 0.377608`
(contrast `0.227268`); B `0.459902 / 0.371749` (`0.088152`); C `0.549744 / 0.412545` (`0.137199`).

chr1 (`17,578` common pairs, 193 mask bins): Reference `same 1.000000 / contrast 0.658221`;
Baseline `0.699845 / 0.303606` (orientation `swapped`); B `0.552730 / 0.121127` (`direct`);
C `0.712885 / 0.225587` (`swapped`).

### 4.1 Reference usage is per-experiment: the `051 C-reference-beads` exception

Do not apply a repo-wide "reference is never used" rule. Two different things are going on:

- **Blind path.** The current baseline, the 014/020 blind source selection and the 049 losses A/B/C do
  not use the reference for selection, fitting, regularisation, hyperparameters, stopping or candidate
  choice. Readouts such as 046 and 055 are evaluations of frozen endpoints, not selections.
- **Working comparator.** Registering `P9016-046-G-random-base-1Mb` as the baseline was a
  **post-evaluation user choice**, made to have a fixed comparison point. It is a legitimate working
  comparator; it is **not** a blind source-selection claim and should not be reported as one.
- **Explicitly non-blind and authorised: `051 C-reference-beads`.** This condition uses a
  **reference-derived per-copy bead presence mask** as its support. `test_res/051-.../config.json` →
  `conditions.reference-beads` records the authorisation verbatim:
  `"reference chr/position/presence only; reference xyz values never used for initialization, target,
  or regularization"`. So reference *presence* information shapes the support, while reference
  *coordinates* never enter the fit. It is a **reference-informed support control**, it is declared as
  such, and it must be labelled non-blind rather than read either as a blind method or as a blind
  method that leaked.

Report the blind/non-blind label per experiment, not for the repository as a whole.

Reproduction status: `055/eval/validation.json` records **12/12 PASS**, including value-by-value
agreement (max abs diff `< 1e-9`) with the frozen 049 `r2_per_chromosome.tsv`, the frozen 051
`per_chromosome.tsv`, and the 053 Spearman table, using a **independently written evaluation script**
(`055/source/evaluate_review.py`) that re-parses the 3DG files and recomputes Pearson/Spearman itself.
Support total `157,529` matches the mask `common` total with `dropped = 0`; copy-swap invariance is
verified per chromosome. The published baseline same-mean is `0.6023605799695414`; the independent
recomputation gives `...5413`, a `1e-16` floating-point summation-order difference, recorded as PASS.

## 5. What is NOT established

Read this section before drawing any conclusion from §4.

1. **L2 is not proven.** "One consistent copy identity is recoverable along a whole chromosome" has
   not been demonstrated. A label-free criterion ranking a chromosome-wide consistent split above a
   pre-registered global mis-joining is an **evaluation** result, not a blind recovery.
2. **L3 is retracted and must not be restated.** "No SNP-free method can work" is not supported. The
   existing evidence rejects specific methods in the current loop, not the class of nonlinear
   solvers.
3. **No endpoint converged.** All endpoints in the 049/055 comparison are `budget_not_converged`
   with the reason `fg_budget_exhausted`; the final gradients are far above `1e-6`. **Exit code 0 is
   not scientific success.**
4. **The B/C-vs-baseline comparison is not equal-cost and not single-factor.** Baseline is a
   `5 → 2 → 1 Mb` multiscale extension (1502 FG including coarse layers); B is a consensus start
   with the raw solver on the full 1 Mb grid (1502 FG); C is a random start with the
   multiscale-preconditioned solver on the full 1 Mb grid (1502 FG). **Start point, solver and layer
   structure all differ**, so the differences in §4 **cannot be attributed to the loss definition
   alone.**
5. **n = 1 cell.** No bootstrap, no significance test, no R1/R3 readouts in the 055 round; it is a
   descriptive comparison of frozen endpoints.
6. **`same`/`cross` without `contrast` is not interpretable**, and gains over a random split of the
   same contacts are required before any structural-recovery claim.
7. **Scale is uncalibrated.** Compare structures by rank or Procrustes; high within-copy correlation
   does not imply that homologous copies were recovered.

## 6. Registered historical deviations (must travel with the results)

From `test_res/055-.../README.md` §7 and `test_res/049-.../README.md` §6.4:

1. **Three registered evaluation-side deviations in round 049:** (a) the pre-gate read the frozen
   mask's *metadata* before the intended point (no coordinate/truth leakage, but a protocol
   deviation); (b) `049/config.json` had a one-character typo in the mask SHA, so evaluation attempt 1
   raised `frozen mask snapshot hash mismatch` and its stderr was overwritten by attempt 2 — the
   original traceback was **not** reconstructed, `open_attempt_count = 2`; (c) 36 derived `npz_path`
   fixes across 12 fits, each deep-compared before writing with **0 numeric change** and no metric
   recomputation. **Round 049 is therefore not a clean audit trail**, even though its frozen numbers
   reproduce.
2. **Baseline documentation was internally inconsistent.** `docs/CURRENT_BASELINE.md` (2026-09-16)
   and `051/config.json` define the baseline as `base_remaining/coords/real-G-random/1Mb.3dg`
   (SHA `ee5eb154...`), while the older `docs/current_baseline.json` (2026-09-15) still listed
   `real-extension-G-full-J` (SHA `4301d4df...`, matched `0.602331`). The values differ by ~3e-5 but
   the endpoints are different objects. **In this snapshot `docs/current_baseline.json` has been
   rebuilt to the current base endpoint with per-field provenance**; the workspace copy was left
   untouched.
3. **Display-scale correction (display only).** The chr1 heatmap scale was changed from whole-cell
   `Rg` to the historical `build_matrix` per-dataset chr1 two-copy merged raw median. This does **not**
   change `same`/`cross`/`contrast`.
4. **Figure annotation revision** (2026-09-16T11:31Z) changed the missing-value annotation from
   matrix-entry counts to per-bin counts (`Missing bins: 4 / 3`). No metric, colour scale, layout or
   support set changed, and no evaluation was re-run.
5. **`docs/RECONSTRUCTION_V1_PROTOCOL.md` is hash-locked** by `pr/reconstruct.py:421-423` and
   `pr/continuation.py:650-658`. Its bytes must not be edited.

## 7. Static checks performed on this snapshot

| Check | Result |
| --- | --- |
| Byte identity of copied sources | manifest lists **754 copied files**; **753 hash-verified byte-identical** to their workspace originals; **1 excluded from the byte check because it was rebuilt at an existing path** (`docs/current_baseline.json`). `mismatch = 0`, `missing = 0`, `unexpected = 0`. The two renamed legacy copies are additionally verified byte-identical to the workspace `README.md` / `AGENTS.md`, outside the manifest count. |
| Forbidden data files | no `*.3dg`, `*.npz`, `*.npy`, `*.h5`, `*.pkl`, `*.gz`, `*.pairs*`, checkpoint or raw-data file present; no file exceeds 1.2 MB |
| Binary exceptions | exactly 2 PNG figures, both from round 055 |
| Credential scan | 0 hits: no `ghp_`/`gho_`/`ghs_`/`github_pat_`, no private keys, no cloud keys, no `.env`/`.netrc`/`*.pem`, no secret assignments; no credential value was printed at any point |
| Python syntax | 494 files parsed with `ast.parse` — **no training, fitting or evaluation was executed** |
| Import closure | 494 files parsed; **0 unresolved local imports**. Every import that is neither stdlib nor a declared external package (`numpy`, `scipy`, `matplotlib`, `plotly`, `PIL`, `torch`, `threadpoolctl`) resolves inside the snapshot |
| Dynamic `sys.path` chain | `049/source/frozen_imports.py` present, and all 12 modules it injects (`045/source/frozen_035`, `frozen_037`, `frozen_pr`, root `pr`) are present |
| Absolute paths | 125 files contain `/mnt/ssd/zliu/...` or `/work/phase...`; retained deliberately (see README §8) — provenance, not runnable configuration |

Reproducing any *number* from this snapshot additionally requires the data listed in README §4, which
is not distributed here.

## 8. Review prompt (copy-paste)

> You are reviewing a computational-biology codebase, not writing a paper summary. All code,
> protocols and metric definitions referenced below are in this repository. Read the files before
> making a claim, and cite `path:line` for every code-level statement.
>
> **Context you must accept as given, not re-derive:**
> - Task: reconstruct the diploid 3D structure of **one** mouse cell (P9016) from contacts with the
>   phase/haplotype columns removed: 20 chromosomes, 40 copy trajectories, fitted jointly in one
>   shared nuclear volume, genome-wide, inter-chromosomal contacts included.
> - Data: `N_raw = 1,703,888` records; same-bin/zero-distance `438,774` are a saturated nuisance
>   layer excluded from structural metrics; `N_off = 1,265,114` (`cis_offdiag 696,680` +
>   `inter 568,434`) enter the count objective; the full 1 Mb grid is `3,496,690 = C(2645,2)` pairs
>   including zero-count pairs. Counts are raw integers; in the 049 max-contact losses (B/C) **no
>   contact is threshold-screened by posterior probability and none is dropped**.
> - Inter-chromosomal contacts split `70,558` same-haplotype / `76,438` different-haplotype, i.e. they
>   carry **no parent-of-origin label signal**; they constrain spatial placement only.
> - **Reference usage is per-experiment, not repo-wide.** On the blind path (current baseline, 014/020
>   blind source selection, 049 losses A/B/C) the reference is used post-hoc only — never for source
>   selection, fitting, hyperparameters, stopping or candidate choice. Registering the current
>   baseline was a **post-evaluation user choice** for a working comparator, not a blind selection.
>   **One condition is deliberately non-blind and authorised:** `051 C-reference-beads` uses a
>   reference-derived per-copy bead *presence* mask as support (`config.json` →
>   `conditions.reference-beads.support_source_authorized`: "reference chr/position/presence only;
>   reference xyz values never used for initialization, target, or regularization").
>   Using a **declared** presence-only mask is therefore not by itself an unauthorised leak — do not
>   report it as a protocol violation on that basis alone, and do not generalise it to the blind
>   experiments. **But do not take the declaration on trust either: independently verify in the code
>   that the implementation actually respects this boundary** (that no reference xyz, distance or
>   ordering information reaches initialization, target, regularisation or stopping), and **report any
>   use beyond the declared boundary as a concrete finding with `path:line` evidence.** Where a file's
>   declared scope and the code's actual behaviour disagree, the code is the evidence.
> - The whole study is **n = 1 cell**; the 20 chromosomes are associated measurements within that
>   cell, not biological replicates.
> - Current baseline `P9016-046-G-random-base-1Mb` (see `docs/CURRENT_BASELINE.md`,
>   `docs/current_baseline.json`) is terminal **`budget_not_converged`** (`fg_budget_exhausted`).
> - **L2 (recovering one consistent copy identity along a whole chromosome) is NOT proven**, and
>   **L3 ("no SNP-free method can work") has been retracted** — do not treat either as settled.
> - The 055 comparison of losses B/C against the baseline is **not equal-cost and not
>   single-factor**: start point, solver and multi-scale layer structure all differ, so its
>   differences cannot be attributed to the loss definition alone.
> - **Documentation tense:** historical documents are frozen and some open with stale framing —
>   `docs/REVIEW-s2-observation-model.md` says there is no S2 and the latest run is 017, and
>   `docs/PROJECT_CONTEXT.md` covers mainly up to round 022. Do **not** treat those "current"
>   sentences as present state. The current focus is rounds **045 / 046 / 049** (fit and evaluation
>   implementations) and **051 / 055** (baseline readouts and the latest comparison); the registered
>   baseline is in `docs/CURRENT_BASELINE.md`. Where an old document conflicts with a current round's
>   frozen `config.json`, the round wins.
>
> **Your task.**
> A. Identify the **most consequential concrete problems** in the current approach, with evidence from
>    the code and from the recorded metrics. At minimum examine: the objective/normaliser definitions
>    in `test_res/049-.../source/max_contact_objective.py`; the shared-capture objective and
>    controller in `test_res/045-.../source/`; the fit driver and continuation logic in
>    `pr/reconstruct.py` and `pr/continuation.py`; the data path and folding in `pr/pairs7.py`,
>    `pr/genome.py`, `pr/contact_model.py`; the evaluation definitions in
>    `pr/reconstruction_evaluate.py`, `pr/allele_r2.py` and
>    `test_res/055-.../source/evaluate_review.py`. State, for each problem, what observation would
>    distinguish it from a competing explanation, and whether the existing recorded readouts already
>    do so.
> B. Propose **2–3 minimal, falsifiable next experiments**, ordered by expected payoff. "Run more
>    iterations" or "tune hyperparameters" is not an acceptable answer on its own. For each one
>    specify exactly:
>    - **Question / hypothesis** and the **falsifying outcome** (what result would kill it);
>    - **cohort and unit**: which chromosomes / which bin size / which cell, and the observation unit;
>    - **denominator**: the exact support set and the number of pairs or records;
>    - **metric**: the primary readout, and the required accompanying readouts (if you use
>      `same`/`matched`, you must also give `cross` and `contrast`, and the random-split control);
>    - **control / comparator**: what it is compared against, including the `u = 0` and random-`u`
>      controls where the `2^20` sign symmetry `u_c → −u_c` applies, and an oracle upper bound where
>      an accuracy is reported;
>    - **budget and stopping rule**: iteration/function-evaluation budget and the terminal-state
>      criteria; say explicitly how "not converged" will be reported;
>    - **biological replicate level**: state whether the claim is within-cell (technical/structural
>      variation only) or requires additional cells, and do not upgrade the former into the latter.
> C. Say explicitly where the current code **cannot** answer the question even in principle, and what
>    minimal code change would be required.
>
> **Rules.** Do not present L3 as a conclusion. Do not report a local or oracle-level match as
> "the haplotype split was recovered". Do not propose work whose only outcome measure is a lower loss
> on the same data with no held-out or reference-independent criterion. Prefer deleting or replacing
> a component over adding one. Keep the answer concrete and prioritised; depth on two or three items
> beats a broad list.
>
> **You may — and should — question the project's methodological assumptions**, not only its
> implementation: the objective design, the choice of denominators and support sets, the definitions
> of `same`/`cross`/`contrast` and R1/R2/R3, the blind/non-blind framing, and the way L1/L2 are posed.
> If you think an assumption is wrong, say so and give the evidence or the experiment that would
> settle it.
>
> **Distinguish declarations from code.** A docstring, a `config.json` field, a README paragraph or a
> validation flag is a *claim*; it is not by itself evidence that the implementation does what it
> says. Verify against the code and the recorded numbers, and flag any place where the two diverge.
> Conversely, do not assume a declaration is false without showing where the code contradicts it.

## 9. Read order (concrete files)

| Step | File | Why |
| --- | --- | --- |
| 1 | `docs/GPT_PRO_REVIEW.md` (this file) | definitions, denominators, limits |
| 2 | `docs/CURRENT_BASELINE.md`, `docs/current_baseline.json` | what the baseline actually is |
| 3 | `docs/PROJECT_CONTEXT.md` | stage history and applicable scope |
| 4 | `docs/MEASURED_FACTS.md` | evidence log and protocol corrections |
| 5 | `docs/legacy_workspace_AGENTS.md` | original project rules and reporting floor (Chinese) |
| 6 | `run.py`, `pr/paths.py`, `pr/genome.py`, `pr/pairs7.py` | driver, constants, data path and folding |
| 7 | `pr/contact_model.py`, `pr/allele_models.py` | the observation model |
| 8 | `pr/reconstruct.py`, `pr/continuation.py` | the fit target, the solver loop, the multi-scale chain |
| 9 | `test_res/045-.../source/shared_capture_objective.py`, `test_res/049-.../source/max_contact_objective.py` | loss variants A/B/C and the max-branch handling |
| 10 | `pr/reconstruction_evaluate.py`, `pr/allele_r2.py`, `test_res/055-.../source/evaluate_review.py` | how `same`/`cross`/`contrast` and R1/R2/R3 are computed |
| 11 | `test_res/055-.../eval/{summary.json,per_chromosome.tsv,validation.json}`, `test_res/049-.../evaluation/results/r2_per_chromosome.tsv` | the measurements |
| 12 | `test_res/049-.../gates/gates_report.json` | 9/9 loss/gradient/gauge/denominator gates |
| 13 | `test_res/051-.../evaluation/pearson_summary.json` | baseline readouts and shared-support audit |
