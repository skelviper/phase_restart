# phase_restart — external review snapshot

Source-code and protocol snapshot of a single-cell Hi-C diploid 3D reconstruction project,
prepared **by an assistant** for external review by GPT Pro. It is a curated read-only slice of a
larger private workspace, not a released software package.

- **Snapshot id:** `github-review-20260916_203503`
- **Repository:** https://github.com/skelviper/phase_restart
- **Prepared:** 2026-09-16
- **Source workspace:** `/mnt/ssd/zliu/phase_restart` (private; not included)
- **Latest results:** [`docs/PRO_REVIEW_RESULTS_056.md`](docs/PRO_REVIEW_RESULTS_056.md) — the
  preregistered exposure comparison, four endpoint table, adoption decision, and limitations.
- **Reviewer guide:** [`docs/GPT_PRO_REVIEW.md`](docs/GPT_PRO_REVIEW.md) — definitions, denominators,
  limits, evidence map and a copy-paste review prompt. Read the 056 report first, then this guide.
- **Git history:** none. This snapshot starts from a single root commit and deliberately does **not**
  carry the upstream repository's 6-commit history, which contained the raw contact file.

## 1. What this snapshot is, and what it is not

**It is** the complete research source (Python, CUDA/C, shell), the scientific protocol and boundary
documents, the frozen dependency copies that the active code imports at runtime, and a small curated
set of result readouts — enough for a reviewer to read the objective functions, the solvers, the
evaluation definitions and the headline measurements.

**It is not** a data-inclusive, one-click reproducible repository. The contact data, the reference
structure, all fitted coordinates and all checkpoints are absent. **Do not assume that training or
evaluation can be run directly from this snapshot.** See §4 for the external inputs that would be
required and §8 for portability limits.

### Provenance of the files in this snapshot

No file in the original workspace was modified, moved, or deleted while preparing this snapshot.
Three groups of files must be distinguished:

1. **Research source and protocol files — byte-identical.** Everything under `run.py`, `pr/`,
   `scripts/`, `tests/`, `native/hickit/` (sources only), `docs/` and `test_res/` was copied
   byte-for-byte and verified against source SHA256. The research code was **not** reformatted,
   re-pathed, or otherwise touched, because frozen-hash checks in the code assert those exact bytes.
2. **New organising files, added for this snapshot.** `README.md` (this file), `.gitignore`,
   `docs/GPT_PRO_REVIEW.md`, and `docs/PRO_REVIEW_RESULTS_056.md` are written for external review; they
   do not replace frozen workspace documents. The two legacy copies `docs/legacy_workspace_README.md` and `docs/legacy_workspace_AGENTS.md` are
   byte-identical copies of the workspace `README.md` and `AGENTS.md` under new paths, so that this
   snapshot's `README.md` can be the English review entry point without destroying the original text.
3. **One file rebuilt at its existing path.** `docs/current_baseline.json` was stale in the workspace
   (it still named the historical extension endpoint); it is rebuilt here in place, with per-field
   provenance, because reviewers would otherwise read the wrong baseline. Its workspace original is
   untouched. Details in §5.

The same distinction applies to the upstream `.gitignore` policy: this snapshot's `.gitignore` is a
new file that protects against data by extension/size, whereas the workspace one excluded all of
`test_res/` (which would also have dropped the research source).

## 2. Layout

```
run.py                       driver: prepare, stage1, split, reevaluate, reconstruct
pr/                          the active library (45 modules)
scripts/                     3DG comparison / plotting CLIs
tests/                       unit tests (14 modules)
native/hickit/               vendored FDG engine source + Makefile + PROVENANCE.md
docs/                        protocol, plans, measured facts, audits, Chinese reading archive
  CURRENT_BASELINE.md        the registered current working baseline
  current_baseline.json      machine-readable baseline record (rebuilt for this snapshot)
  GPT_PRO_REVIEW.md          reviewer guide: definitions, metrics, read order, review prompt
  PRO_REVIEW_RESULTS_056.md  latest preregistered exposure comparison and decision
  legacy_workspace_README.md the original Chinese workspace README, kept as historical record
  legacy_workspace_AGENTS.md the original Chinese workspace rules, kept as historical record
test_res/                    one directory per formal run; source, config, protocol and readouts only
```

`test_res/` is where most of the scientific record lives: **dozens of formal run directories totalling
about 4.8 GB** in the workspace, almost all of it coordinates, `.npz` arrays and plots. The workspace
`.gitignore` excludes `test_res/` wholesale, which would also have dropped the research source that
lives inside it; here the policy is inverted — **all research source under `test_res/` is included**,
and only data is excluded, by extension and by size.

## 3. What is deliberately excluded

| Category | Why |
| --- | --- |
| `data/P9016.pairs.gz` (raw contacts, 16.7 MB) | Out of scope for this snapshot. The scope here is **source-code review**, so raw experimental data is not redistributed. No download source and no redistribution licence were verified or asserted; that question is left entirely to the data owner. It is also the reason this snapshot carries no upstream history — the file was committed in the workspace repository's first commit. |
| `data/P9016.1m.3dg.gz` (reference structure) | Not redistributed here for the same scope reason. Where the reference may be used is a scientific-boundary question, not a redistribution one — see §5. |
| `inputs/` (derived 7-column contact file) | Regenerated by `python run.py prepare` from the raw pairs. |
| All `*.3dg`, `*.npz`, `*.npy`, `*.h5`, `*.pkl`, checkpoints | Coordinates and numeric arrays. They are the bulk of the ~4.8 GB of `test_res/`, and none of them is needed to read the code. |
| `coords/`, most `plots/`, `logs/`, `work/`, `stages/` payloads | Run products. Five reviewed figures are the only binary exceptions (below). |
| `scratch/` | Temporary work products by project policy; contains duplicated/translated copies of modules. One dependency-driven exception is included — see §9. |
| `__pycache__/`, `.pytest_cache/` | Build detritus. |
| `docs/audits/**/*.npz,*.png,*.pdf` (28 MB) | Audit binaries. The audit *text* and audit *code* are included. |

**Binary exceptions included on purpose:** five reviewed figures — the two round-055 comparison
figures and the three round-056 result figures under
`test_res/056-20260916T152353Z-pro-review-experiments/plots/`.

## 4. External inputs this code expects (not present here)

The **main entry points** resolve their inputs relative to the repository root — `pr/paths.py` defines
`ROOT`, `PAIRS`, `REF3DG`, `SNPFREE` and `HICKIT` that way, and `run.py` uses them. This is **not**
true of every file: historical and round-specific code still contains hard-coded absolute paths such
as `/mnt/ssd/zliu/...` and `/work/phase3/...` (125 files). Those paths were kept byte-identical on
purpose; see §8. The table below lists the inputs by their repository-relative identity.

| Path | Role | Present? |
| --- | --- | --- |
| `data/P9016.pairs.gz` | Raw P9016 contact file, **including the phase/haplotype columns**. `run.py prepare` reads it and writes the phase-free 7-column file; the phase columns are stripped there and are not part of the training input. Historical and evaluation code can also read phase columns, but only after candidate coordinates have been written and hashed. | no |
| `inputs/P9016.snpfree.pairs.gz` | Derived 7-column phase-free file, SHA256 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`. This is what the reconstruction training consumes; the loader rejects any file whose `#columns` header contains a phase field. | no |
| `data/P9016.1m.3dg.gz` | Reference diploid structure, SHA256 `1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`. Used for post-hoc evaluation and reporting, with one explicitly authorised, non-blind exception — see §5. | no |
| `native/hickit/hickit` | Compiled FDG binary; build with `cd native/hickit && make hickit` (needs `cc`, `make`, `zlib`) | source yes, binary no |
| `test_res/045-.../inputs/real_1000000_aggregate.npz` | 1 Mb aggregated contact counts used by the 045–055 chain | no |
| `test_res/046-.../base_remaining/coords/real-G-random/1Mb.3dg` (+ `.npz`) | The registered current baseline candidate, SHA256 `ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b` | no |
| `test_res/046-.../evaluation_final/results/frozen_legacy_mask_snapshot.npz` | Frozen evaluation mask, SHA256 `9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9` | no |

Full input lists and hashes for individual rounds are in each run's `config.json` and in
`test_res/049-20260915T162917Z-max-contact-unified-multiscale/evaluation/results/REQUIRED_INPUTS.md`.

## 5. Scientific context (minimum needed to read the code)

**Task.** Reconstruct the diploid 3D structure of one mouse single cell (P9016) from contact data
with the haplotype/phase columns removed: 20 chromosomes, 40 copy trajectories (2 per chromosome),
fitted jointly in one shared nuclear volume. Genome-wide, including inter-chromosomal contacts.

**Observation model and denominators.**

- Raw records `N_raw = 1,703,888`.
- Same-bin (zero genomic distance) records `438,774` are an independent saturated nuisance layer:
  they carry no structural information and are excluded from structural metrics.
- Off-diagonal records entering the count objective `N_off = 1,265,114`
  (`cis_offdiag = 696,680`, `inter = 568,434`).
- Full 1 Mb grid `n_pairs = 3,496,690 = C(2645,2)`, including zero-count pairs.
- Inter-chromosomal contacts split `70,558` same-haplotype / `76,438` different-haplotype, i.e. they
  carry **no maternal/paternal label signal**. They are still genuine geometric constraints used for
  spatial placement, but they are never used as haplotype accuracy evidence.

**Three claims that must stay separate.**

| Claim | Current status |
| --- | --- |
| **L1** — mixed contacts contain predictable signal correlated with the difference between the two structures | Supported on chr1; marginal on chrX |
| **L2** — one consistent copy identity is recoverable along an entire chromosome | **Not proven.** Evaluation tooling can penalise global mis-joining; that is not the same as a blind method having found a consistent split. |
| **L3** — no SNP-free method can work | **Retracted.** The evidence only rejects specific methods in the current loop, not the class of nonlinear solvers. |

L1 does not imply L2. Local correctness, global mis-joining and oracle upper bounds must never be
reported as "the split was recovered".

**Current baseline.** `P9016-046-G-random-base-1Mb`: candidate `real-G-random` under
`base_remaining`, model `G` / objective `full-J` / gauge `fixed_e`, lineage `612+404+486 = 1502` FG
from a label-free random start (seed 2207), terminal state `budget_not_converged`
(`fg_budget_exhausted`). The reference structure was not used for source selection, fitting,
hyperparameter choice or stopping.

**Headline metrics** on the frozen legacy mask (20 chromosomes, 157,529 common pairs, dropped 0),
signed Pearson of pairwise Euclidean distances:

| Group | same (mean) | cross (mean) | contrast (mean) |
| --- | --- | --- | --- |
| Reference (self-control) | 1.000000 | 0.261784 | 0.738216 |
| **Baseline `046-G-random`** | **0.602361** | 0.365617 | 0.236743 |
| B `hard_observed` | 0.452346 | 0.353771 | 0.098575 |
| C `max_rate` | 0.542721 | 0.393174 | 0.149547 |

`direct = (A_mat + B_pat)/2`, `swapped = (A_pat + B_mat)/2`, `same = max(direct, swapped)`,
`cross = min(direct, swapped)`, `contrast = same − cross`, with **one whole-chromosome A/B swap**
(not per-bin relabelling). `same`/`cross` are therefore a selected/subleading pair and must be
reported together; `contrast` is the swap-invariant readout. The reference self-control has
`same ≡ 1` by construction and is not evidence of fit quality.

### Reference-structure usage: what is blind and what is not

"Blind" means: no information derived from the reference structure influenced the result. That holds
for the main line of work, but **not for the whole repository**, and the distinction matters:

- **Blind (reference not used for selection, fitting, hyperparameters, stopping or candidate choice):**
  the current baseline (a label-free random start under the same-model count likelihood), the 014/020
  blind source selection, and the 049 losses A/B/C. The 046 and 055 readouts are evaluations, not
  selections.
- **Baseline as comparator — post-evaluation user choice.** Registering
  `P9016-046-G-random-base-1Mb` as the working baseline was a decision taken *after* evaluation, to
  have a fixed comparison point. It is **not** a blind source-selection claim, and it should not be
  read as one.
- **Explicitly not blind, and authorised: `051 C-reference-beads`.** This condition uses a
  **reference-derived per-copy bead presence mask** as its support: reference chromosome / position /
  presence only. The recorded authorisation is that **reference xyz values are never used for
  initialization, target, or regularization** — only which loci are present per copy. It is a
  *reference-informed support control*, it is declared as such in
  `test_res/051-.../config.json` (`conditions.reference-beads.support_source_authorized`), and it must
  **not** be read as a blind method, nor as a blind method that leaked.

  That declaration is a claim about the implementation, not a substitute for checking it: a reviewer
  should verify in the code that no reference xyz, distance or ordering information reaches
  initialization, target, regularisation or stopping, and should report any use beyond the declared
  boundary. Where a declared scope and the code's actual behaviour disagree, the code is the evidence.

When reading the code or the review response, apply the blind/non-blind label per experiment rather
than repo-wide.

Note the extension endpoint `real-extension-G-full-J` (3DG SHA `4301d4df...`, matched `0.602331`) is
**historical** and is not the current baseline; the workspace JSON that still listed it was stale and
is corrected in this snapshot.

## 6. Documentation tense — read this before trusting any "current" sentence

Several older documents in this snapshot describe the project state **as of the stage they were
written for**, and they say "current" or "no such feature yet" about things that are long superseded.
**Their bytes were deliberately not edited**, because this project freezes historical records. Treat
their opening framing as historical, not as present state.

| Document | What it is | What it is not |
| --- | --- | --- |
| `docs/REVIEW-s2-observation-model.md` | A stage-2-era review that opens by stating there is no S2 and that the latest run is 017 | **Not** a statement about today. Later reconstruction and shared-capture implementations exist (rounds 045 and 049). Do not read this as claiming that the originally planned S2/S3 programme was completed — only that the work moved on. |
| `docs/PROJECT_CONTEXT.md` | Stage history whose coverage runs mainly to round 022 | **Not** the current index; rounds 045/046/049/051/055 are not in it |
| `docs/legacy_workspace_README.md`, `docs/legacy_workspace_AGENTS.md` | The original workspace README and rules at snapshot time (Chinese) | Partially stale: the legacy README still names the extension endpoint as the working baseline |
| `docs/MEASURED_FACTS.md`, `docs/POST020_*`, `docs/PLAN-*`, `docs/RECONSTRUCTION_V1_*` | Evidence log and stage plans, each anchored to its own round | Not a description of the latest state |

**The latest result is round 056**, documented in `docs/PRO_REVIEW_RESULTS_056.md`. It is a frozen
80/20 record-split comparison of two exposure definitions and does **not** replace the registered
full-data baseline. The baseline remains `P9016-046-G-random-base-1Mb`; rounds 045 / 046 / 049 are
fit/evaluation implementations, while 051 / 055 remain historical baseline readouts and comparison
records in their original time scope.

## 7. Read order

1. `docs/PRO_REVIEW_RESULTS_056.md` — latest preregistered comparison, endpoint table, decision, and limits.
2. `docs/GPT_PRO_REVIEW.md` — definitions, evidence map, limits, and the review prompt.
3. `docs/CURRENT_BASELINE.md`, `docs/current_baseline.json` — the unchanged registered baseline.
4. `docs/PROJECT_CONTEXT.md` — stage history and applicable scope (**historical context only**).
5. `docs/MEASURED_FACTS.md` — evidence log and protocol corrections.
6. `docs/legacy_workspace_AGENTS.md` — original project rules: scientific boundaries and reporting floor (Chinese, partly stale).
7. Round 056 code: `test_res/056-.../code/`, plus `pr/solver_state.py` and `tests/test_solver_state.py`.
8. Earlier chain: `test_res/049-.../source/max_contact_objective.py`, `test_res/045-.../source/*`,
   and `test_res/055-.../source/evaluate_review.py`.
9. Readouts: round 056 `results/` and `reference_eval/`, then historical round 055 `eval/`.

## 8. Portability limits (intentional, do not "fix" blindly)

- **Absolute paths.** 128 files contain `/mnt/ssd/zliu/...` or `/work/phase3/...`
  (e.g. `pr/allele_r2.py`, `pr/multires_variant_runner.py`, `tests/test_allele_r2.py`). These were
  left byte-identical on purpose: some are frozen snapshots whose SHA256 is asserted by other code,
  and rewriting them would break frozen-hash checks and the project's byte-identity requirements.
  Treat them as provenance, not as runnable configuration. Main entry points are unaffected because
  `pr/paths.py` derives its paths from the repository root; the hard-coded paths belong to historical
  and round-specific scripts.
- **Environment.** Python commands and tests assume a conda environment named `analysis`
  (numpy, scipy, matplotlib). The native engine needs only `cc`, `make`, `zlib`.
- **Frozen copies.** `test_res/*/source/frozen_*/` are deliberate byte-frozen copies of earlier
  modules; `test_res/049-.../source/frozen_imports.py` injects them into `sys.path` at runtime.
  They duplicate `pr/` on purpose. Do not deduplicate them.
- **Hash-locked documents.** `docs/RECONSTRUCTION_V1_PROTOCOL.md` is locked by hash from
  `pr/reconstruct.py` and `pr/continuation.py`; its path and bytes are intentionally unchanged.
- **Reference-structure ordering.** On the blind path, the reference 3DG may only be opened after
  candidate coordinates have been written and hashed; this ordering is enforced by pre-reference hash
  gates (for example `test_res/055-.../eval/pre_reference_hash_gate.json`). It is a rule of the blind
  protocol, not a claim that the reference is absent from the whole repository — see §5.

## 9. Snapshot integrity

- **802 files, ~17.5 MiB** of content (excluding `.git`).
- The original snapshot inventory contained 754 workspace-derived files (753 byte-identical and one
  rebuilt review record, `docs/current_baseline.json`). Round 056 adds 42 byte-identical workspace
  files. Six organising files are review-only: `README.md`, `.gitignore`,
  `docs/GPT_PRO_REVIEW.md`, `docs/PRO_REVIEW_RESULTS_056.md`, and the two renamed legacy copies.
- Byte identity of the 42 round-056 additions was verified against source SHA256 after the final plot
  annotation revision: 0 mismatches.
- **Dependency closure:** all 507 Python files parse; every import that is neither standard library
  nor a declared external package (`numpy`, `scipy`, `matplotlib`, `plotly`, `PIL`, `torch`,
  `threadpoolctl`) resolves inside the snapshot after accounting for the two declared dynamic roots
  (`frozen_037` and the one retained `scratch` dependency) — 0 unresolved local imports.
- The dynamic `sys.path` entry point `test_res/049-.../source/frozen_imports.py` and all 12 modules it
  pulls in (`045/source/{frozen_035,frozen_037,frozen_pr}` plus root `pr/`) are present.
- One dependency-driven addition: `scratch/post020_real_controller.py` is included on its own because
  `test_res/029-.../provenance/post020_real_controller_v2.py` imports it. The rest of `scratch/` is
  still excluded.
- Credential scan: no tokens, keys, `.env` files or secret assignments were found in the copied set.
  The only sensitive content is local absolute paths, reported in §8. No credential value was ever
  printed during preparation.
- No file exceeds 1.2 MB; the only binaries are the five intentionally included PNG figures.
- No root `LICENSE` file is added: the research code's licensing is the owner's decision, and no
  authorship claim is implied. Vendored third-party headers retain their original notices
  (klib `khash.h`/`ksort.h`/`kseq.h`/`klist.h`/`kavl.h`: MIT, Attractive Chaos;
  `stb_image_write.h`: public domain / MIT). `native/hickit/` is the owner's own hickit fork;
  `native/hickit/PROVENANCE.md` records the per-file SHA256 of the vendored copy.
- Static checks covered syntax/AST parsing, import resolution, byte comparison, exclusions, links and
  credential patterns. Three lightweight CPU fixture modules were also run (solver-state contract,
  exact-budget accounting and tie-margin semantics). **No training, fitting or scientific evaluation
  was executed in the review snapshot.**
