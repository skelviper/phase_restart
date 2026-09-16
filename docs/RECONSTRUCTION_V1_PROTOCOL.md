# Reconstruction V1 Observation Protocol

## Status and scope

**Status: prototype / calibration pending.** This document freezes the first
continuous observation-model prototype for the P9016 Stage 2 programme. It is
not a completed reconstruction, a biological recovery claim, or authorization
to run a formal P9016 optimization. The parent-stage release is required before
any formal fit, parameter sweep, or reference-side evaluation.

The prototype reconstructs two unlabeled trajectories for every chromosome in
one shared dimensionless nuclear ball. It is intentionally separate from the
existing native-FDG and `run.py` paths. It does not read phase columns, phase
probabilities, the reference 3DG, or any derivative of either on its training
side.

## Frozen dataset and scientific unit

| Item | Frozen value |
| --- | --- |
| Biological sample | P9016, one cell; biological replicate level = 1 |
| Training file | `inputs/P9016.snpfree.pairs.gz` |
| Training SHA256 | `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa` |
| Source raw SHA256 | `071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505` |
| Raw contact budget | 1,703,888 = 1,135,454 intra + 568,434 inter records |
| Genome scope | 20 chromosome headers, two copies each, 40 trajectories |
| Training records | Every SNP-free record; no fold, phase-based filter, chromosome subset, or label-derived sampling |

The observation unit is the aggregated count `C_ij` for a genomic bin pair.
Repeated raw records at the same pair add to the same nonnegative integer
count. Initialization variation and synthetic sampling variation are technical
variation only; they are not biological replicates.

All real-data loading first verifies the SNP-free schema and the frozen output
hash. It must reject phase-bearing columns before records are interpreted.
Reference 3DG and phase data remain evaluation-only resources and must not be
used to choose a model, regularizer, initialization, stopping time, or
candidate.

## Coordinate and grid convention

At resolution `B` bp, a chromosome with header length `L` has
`ceil(L / B)` loci with bin indices `0, ..., ceil(L / B)-1`. A bin `b` denotes
the half-open genomic interval

```text
[b * B, min((b + 1) * B, L)) .
```

The terminal partial bin is present. A real record endpoint is valid only when
`0 <= pos < L` and is mapped with the frozen project convention
`bin = pos // B`. The full grid starts at genomic bin zero; it does not inherit
the legacy 3 Mb evaluation offset. Header bins without endpoints remain in the
training grid. Coordinate export records the bin start `b * B`, including the
terminal bin.

Concatenate chromosome-local bins into `N_loci` global loci in the original
SNP-free `#chromosome` header order. The aggregation stores an explicit
`track -> (chromosome_index, chromosome_name, copy_index)` mapping: tracks are
`c01a`, `c01b`, then `c02a`, `c02b`, and so on in that header order. No caller
may infer chromosome identity by lexicographically sorting track names. Every
future coordinate export writes both copies for every full-grid bin, including
zero-endpoint bins, bins before the legacy 3 Mb offset, and a terminal partial
bin. Legacy native coordinates that lack beads must be expanded by a
predeclared interpolation/endpoint-fill procedure and a predetermined small
nonzero perturbation; records are never dropped to match an old coordinate
inventory.

The structural eligible set is the complete unordered set

```text
E = { (i, j) : 0 <= i < j < N_loci } .
```

It contains intra- and inter-chromosome pairs and includes zero-count pairs.
The implementation constructs `E` from headers, never from observed records.
At every resolution it asserts exact raw-record conservation:

```text
M_all = M_diag + M_cis_offdiag + M_inter
      = 1,703,888 for the frozen P9016 file.
```

`M_diag` counts cis records with both endpoints in the same genomic bin;
`M_cis_offdiag` and `M_inter` are the other two classes. Aggregated sums over
`C_ij` must equal their corresponding raw budgets exactly.

## Coordinates and nuclear constraint

For locus `i`, copies are `X_i, Y_i in R^3`. The optimizer owns unconstrained
variables `y` and applies independently to every physical bead

```text
x = y / sqrt(1 + ||y||^2) .
```

Thus every coordinate satisfies `||x|| < 1` in the dimensionless nuclear ball.
The scale is not calibrated and must never be described as micrometers. For an
incoming coordinate gradient `g_x`, the analytic pullback is

```text
g_y = r^-1 [ g_x - x (x dot g_x) ],  r = sqrt(1 + ||y||^2) .
```

Both copies move jointly; there is no frozen consensus `Z`, no radial hard
potential, and no unconstrained coordinate export.

## Fixed exposure and bounded contact kernel

For each locus, let `endpoint_count_i` count both ends of every raw record
mapped there, including same-bin records. The copy-shared fixed exposure is

```text
e_i = sqrt(endpoint_count_i + 10) / mean_full_grid(sqrt(endpoint_count + 10)).
```

It is not optimized. This is an unlabeled working assumption, not a claim that
visibility has been separated from geometry. It can mix capture, mappability,
visibility, bin capacity, and real spatial signal. Uniform and bin-length
exposure are future sensitivity analyses, not parameters to automatically
search in this stage. Phase information must not calibrate exposure.

At a resolution layer, with all `2 * N_loci` physical beads in radius one,

```text
l0 = (2 * N_loci)^(-1/3)
r0 = 2 * l0
epsilon = 1e-6
K(d) = epsilon + (1 - epsilon) * (1 + d^2 / r0^2)^(-2).
```

The tail exponent is four. `K` is finite at zero; same-bin records nevertheless
are not used as geometry terms.

## Count model

For every eligible pair, define its rate using the four copy distances. For a
cis pair,

```text
rate_ij = e_i e_j * [
    p / 2       * (K(X_i, X_j) + K(Y_i, Y_j))
  + (1 - p) / 2 * (K(X_i, Y_j) + K(Y_i, X_j))
].
```

For an inter pair,

```text
rate_ij = e_i e_j / 4 * [
    K(X_i, X_j) + K(X_i, Y_j) + K(Y_i, X_j) + K(Y_i, Y_j)
].
```

There is exactly one genome-wide cis nuisance `p`. It initializes at `0.75`
(the value is preselected and unrelated to historical phase proportions) and
is parameterized by an unconstrained `q`:

```text
p = 1e-4 + (1 - 2e-4) * logistic(q).
```

No inter same-haplotype nuisance is fitted. The prior added to the normalized
objective is

```text
-1e-4 * log(p * (1 - p)).
```

It is symmetric and only weakly repels the bounded endpoints.

For `g in {cis_offdiag, inter}`, fix the observed group total `M_g`. Let
`R_g = sum_{(i,j) in E_g} rate_ij`. The candidate-dependent conditional count
negative log likelihood is

```text
NLL_cond = sum_g [ M_g * log(R_g) - sum_{(i,j) in E_g} C_ij * log(rate_ij) ].
```

This is the independent-Poisson likelihood after profiling each group scale,
equivalently the conditional-total multinomial form. It deliberately omits the
candidate-independent conditional count-factorial constants

```text
sum_g [ -log(M_g!) + sum_{(i,j) in E_g} log(C_ij!) ].
```

The code records this omitted quantity for audit; it never treats zero-count
pairs as absent from `R_g`.

Same-bin counts are retained as a separate nonstructural, per-bin saturated
Poisson nuisance layer. For every diagonal bin, profile its own nuisance mean
at `mu_i = C_ii`; no common diagonal exposure or geometry model is assumed.
Its complete candidate-independent NLL is

```text
NLL_diag = sum_i [ C_ii - C_ii * log(C_ii) + log(C_ii!) ],
```

where a zero-count term is defined as zero. This layer has no coordinate or
`p` gradient. It is reported only with the raw count budget; all candidates at
the same resolution have the same per-bin nuisance layer, and it is never used
to call a structural improvement. It is never substituted into `K(0)` or
silently dropped.

The normalized count contribution is

```text
(NLL_cond + NLL_diag) / M_all.
```

All compared candidates at a resolution have the same parameter dimension,
fixed priors, and `M_all` denominator. Absolute NLL values must not be compared
across resolutions because their grids and profiled count layers differ.

## Joint polymer prior

The full prototype objective is the normalized count term, the `p` prior, and
three coordinate terms. All use `l0` above and are reported separately.

```text
bond = mean_tracks_and_adjacent[
    hinge(0.75 - d/l0)^2 + hinge(d/l0 - 1.25)^2
]
weight_bond = 1

repulsion = sum_{all unordered physical bead pairs}
    hinge(1 - d/(0.7*l0))^2 / (2*N_loci)
weight_repulsion = 1

bend = mean_track_interior[ ||x_(i+1) - 2*x_i + x_(i-1)||^2 / l0^2 ]
weight_bend = 0.01.
```

Repulsion includes every pair among the `2*N_loci` physical beads: the four
copy combinations for every unordered locus pair plus the homolog pair at each
locus. Bonds and bends are only within one chromosome track. There is no
additional hard exclusion, no fixed consensus coordinate, and no silent
cross-chromosome exception.

## Optimization and implementation checks

The prototype uses SciPy L-BFGS-B as an unconstrained L-BFGS optimizer over
`y` and `q`, with analytic gradients for the contact likelihood, `p`, bond,
repulsion, bend, and sphere pullback. It scans `E` in blocks (default 65,536)
and does not form a dense `(N, N, 3)` coordinate array. The count calculation
uses a first pass for each group rate sum and observed log-rate term, followed
by a second pass for gradients. Repulsion is also block-scanned over physical
pairs.

The code exposes a 40-track coordinate export interface for future runs, but
this prototype does not modify `run.py`, native FDG, existing result folders,
or formal command-line behavior. Integer-contact FDG is outside this primary
continuous optimization loop. The L-BFGS API records actual objective/gradient
call count and elapsed time. Its callback reuses a value/components cache only
when the optimizer theta matches the most recent analytic-gradient theta
exactly; it never uses tolerance matching. An optional caller-owned
`checkpoint_hook` receives copy-safe `theta`, coordinates, components, count,
and elapsed-time snapshots at a declared interval (default 10 accepted
iterations when requested). The API itself does not write checkpoints.

## Identifiability, limitations, and reporting boundary

A whole-chromosome copy swap is an exact gauge symmetry. In addition, at
`p = 0.5`, a local bin-level A/B swap only permutes the four data terms, so the
contact data likelihood cannot distinguish it. Chain and other polymer priors
can change under that local swap. Therefore a coherent-looking trajectory is
not, by itself, evidence that contacts recovered long-range copy identity.
Report fitted `p`, its bounded parameterization, component losses, candidate
initialization, and the `p = 0.5` local-swap diagnostic. Do not use `p` or a
converged objective to claim L2 without the separately frozen evaluation.

Known limitations are: a single global cis mixture weight; fixed exposure that
can confound geometry; no genomic-distance nuisance beyond the finite kernel;
no calibrated physical length; one biological cell; and a nonconvex objective
with collapse/symmetry stationary points. The protocol neither claims that
`u = 0` is stable nor tries to invert only positive residuals.

Real P9016 likelihood must not be used to search indefinitely over model
capacity, regularization, kernels, exposures, or initialization families.
Formal complexity and regularization require a parent-approved, prespecified
budget and synthetic calibration. The reference never chooses these values.

## Planned synthetic calibration (not yet a biological result)

Synthetic truth is stored and used only on the evaluation side. Training inputs
contain only unlabeled synthetic contacts. Each condition is 20 chromosomes,
40 tracks, the same full-grid construction and total-record accounting as the
intended real resolution unless a tiny numerical fixture is explicitly labeled
as such.

1. **Noiseless expected counts:** construct an explicitly marked
   `synthetic_expected` full-grid clone with fractional expected masses. This is
   a conditional cross-entropy numerical calibration, not a Poisson probability
   model for fractional observations; count-factorial constants are therefore
   not defined or used for selection. Verify objective/gradient consistency, not
   P9016 biology.
2. **Known sampling noise with known synthetic exposure:** use the synthetic
   generating exposure in a well-specified calibration, draw the prespecified
   count process, and report all starts and component values.
3. **Realistic sparsity and capture misspecification:** repeat with the
   production observed-endpoint exposure formula while synthetic capture/
   visibility differs from it. Quantify the misspecification sensitivity rather
   than selecting the best model by truth.
4. **Two-copy-identical negative controls (two levels):** First, use exact
   `u = 0` / co-located copies only as a data-term algebra and finite-value
   control. It conflicts with the physical-bead repulsion prior, so a complete
   MAP fit is not required or expected to retain coincident beads. Second, use
   a biologically interpretable negative control in which each chromosome has
   two copies with the same internal shape but can be separated by a rigid
   translation inside the ball. The expected result is no reportable
   difference-shape or copy-identity signal; evaluation-side R2 contrast and R3
   must report the appropriate tie/unavailable state. Merely separating the two
   physical copies is not a negative-control failure.

Calibration reports copy-difference diagnostics and likelihood noise gain
alongside all candidates, but neither is used as a recovery score. Synthetic
expected-count objects must explicitly declare their count mode and
exposure mode. The well-specified condition uses a known synthetic exposure;
the misspecified condition uses the production observed-endpoint formula. Their
fractional group totals are checked with a tight declared tolerance, without
integer coercion. This API is separate from the raw loader: the real SNP-free
path remains exact-integer and retains exact endpoint conservation.

Numerical validation is distinct from biological recovery. Dense-versus-block
equivalence, finite-difference gradients, bounded coordinates, count
conservation, and short synthetic objective decrease only establish that the
prototype is internally implemented as specified. They do not establish blind
P9016 recovery, L1, or L2.

## Planned formal sequence

Two future phase-free initializations are frozen in
[`RECONSTRUCTION_V1_INITIALIZATION.md`](RECONSTRUCTION_V1_INITIALIZATION.md).
That document records the approved S0 014 source hashes, full-grid expansion,
nonzero perturbation, and warm-start policy; this observation protocol does not
repeat or alter them. The planned resolution schedule is 5 Mb to 2 Mb to 1 Mb;
both initializations reach 1 Mb, all candidates are retained, and final ranking
uses the same complete 1 Mb label-free count likelihood with fixed priors and
parameter dimension. This document does not authorize those runs.
