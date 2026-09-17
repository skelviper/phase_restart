# Copy-Link Development Validation 057

This is the concise external report for round 057. It adds development-validation evidence for the
eight fixed copy-link splices scored on training data in round 056, then reports a broader frozen
training-only search for a directly improving discrete swap. It is not a rerun of round 056 and does
not replace the registered full-data baseline (`P9016-046-G-random-base-1Mb`).

## Scope and decision

**Experiment A passed.** For each of the two `G-original` optimization seeds, all `8/8` fixed splices
increased development NLL relative to the original endpoint. Median increases were
`0.00904292036645149` (seed `560101`) and `0.00929004905928732` (seed `560102`) nat per off-diagonal
contact. The corresponding round-056 values, `0.00916178115686872` and `0.00948635806784548`, were
**training** splice medians. Round 057 supplies the missing development-validation check; it does not
retrain or repeat the round-056 fits.

**Experiment B found no acceptable first-round discrete swap.** The complete training-only scan
ranked 1,992 frozen candidates for each seed. The best candidate for each seed failed both frozen
acceptance thresholds, so both decisions were no-op and both modified-chromosome sets are empty. The
preregistered branch rule stopped the experiment at that point: Control/Swap continuation was not
started, no training FG was spent, and paired predictive gains and structural differences are `NA`,
not zero.

The bounded conclusion is that this direct decrease-style candidate family produced no improving
first-round swap under the frozen objective and gates. It does **not** show that discrete swaps are
worse than continued optimization, because that paired comparison was never triggered. It does not
falsify all SNP-free methods. L2 remains unproven, and no L3 claim is made.

## Experiment A: development validation

The development set is the fixed 20% record-level fold already inspected in round 056, not a fresh
test set and not a molecule-isolated split. Every source `readID` is `.`, so only record-level
separation is available. Training exposure `e`, coordinates `x`, and endpoint parameter `p` remain
fixed; only the counts used for scoring change.

- Primary denominator: `Noff_dev = 253,067` off-diagonal records.
- Full normalizer: all `3,496,690 = C(2645,2)` eligible 1 Mb pairs, including zero-count pairs.
- Primary score: `log(Z_full) - sum(Cdev * log(rate)) / 253067`.
- Neither `K0` nor regularizers enter this primary score.
- Formal work: 18 contact full-grid forwards and 0 training FG.
- Measured A timing: host wall `2.180476164445281 s`; CUDA event `2.042949600219726 s`.

The two seeds are optimization initializations, not biological replicates. Machine evidence is in
[`results.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/results.json),
[`summary.tsv`](../test_res/057-20260917T013859Z-copy-link-dev-validation/summary.tsv), and
[`ledger.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/ledger.json).

## Experiment B: complete frozen candidate scan

Candidates were generated from the true chromosome lengths and full header grid for all 20
chromosomes. Starts advance every 5 Mb, with suffix, 5 Mb, 10 Mb, and 20 Mb half-open intervals;
empty bead intervals and whole-chromosome gauge swaps are excluded and duplicate bead intervals are
removed by frozen type priority. This produced 1,992 candidates, below the 2,000 cap, so no sampling
occurred. Both seed manifests retain the complete selected candidate list.

For each seed, the training-side procedure first took the global argmin of cached exact-incremental
`delta_full_J` across all 1,992 candidates, then checked the selected candidate with exact full-G.
Acceptance required both `delta_full_J <= -1e-8` and `delta_count/Nraw <= 1e-10`; failure produced a
no-op, with no search for a secondary candidate.

| Seed | Ranked best interval | Exact full-G delta J | Exact delta count / Nraw | Decision |
| ---: | --- | ---: | ---: | --- |
| 560101 | `chr5 [130000000,151834684)` bp, suffix | `+0.06216137971916602` | `+0.0035478624511817713` | rejected, no-op |
| 560102 | `chr2 [60000000,182113224)` bp, suffix | `+0.016196467499737466` | `+0.004579903299607224` | rejected, no-op |

The complete 1,992-row-per-seed scans are
[`scan_round1_seed560101.tsv`](../test_res/057-20260917T013859Z-copy-link-dev-validation/scan_round1_seed560101.tsv)
and
[`scan_round1_seed560102.tsv`](../test_res/057-20260917T013859Z-copy-link-dev-validation/scan_round1_seed560102.tsv).
Candidate identities and hashes are recorded in
[`candidate_manifest.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/candidate_manifest.json).

## Validation, budget, and terminal state

- Real-data cached-incremental versus exact-full maximum absolute error was
  `2.3141211169530607e-15`, below the frozen `1e-9` tolerance.
- The corrected bounded-kernel inter-rate fixture maximum error was
  `1.1102230246251565e-16`. Solver-state consistency, sphere-domain checks, and immutable export all
  passed.
- B used 6 logical full-G calls / 12 physical sweeps against a cap of 32, plus 368,032 partial-cis
  kernel pairs and 0 training FG.
- Candidate-scan CPU wall times were `0.14991418551653624 s` and `0.09520479291677475 s`.
- Measured B timing was host wall `2.4834352554753423 s` and CUDA event
  `2.3344471130371094 s`.

These timings cover different instrumented operation segments. They are not end-to-end runtimes and
do not support an equal-wall-time performance comparison.

An internal review corrected the acceptance decision to use the already stored exact full-G values
rather than cached values; the formal decisions and conclusions did not change, and no additional
real-data forward call was made. The fixture's inter-rate helper was also aligned to the original
bounded kernel and rechecked on the small CPU fixture.

Both terminal records are complete. After the first-round B stop, no further endpoint
development-validation or reference evaluation was performed; A's 18 development scores had been
completed earlier. Because both first-round selections were no-op, no four new Control/Swap endpoints
exist. The registered 046 baseline is unchanged.

## Limits and evidence map

- The study contains one cell. Two optimization initializations are not biological replicates.
- All source `readID` values are `.`, so molecule-level isolation is unavailable.
- Round 057 supports rejection of eight fixed copy-link disruptions on a previously viewed
  development fold and rejection of the scanned first-round swap candidates under the frozen
  training gates. It does not establish chromosome-long copy recovery.
- Raw or derived contacts, numeric arrays, coordinates, checkpoints, gradients, fixture binaries,
  and full logs are intentionally not distributed.

Published evidence:

- Internal run record: [`README.md`](../test_res/057-20260917T013859Z-copy-link-dev-validation/README.md)
- Frozen configurations: [`config.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/config.json) and [`config_B.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/config_B.json)
- A results and terminal: [`results.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/results.json) and [`terminal.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/terminal.json)
- B decisions and terminal: [`results_B.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/results_B.json) and [`terminal_B.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/terminal_B.json)
- Exact verification and correction record: [`round1_results.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/round1_results.json)
- Fixture and budget records: [`fixture_results.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/fixture_results.json), [`ledger_B.json`](../test_res/057-20260917T013859Z-copy-link-dev-validation/ledger_B.json)

For the preceding optimization comparison and the source of the training splice medians, see
[`PRO_REVIEW_RESULTS_056.md`](PRO_REVIEW_RESULTS_056.md). The registered baseline remains
`P9016-046-G-random-base-1Mb`; see [`CURRENT_BASELINE.md`](CURRENT_BASELINE.md).
