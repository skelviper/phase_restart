# Round 057 Publication Receipt

- Publication content commit: `e764b87756cdca7c2a6bf128ca4fc5913e8e9957`
- Remote: `git@github.com:skelviper/phase_restart.git`, branch `main`
- Remote verification: `refs/heads/main` matched the content commit at `2026-09-17T02:51:13Z`
- Previous published commit: `644bc1444df9e1ce314345cfb5e15ad6d047bacc`
- Scope: 25 byte-identical round-057 research text/source files, the external 057 report, and review-entry updates; no raw or derived contacts, numeric arrays, coordinates, checkpoints, gradients, fixture binaries, full logs, or credentials
- Research execution: none; publication used existing evidence only
- Source verification: 25/25 selected files byte-identical; both scan TSV files contain one header plus 1,992 candidates and hash-match their recorded SHA256; six new Python modules parse and import with bytecode writing disabled
- Archive path: `scratch/github-review-20260916_203503-no-git.zip`
- Pre-receipt archive check of the content commit: 1,050 ZIP entries, 6,547,061 bytes, SHA256 `aacb2f2c5c71bfd484e18e8c71c00c1ccdde2433cba29a838779f1fa64a9caf5`, zero `.git` entries
- Final archive rule: regenerate the same path with `git archive HEAD` after this receipt is committed and pushed; the final archive SHA256 and size are verified in the publication response rather than self-embedded in its own commit
- Formatting note: the three source TSV files retain their original CRLF bytes, which `git diff --check` reports as trailing whitespace; changing them would break the required source hashes
- Prompt correction at `2026-09-17T02:53:31Z`: the copy-paste GPT Pro prompt now identifies round 057 as latest, directs the reviewer to `PRO_REVIEW_RESULTS_057.md`, and distinguishes round 056's preceding exposure comparison from round 057 A PASS / B first-scan no-op / paired-continuation-not-triggered status

The final release commit is the commit containing this receipt. Its exact SHA is resolved with
`git log -1 --format=%H -- test_res/057-20260917T013859Z-copy-link-dev-validation/PUBLICATION_RECEIPT.md`
and is verified against the remote `main` ref after the final push.
