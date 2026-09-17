# Round 058 Publication Receipt

- Primary publication content commit: `65521fd4a1538cefe76b6a42084bb3b5d1913b15`
- Pre-receipt review correction commit: `1b91dacec33e6e8f294f7c8f04377ee1fc99d27e`
- Remote: `git@github.com:skelviper/phase_restart.git`, branch `main`
- Remote verification: `refs/heads/main` matched the pre-receipt correction commit at `2026-09-17T07:00:34Z`
- Previous published commit: `e440568b556c1b4dfaf5320b56f4bdc54dc848b3`
- Scope: 39 byte-identical round-058 research text/source files, the English external report, and review-entry updates; no raw or derived contacts, numeric arrays, coordinates, checkpoints, fixture binaries, full logs, or credentials
- Research execution: none; publication used existing completed evidence only
- Source verification: 39/39 selected files byte-identical; paired TSVs contain 4 / 2 / 24 complete data rows; nine new Python modules pass `ast.parse`; 26 JSON files parse; the frozen config SHA and 12-endpoint / 36-artifact gate match the stored records
- Archive path: `scratch/github-review-20260916_203503-no-git.zip`
- Pre-receipt archive check of commit `1b91dacec33e6e8f294f7c8f04377ee1fc99d27e`: 1,099 ZIP entries, 6,717,815 bytes, SHA256 `890eed46baa7f01549f1ba932f7dd039716660a9694c590000a27d3862ce7e63`, zero `.git` entries
- Final archive rule: regenerate the same path with `git archive HEAD` after this receipt is committed and pushed; the final archive SHA256 and size are verified in the publication response rather than self-embedded in its own commit

The final release commit is the commit containing this receipt. Its exact SHA is resolved with
`git log -1 --format=%H -- test_res/058-20260917T053435Z-a-swap-shortopt-b-p-profile/PUBLICATION_RECEIPT.md`
and is verified against the remote `main` ref after the final push.
