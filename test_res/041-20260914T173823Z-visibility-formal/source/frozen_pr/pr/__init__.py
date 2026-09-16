"""phase_restart: SNP-free diploid reconstruction of P9016.

Package layout
    paths.py    constants (bin size, bin offset, chromosome order)
    gate.py     coordinate-hash gate that guards every label read
    pairs7.py   SNP-free reader/writer (exactly seven columns, hard phase rejection)
    labels.py   the ONLY module that opens the phase columns; gate-guarded
    folds.py    mixed-hash train/val/test split of numeric bin pairs
    fdg.py      wrapper around the vendored native hickit FDG engine
    splits.py   candidate contact splits (oracle, fragment swap, random, consensus)
    score.py    held-out Spearman, stratified readouts, paired bootstrap CI
    figs.py     figure helpers (3-inch panels, 300 DPI, 7 pt)
"""
