"""独立核验评审提出的具体断言。"""
import gzip
import numpy as np

path = "data/P9016.pairs.gz"
n_intra = 0; n_str_dropped = 0
ch1_total = 0; ch1_dropped = 0; ch1_bothphased = 0; ch1_bp_dropped = 0
n_samebin = 0; n_total7 = 0
ex = []
with gzip.open(path, "rt") as f:
    for line in f:
        if line[0] == "#": continue
        n_total7 += 1
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3]: continue
        n_intra += 1
        s_gt = c[2] > c[4]                 # 字符串比较（有问题的过滤条件）
        v_gt = int(c[2]) > int(c[4])       # 数值比较（正确）
        if s_gt and not v_gt:
            n_str_dropped += 1
            if len(ex) < 3: ex.append((c[1], c[2], c[4]))
        if c[1] == "chr1":
            ch1_total += 1
            if s_gt and not v_gt: ch1_dropped += 1
            if c[7] in "01" and c[8] in "01":
                ch1_bothphased += 1
                if s_gt and not v_gt: ch1_bp_dropped += 1
        if c[1] == c[3] and int(c[2]) // 1_000_000 == int(c[4]) // 1_000_000:
            n_samebin += 1
print("=== claim 1: string vs numeric coordinate comparison ===")
print("intra contacts                       : %d" % n_intra)
print("dropped by string comparison         : %d  (%.2f%%)" % (n_str_dropped, 100*n_str_dropped/n_intra))
print("chr1 intra                           : %d ; dropped %d (%.2f%%)" % (ch1_total, ch1_dropped, 100*ch1_dropped/ch1_total))
print("chr1 both-end-phased                 : %d ; dropped %d (%.2f%%)" % (
    ch1_bothphased, ch1_bp_dropped, 100*ch1_bp_dropped/ch1_bothphased))
print("examples (chrom, pos1, pos2)         :", ex)
print()
print("=== claim 2: the hash is genomic-distance parity ===")
b1, b2 = np.meshgrid(np.arange(200), np.arange(200), indexing="ij")
h = (b1*7919 + b2*104729) % 2
print("(i*7919 + j*104729) %% 2 == (i+j) %% 2 for all i,j<200 :", bool(np.all(h == (b1+b2) % 2)))
print("same-bin pairs (i==j) -> hash value  :", int((0*7919+0*104729) % 2), "(0 = the script's test fold)")
print()
print("=== claim: same-bin 1 Mb contacts exist ===")
print("intra contacts whose two ends fall in the same 1 Mb bin: %d (%.2f%% of intra)" % (
    n_samebin, 100*n_samebin/n_intra))
