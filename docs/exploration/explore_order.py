import gzip
n = 0; sorted_ok = True; prev = None
same_end = 0; same_pair = 0; blocks = 0; runlen = 0; maxrun = 0
rows = []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        key = (c[1], c[3], int(c[2]), int(c[4]))
        if prev is not None:
            if key < prev: sorted_ok = False
            if key[0] == prev[0] and key[1] == prev[1] and (int(c[2]) == prev[2] or int(c[2]) == prev[3] or int(c[4]) == prev[2] or int(c[4]) == prev[3]):
                same_end += 1
            if key == prev: same_pair += 1
        prev = key; n += 1
        if n <= 12: rows.append(c)
print("rows", n, "sorted:", sorted_ok)
print("consecutive rows sharing an endpoint:", same_end, "(%.2f%%)" % (100.0*same_end/n))
print("consecutive identical pairs:", same_pair, "(%.2f%%)" % (100.0*same_pair/n))
print("first rows:")
for r in rows: print("  ", r)
