# `native/hickit/` 的溯源

该副本于 2025-09-12 从 `/work/phase3/hickit/` 引入（hickit r291，分支 `skelviper/hickit`）；后者是生成 `data/P9016.1m.3dg.gz` 所使用的 FDG 实现。文件是逐字节相同的副本；不要编辑它们，也不要对其进行封装。

构建（CPU、无 CUDA）：

```bash
cd native/hickit && make hickit
```

核验：随仓库提供的二进制逐字复现 phase3 构建的 chr1 oracle 拟合（Spearman `chr1a` 对 `chr1(pat)` = 0.5894，`chr1b` 对 `chr1(mat)` = 0.5719）。

## SHA-256

```
702e5ea8bc04f904282b7bb76944f3d3b7d089b1672c08f9b02ead878bf128f8  bin.c
eea1507387eeddec6fda2d2550d372b312cb1e9e5e4735c6eb7863a0bb6342f7  count.c
1033e3195bcafb59e1b389ba9b0d9399d912000d49899b8da9c303d47a86c0ad  fdg.c
a45d1b09127b0e54af0eb63283f9180bceb3f146b813f19027243035e1315fe2  fdg_gpu.h
9055f0a9c26699736bcb7e3b900cb4761a2be38b88a38153a62a598fddad4eff  fdg_gpu_stub.c
4b5c7c7f5ca7e29c6281f84171e01bc84ac51a4790bf9f38d0eca104a3334463  hickit.h
28892649228d6bbba0066e09fe2fce282f6b8388d369efde0b7caafc599ae4fd  hkpriv.h
42a298d0f152696ca63b88346e1be711aa91b9720e288f874821db8960688a05  image.c
f767101f04cb7691eee8cdd809224e4256e010caf6ec4246c324e0d245034638  io.c
9ab6853ff9765b4f274a326bb599d6d59e0b2b963b42115bbcda95e7ee985bd5  kavl.h
76197162b53e0e66f69880c81c4fef113c31d61747598ecbbf2e129e3170a3c6  khash.h
8e91f97d9568cf4a3f2007c9267301d8a8a23b7e5c1bae812a4c2c0d8fea0cdb  klist.h
af5549fbfd5dde191ca6fb24894a22124385d89bc01032b9b1974a5311f56f57  krng.h
6ec1fdd30ebfb66c29ca20b73175023d44196b6e7165c014ba19cd2919ac4ec1  kseq.h
0ab17bab513077750ae5e476b409a137cdc0fe653c587fd072b8bf1e237a8223  ksort.h
3d8ffabf72e250ac5f6a4ee7d102606c286e458b294cdbfdf8b84dd9306ea948  main.c
7f2c734556147e657ca99b0ce0e5b50a5e3441641d2c6ed4a9983c47579c77c7  Makefile
996f314c3e7a9afaa122f0a2644d12f2f4f9385ddd6db6b94faa4581930b72cd  pair.c
d1546a71825968344f8b73086a2314477dd26afbb65860012325b92d17c17af9  phase.c
881b844acd503351f083df16ca39b4a2b0009581b2fce3298c9f355863c2e4cc  sdict.c
d81e992c90f4d7824e125e1617048c85e17447314d9ac578401e2c17ed344f31  stb_image_write.h
63b0dca77f5b2ef335f2e22f76d6b2280a78f0f83531e29753549d6f52a31e45  view3d.c
```

非随仓库提供的文件：`blind.c`、`audit_*.c`、`run_blind_*.c`、`test_*.c`、`fdg_gpu.cu`、GL 源码。
它们属于 phase3/phase4 blind framework，本项目有意不复用。
