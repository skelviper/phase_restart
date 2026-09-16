# 发布记录：Reconstruction V1 P9016 联合拟合

- 正式输出：`test_res/020-20260913_071841-v1-p9016-joint/`
- 发布依据：父侧对 019 校准恢复的审查；读取真值前校验、summary/README、数学/实现 gate 和独立端点审计均已通过。
- 先前校准运行：`test_res/019-20260913_064842-v1-calibration-recovery/`
- 019 metrics SHA256：`36665687ac7156b4bc535a3a767e5bcb8bcce5af55953d12e89685ab828afcb4`
- 019 读取真值前校验 SHA256：`bf47cbb977d758d14d5b42244979bfac6fa4359494e313b45de155e041c1ef63`
- 019 独立端点审计 SHA256：`44f926f8a47d0e09eb36cf45910f41dade2c96d897d907ab9e2dd842678de70b`
- 校准解释：中断后的共同迭代 50；登记的 80 次迭代预算未完成，因此不能建立 L2。该发布仅是探索性 V1。
- 冻结运行：候选为 `consensus_joint`（014 盲 consensus，seed 1103），随后为 `random_joint`（014 盲 random，seed 2207）；没有额外候选、调参、预算或重试。
- 冻结顺序：全部 1,703,888 条 SNP-free 记录、20 条染色体、40 条轨迹、5 Mb -> 2 Mb -> 1 Mb，maxiter 300/200/240，maxfun 930/630/750，maxls 20，ftol 1e-10，gtol 1e-6，第一层 p 为 0.75，传递原始 q。
- 训练边界：不访问 phase、reference 或 oracle 载荷；原始结构和 p 仍不作保证；reference 不能改变训练或选择。
- 选择：等待两个候选完成并写出哈希，然后对同一个 1 Mb 完整计数似然重新评分；最小化 `count_nll_normalized`，`1e-9` 内的平局按预注册顺序解决。保留每次尝试和失败。
- 环境：`analysis` conda 环境；`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；两个候选工作进程。
- 精确命令：

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python run.py reconstruct --out test_res/020-20260913_071841-v1-p9016-joint --workers 2 --threads 1
```

## 运行后结果

- 受管作业：`bash-31`，退出码 0。
- 训练状态：`training_complete`；两个候选都完成三个阶段和最终计数重评分。
- 选中候选：`random_joint`。
- 1 Mb `count_nll_normalized`：random `9.593585931292237`；consensus `9.598781292397067`；consensus 减 random 为 `0.0051953611048301`，高于 `1e-9` 平局容差。
- 选中坐标：`selected.3dg`，SHA256 `afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078`。
- 独立终止审计：`termination_audit.json`；六次阶段尝试都达到迭代上限，没有达到函数上限，所有原始 SciPy 结果均为 `status=1`、`success=false`，终止消息为迭代上限。
- `selection.json` SHA256：`5ee6376527d6139eb4fe55ac057b40ca94a6e731289a463989f2343a815252a6`。
- `run_status/attempts.jsonl` SHA256：`8356dde6af3cb839d0bd261f02c4ecee3026b65738eb3d26b2b6d49071809ca8`。

本发布记录仅用于 provenance。真实评估是后续隔离阶段，须在 `selection.json` 和 `selected.3dg` 完成并通过哈希核验后进行。
