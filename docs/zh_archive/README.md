# 中文阅读副本索引

本目录保存因原文有执行端哈希锁、明确冻结快照，或需要审查 provenance 但不能原地改写的项目文档中文阅读副本。机器执行、哈希校验和引用仍使用原路径；下表只提供阅读映射。`.tmp*`、任务备份、`test_res` 活动报告、`deliverables/` 和 `scratch/` 不属于本索引。

## 活动哈希锁定原文

| 原始路径（保持不变） | 中文阅读副本 | 原文 SHA256 |
|---|---|---|
| `docs/RECONSTRUCTION_V1_PROTOCOL.md` | [中文副本](RECONSTRUCTION_V1_PROTOCOL.zh-CN.md) | `cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5` |
| `docs/POST020_ALLELE_ABLATION_PROTOCOL.md` | [中文副本](POST020_ALLELE_ABLATION_PROTOCOL.zh-CN.md) | `87da289913fe27ab935c816a30e4f3847bd525ce863ac99f926af9c5e3900957` |
| `docs/PLAN-post020-allele-signal.md` | [中文副本](PLAN-post020-allele-signal.zh-CN.md) | `6bf156cf3591f3fa7e2336ec9413e866d3084d2b90829a59a0c5ec4ee5fd5d63` |

## 分类边界

- **实际执行端哈希锁：**目前只有上面的 3 个活动 `docs/` 原文由运行代码/selection gate 按冻结清单核验；本索引沿用此前采集的锁定 SHA256，不把中文副本或当前工作区重哈希当作原件未改证明。
- **明确的冻结快照路径：**`frozen_protocol/`、`source_snapshot/` 和 `archived020/` 表示项目保存的原始快照。它们的字节和 SHA 分组列出，但除活动 3 个原文外，不在这里声称存在独立的运行时哈希锁。
- **仅 provenance/release 文件名命中：**020 release、022 `AGENTS.md`、024 `revision_v2_impact.md`、028 `FUSED_KERNEL_DESIGN.md`/`RESUME_GPU.md` 纳入完整审查，是溯源或发布记录，不因目录名而升级为执行端锁定文件。036 `release_ready_report.md` 另有运行端/冻结发布合同中的显式 SHA 约束，按独立契约记录，不归入上面的 3 个活动 `docs/` 原文锁。

## 冻结/溯源 Markdown 的实际清单

以下 47 个 `test_res` 冻结、protocol、provenance、source_snapshot 或 release Markdown 已按**此前采集的预编辑 SHA256 证据**分组；同一组只保留一个中文阅读副本。路径均为工作区内真实路径，不使用 glob 占位符。若源文件后来发生交集漂移，会在计数和备注中明确排除，绝不把新 SHA 当作未修改证明。

### 已有中文原文或中文原件

| SHA256 | 实际原始路径 | 阅读映射 |
|---|---|---|
| `dac2b9f25eac53ff00257d2f4049d89d60f7851354c391f8ff008c1894730a2c` | `test_res/018-20260913_121446-v1-synthetic-calibration/preregistration.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/preregistration.md` | [原文（中文）](../../test_res/018-20260913_121446-v1-synthetic-calibration/preregistration.md)；[同内容来源](../../test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/preregistration.md) |
| `687666a3ca61ff09f164bb1a672c4d3125d2561c18e8135f570de06b3e041573` | `test_res/018-20260913_121446-v1-synthetic-calibration/logs/status.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/logs/status.md` | [原文（中文状态记录）](../../test_res/018-20260913_121446-v1-synthetic-calibration/logs/status.md)；[同内容来源](../../test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/logs/status.md) |
| `47fac075ffae5ca3fef60a8e9ca386fe59392e7dfabeafa6a5996675b3969636` | `test_res/018-20260913_121446-v1-synthetic-calibration/frozen_protocol/RECONSTRUCTION_V1_CALIBRATION.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/frozen_protocol/RECONSTRUCTION_V1_CALIBRATION.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/frozen_protocol/RECONSTRUCTION_V1_CALIBRATION.md` | [中文阅读副本](RECONSTRUCTION_V1_CALIBRATION.zh-CN.md) |
| `3407be2cfa407c13360dea0d3278ff0c1ee8e8ce452eb3f31a17c3721013d1ff` | `test_res/019-20260913_064842-v1-calibration-recovery/frozen_protocol/RECOVERY_PROTOCOL.md` | [原文（中文）](../../test_res/019-20260913_064842-v1-calibration-recovery/frozen_protocol/RECOVERY_PROTOCOL.md) |
| `c012d4c36f6c3f97e4556a889acbd15dabd3250cd9dde8f945d5664bea43dce6` | `test_res/028-20260913_151456-020-gpu-independent/provenance/RESUME_GPU.md` | [原文（中文）](../../test_res/028-20260913_151456-020-gpu-independent/provenance/RESUME_GPU.md) |
| `260bf5849a28cf83f9c71b8cd14b9d7539f1be441e502818d4f59194a3967860`（当前报告 SHA；此前清单 SHA 为 `43bdece8410f3fb4ba91814cab19b726a210a98752fa504add218faf32b8ec88`） | `test_res/036-20260914T064651Z-gpu-multires/release_ready_report.md` | [当前原文（中文，非中文副本）](../../test_res/036-20260914T064651Z-gpu-multires/release_ready_report.md)；与 `build_multires_release.py:306`、`release_contract.json:596-597`、`terminal_evidence.json:61-62` 的既有 SHA 约束一致；保留历史差异，不声称逐字节未改 |

### 已中文化的英文来源（含冻结快照与 provenance 候选）

| SHA256 | 实际原始路径 | 中文阅读副本 |
|---|---|---|
| `cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5` | `test_res/018-20260913_121446-v1-synthetic-calibration/frozen_protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/frozen_protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/frozen_protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/020-20260913_071841-v1-p9016-joint/provenance/protocol/docs/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/022-20260913_111031-v1-continuation-fdg-r2/provenance/protocol/docs/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/source/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/fused-cuda/attempt-20260913T165036Z/provenance/source_snapshot/source/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163638Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163908Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T164158Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T170910Z-retry/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_PROTOCOL.md`<br>`test_res/033-20260914_044246-c0-controlled-reproduction/provenance/protocol/docs/RECONSTRUCTION_V1_PROTOCOL.md` | [中文副本](RECONSTRUCTION_V1_PROTOCOL.zh-CN.md) |
| `794ffe6d00c09eeeb577abf8ae3fddda4cc212e3bdbfe3be5403ae9605b63f2b` | `test_res/018-20260913_121446-v1-synthetic-calibration/frozen_protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/frozen_protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/019-20260913_064842-v1-calibration-recovery/source_snapshot/018/frozen_protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/020-20260913_071841-v1-p9016-joint/provenance/protocol/docs/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/source/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/fused-cuda/attempt-20260913T165036Z/provenance/source_snapshot/source/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163638Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163908Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T164158Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T170910Z-retry/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_INITIALIZATION.md`<br>`test_res/033-20260914_044246-c0-controlled-reproduction/provenance/protocol/docs/RECONSTRUCTION_V1_INITIALIZATION.md` | [中文副本](RECONSTRUCTION_V1_INITIALIZATION.zh-CN.md) |
| `3a7904106de44e96e93811c2bcae512db47316330f05dccf54b15d9d8e000e9d` | `test_res/020-20260913_071841-v1-p9016-joint/provenance/protocol/docs/RECONSTRUCTION_V1_RUN.md`<br>`test_res/022-20260913_111031-v1-continuation-fdg-r2/provenance/protocol/docs/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/source/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/fused-cuda/attempt-20260913T165036Z/provenance/source_snapshot/source/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163638Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T163908Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T164158Z/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T170910Z-retry/source_snapshot/archived020/protocol/RECONSTRUCTION_V1_RUN.md`<br>`test_res/033-20260914_044246-c0-controlled-reproduction/provenance/protocol/docs/RECONSTRUCTION_V1_RUN.md` | [活动中文版本](../RECONSTRUCTION_V1_RUN.md) |
| `a27387325dba5761d3208bfc3f97d539e4e9120efb65a8a8f9115eb4ee230a93` | `test_res/022-20260913_111031-v1-continuation-fdg-r2/provenance/protocol/AGENTS.md` | [中文副本](AGENTS-022-v1-continuation-fdg-r2.zh-CN.md) |
| `a5fdc4045e3a32b83e281cc5b693aec2bf4968ced74445efd730d13ed5004370` | `test_res/024-20260913_133109-r2-allele-signal-derived/provenance/revision_v2_impact.md` | [中文副本](revision_v2_impact.zh-CN.md) |
| `b555222e3242cd522caf1c072884cda6fc9c316fc562899305474465f75b0d71` | `test_res/020-20260913_071841-v1-p9016-joint/provenance/release-020-v1-p9016-joint.md` | [中文副本](release-020-v1-p9016-joint.zh-CN.md) |
| `ca21d7cf37547db694288e85dc4b0971673c5dcbe207586dd3c93ba56f7d68d6` | `test_res/028-20260913_151456-020-gpu-independent/provenance/FUSED_KERNEL_DESIGN.md`<br>`test_res/028-20260913_151456-020-gpu-independent/backend_runs/fused-cuda/attempt-20260913T165036Z/provenance/source_snapshot/provenance/FUSED_KERNEL_DESIGN.md` | [中文副本](FUSED_KERNEL_DESIGN.zh-CN.md) |

## POST020 历史计划

| SHA256 | 实际原始路径 | 阅读映射 |
|---|---|---|
| `3d85720af477b4a87cffdd4e19780a2e9b3e0937f04579bfca2bea2516cb2215` | `docs/archive/PLAN-post020-allele-signal.prefit-3d85720a.md` | [原文（中文）](../archive/PLAN-post020-allele-signal.prefit-3d85720a.md)；当前 Revision 2 [中文副本](PLAN-post020-allele-signal.zh-CN.md) |

## 覆盖与排除计数

- 清单分类：47 个 `test_res` Markdown 分为 13 个 SHA256 组；其中 29 个路径位于明确的 `frozen_protocol/` 或 `archived020/` 快照目录，4 个是 preregistration/status 辅助记录，14 个只是 provenance/release 文件名命中的候选；这些分类都不替代实际执行端哈希锁。
- 其中 37 个英文来源由 7 个去重中文副本覆盖，10 个 test_res 原文已是中文并直接链接。另有 1 个 `docs/archive` 中文历史版本，单独列出；因此本索引覆盖 48 个冻结/历史原始 Markdown 路径。
- 实际执行端哈希锁仍只有上面列出的 3 个活动 `docs/` 原文；这里登记的 test_res/024 revision、028 resume/FUSED SHA 是此前采集的源文件身份分组，不是运行时锁定证明。036 release report 另由运行端校验和冻结发布合同共同约束，当前报告 SHA 与三处既有契约记录一致。
- 036 release report 的历史清单与正式契约存在差异：此前清单 SHA 为 `43bdece8410f3fb4ba91814cab19b726a210a98752fa504add218faf32b8ec88`，当前重新计算并由运行端/冻结发布合同共同引用的 SHA 为 `260bf5849a28cf83f9c71b8cd14b9d7539f1be441e502818d4f59194a3967860`；因此保留 43bde... 历史差异记录，不声称报告逐字节从未改过。
- `test_res` 全树实际 Markdown 共 119 个；其余 72 个是活动 README/报告/评估输出或其他非冻结文档，按任务边界排除，不修改也不纳入本索引。
- `.tmp_batch2_baseline_recovered/` 等临时 baseline、`deliverables/`、`scratch/`、`docs/PROJECT_CONTEXT.md`、锁定原文自身及第三方/原始资产均排除。临时 baseline 不计入上述文件数。

活动锁定原文的字节不在本目录工作范围内。冻结快照、`source_snapshot` 和 `archived020` 只保留必要的中文阅读映射；相同内容按 SHA256 去重。机器命令仍指向原始路径。
