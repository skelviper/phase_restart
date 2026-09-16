# Visibility R2 独立准备

本目录是 visibility-only 下一轮的独立评价适配器与协议准备，不是正式评价目录。准备阶段不读取新 candidate、reference 或 synthetic truth 坐标，不启动 fit/native，也不选择终点。

## 冻结范围

- real：`V0/VZ/V1 × consensus_joint_base1103/random_joint_base2207`，共 6 个 endpoint。
- synthetic：`P2/N2 × V0-production/V1-profile/V0-known-generating-e`，共 6 个 endpoint；不增加 synthetic VZ 重复。
- real 使用全部 `1,703,888` 条 contact、20 条染色体、40 条轨迹和原来的 `5 Mb -> 2 Mb -> 1 Mb` 流程；本轮 real 合法预算只有 B2 `612/404/486`，B1 `306/202/243` 仅作历史记录，B2 从原始 014 start 独立开始。总 `10470` 是 12 fits 的 outer-FG maximum，允许合法早停，必须报告实际值。
- synthetic 两个 fixture 共享同一个 `x0` 和 `p0=0.75`，1 Mb 上限 `243` FG。
- V0 为原 C0 的 fixed `e=sqrt(endpoint_count+10)`（full-grid mean 归一化）。VZ 只把观测 offdiag degree 为 0 的 bin 设为 `e=0`，其余 exposure 与 V0 固定一致；它是 zero-coverage boundary control，不是真实不可见状态。训练侧已核对 P2/N2 的 zero-degree 都为 0、active 都为 `2645/2645`，因此不做 synthetic VZ fit。
- V1 为每个 genomic bin 一个两拷贝共享的 `e=exp(eta)`；full `e` vector 长度为 `2645`，但不是 `2645` 个独立 eta。real 的 eta active/自由度为 `2581/2580`（zero `64`），synthetic 为 `2645/2644`（zero `0`），使用 active-bin mean-zero gauge。同-bin 是 saturated nuisance，不进入 eta profiling；K、p、prior、sphere 和原 count objective 保持不变。V1 仅要求 `degree` 与 `predicted_degree` 两个 full-grid array（可与同一个 `*.visibility.npz` 共用路径），adapter 计算 `degree_residual=degree-predicted_degree`，并验证 `max(abs(r)/max(1,d)) <= 1e-10`。V0/VZ/known-e 不做 eta profile，degree prediction 可标 `NA` 或省略，不伪造 finite-zero residual。
- 失败尝试不进入本轮矩阵：040 只完成 `V0-consensus` 首个 5 Mb 的 `612 FG`，随后 controller 字段读取失败，且 warm positions/q carry 接口审查未通过；该约 50 秒开销不计入本轮 12 endpoint 的 `10470` maximum。待训练侧修复后必须在新目录完成全 12 fits 并锁定 release。

## R2 规则

R2 使用 Spearman structural comparison，不是 regression `R^2`。复用 frozen old21 mask，固定为 `157,529` common pairs / `176,201` total non-diagonal pairs；新 endpoint 不进入 mask。每条染色体只允许把 candidate A/B 整体交换一次，reference 的 `mat/pat` 列固定，然后报告 `matched`、`cross`、`contrast=matched-cross` 和 `minmargin`。本轮不计算 R1、R3、p-value 或 CI。

主比较为 `VZ-V0`（zero-coverage boundary）、`V1-VZ`（active profile）和 `V1-V0`（overall）；source 细节只保留在表中。新 V0 与 VZ/V1 使用同一 Torch backend，本轮同 backend 的 `V0/VZ/V1` 才是主因果比较。038 的 fused M0/B2 由于长非凸路径可能与本轮 V0 不同，只能引用已有同 source R2 表作历史参照；不把它用于新结果的 selection 或科学指标。即使新模型胜过本轮 V0，也不能自动表述为胜过 038 C0。release gate 另注册一个固定 old038 M0/B2 random 的行为回归：在 12 endpoint、terminal、source/selection 和 21 个 frozen-mask bits 全部 hash 后，用冻结的 `r2comparison.pyc` 复算 20 条染色体并逐字段与已发布表按 `1e-12` 对比；这不是新指标或拟合，失败就暂停新指标解释并诊断。主图限制为一张 2x2 PNG，selected V0/VZ/V1 三个模型，基础 panel 3 inch、300 DPI、7 pt；不生成 PDF、HTML 或重复单 panel 图。

## 群体预测 KL

该指标只用于 synthetic，且只在全部 endpoint、终态、sidecar、source、selection 和 frozen mask hash 通过后计算。它不是当前采样 counts 上的训练 NLL，也不替代各模型内部按自身 count NLL 的 source 选择。`frozen_pr/v1_calibration.py` 已提供精确 `generation_rates`、`v1` kernel 和 cis/inter group normalization；P2/N2 metadata 固定 `p_gen=0.8`、`kernel=v1`、`r0=2*l0`、`epsilon=1e-6`，所以该定义无需近似、重拟合或重新采样。η/profile 带来的改善与 VZ zero-coverage boundary 影响分别报告，不从 NLL 自然下降推断模型胜出。

对 `g in {cis_offdiag, inter}`：

- `pi_true`：真实坐标、生成 exposure 和 `p_gen=0.8` 代入精确 generation rate 后，在同一 group 的全部 eligible pairs 上归一化。
- `pi_fit`：锁定 candidate 坐标、该 endpoint 的 `p` 和最终 `e` sidecar 代入同一 kernel 后，在同一 eligible pair 集合上归一化。
- `KL_g = sum(pi_true * log(pi_true/pi_fit))`，包含观测 count 为 0 的 eligible pair；same-bin/diag saturated nuisance 完全排除。
- 加权结果为 `(696680*KL_cis + 568434*KL_inter)/1265114`，单位为 population-predictive nats per offdiag record；可选 `population_predictive_excess_NLL` 为该值乘 `1,265,114`，不是当前采样 counts 的训练 NLL。
- 若 `pi_true>0` 而 `pi_fit=0`，直接报告 `+inf`，不 clipping。VZ 的 `e=0` 由 adapter 的 nonnegative-rate path 处理，不修改 frozen generator。
- KL 不参与 endpoint/source/model selection，不在 evaluation counts 上 refit `e`，不采样新数据。

本轮 metadata-only feasibility 结果为 `FEASIBLE_EXACT_GENERATION_RATES`。eligible pair 分母为 `cis_offdiag=184,016`、`inter=3,312,674`；weighted denominator 为 `1,265,114`。

## 发布边界

`release_contract.template.json` 只包含 schema 和必填字段，当前 `status=pending_parent_release`、`release_authorized=false`、endpoint_records 为空。父侧必须提供 12 个 terminal endpoint、candidate 3DG、`p/q`、final `e`、active mask 及各自 SHA256；V1 另需同一 full-grid order 的 `degree`/`predicted_degree` sidecar（可与 `*.visibility.npz` 共用路径）。V0/VZ/known-e 的 degree prediction 可为 `NA`/省略。还需提供 source/selection/terminal/preflight evidence 后，才能将 contract 置为 released。

`visibility_r2_entry.py` 的访问顺序为：

1. 仅元数据准备/预检；
2. 父侧发布合同；
3. 12 个 candidate、`*.visibility.npz`/p/q sidecar 和 terminal record 的 hash（V1 的 `degree`/`predicted_degree` 可同 NPZ）；
4. parent source/selection evidence 与 21 个 frozen mask input 的 hash；
5. old038 M0/B2 random 历史行为回归 gate（只在上述 hashes 完成后执行）；
6. reference/truth hash 和 payload；
7. R2、N2 diagnostic 和 synthetic population KL。

041 短集成检查已按实际 controller 输出验证：每个 case 的 `1Mb.visibility.npz` 使用 `numeric_position`（单数）和 `chromosome_index`，V1 在同一 NPZ 提供 `degree`/`predicted_degree`，V0/VZ/known-e 不要求 degree residual；p/q 与 `p_from_q(q)`、terminal metadata 一致。证据写入 `integration_schema_validation.json`。该目录的 3DG 从未打开，且短 check 不构成正式 release。

因此当前准备不会产生正式 R2 数值，也不会打开任何新 candidate/reference/truth 坐标。

## 验证命令与证据

```bash
python docs/audits/visibility-r2-preparation-20260914T164603Z/visibility_r2_entry.py --validate-preparation
python docs/audits/visibility-r2-preparation-20260914T164603Z/visibility_r2_entry.py --population-kl-feasibility
```

两条命令均不触碰坐标 payload。`preparation_validation.json` 和 `preparation_terminal_evidence.json` 记录 `candidate_coordinate_payloads_opened=0`、`reference_opened=false`、`synthetic_truth_coordinate_payloads_opened=false`、`fit_called=false`、`native_called=false`。

`source_snapshot.json` 保存 adapter 将使用的实际依赖 hash，`frozen_pr/` 保存对应的实体源码和旧 `r2comparison.pyc`；正式 synthetic evaluator 通过 `visibility_frozen_pr` 加载，不回退到可变 live `pr`。old 038 的 `r2comparison` source hash 作为 provenance 保留，运行时使用其 pre-edit bytecode snapshot；real mask adapter 原字节仍从旧 prep 路径加载，legacy alias 由 adapter 指向 frozen snapshot。loader 将 `__file__` 虚拟映射回仓库 `pr/<module>.py`，因此 `ROOT` 和 `SNPFREE` 默认路径保持正确。当前 `genome.py`、`score.py`、`contact_model.py`、`reconstruction_init.py` 的实际 hash 被显式记录，不静默复用过期 026 hash。正式 gate 还会逐项重 hash `frozen_pr/` 的所有依赖。
