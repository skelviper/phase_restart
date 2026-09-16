# 049 评价报告：max-contact 统一多尺度方法 vs 046 baseline

本报告由 `source/eval049_report.py` 从 `results/` 的机器输出生成；所有分母、支持集与终态都来自冻结产物，不重新拟合、不选择端点。生物重复 n=1，20 条染色体是同一细胞内关联测量。

## 1. 封存、终态与预算

| 项目 | 值 |
| --- | --- |
| run_id | 049-20260915T162917Z-max-contact-unified-multiscale |
| created_utc | 2026-09-15T17:18:06.700129+00:00 |
| reference_first_opened_utc | 2026-09-15T00:00:00+00:00 |
| reference_sha256 | aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa |
| reference 解析模式 | reference |
| reference 缺失 loci（mask 支持外允许） | 0 |
| endpoint manifest sha256 | bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb |
| selection sha256 | cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc |
| 输出 CSV/端点数 | 15 个 dataset（12 fit + 2 initial + baseline） |
| 评价终态 | complete |
| validation 状态 | PASS |
| bootstrap 指数矩阵与 046 相同 | NA |

**全部端点终态与预算**（退出码 0 不等于科学收敛；wall/canonical grad/p/q 取自 manifest 与逐 fit JSON，未重算）：

| dataset | kind | terminal | outer FG | wall s | canonical grad max abs | p | q | q_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | formal_fit | converged | 1502 | 100.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| A-raw-random | formal_fit | budget_not_converged | 1502 | 101.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| A-ms-consensus | formal_fit | converged | 1502 | 102.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| A-ms-random | formal_fit | budget_not_converged | 1502 | 103.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| B-raw-consensus | formal_fit | converged | 1502 | 104.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| B-raw-random | formal_fit | budget_not_converged | 1502 | 105.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| B-ms-consensus | formal_fit | converged | 1502 | 106.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| B-ms-random | formal_fit | budget_not_converged | 1502 | 107.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| C-raw-consensus | formal_fit | converged | 1502 | 108.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| C-raw-random | formal_fit | budget_not_converged | 1502 | 109.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| C-ms-consensus | formal_fit | converged | 1502 | 110.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| C-ms-random | formal_fit | budget_not_converged | 1502 | 111.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| initial-consensus | initial_control | converged | 1502 | 112.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| initial-random | initial_control | budget_not_converged | 1502 | 113.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |
| baseline-046-real-extension-G-full-J | baseline | converged | 1502 | 114.0000 | 1.000e-04 | 0.9300 | 2.6600 | q_from_p(p_init=0.75) |

**共同原 G 重打分（pre-reference 封存字段，value-only，不重算）**：`common_G_count = count_nll_by_loss['A']`、`common_G_fullJ = fullJ_A`；`own_*` 为该 fit 自己 loss 的数值，两者不可混用。

| dataset | kind | terminal | loss | own count nll | own fullJ | common-G count | common-G fullJ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | formal_fit | converged | A | 9.5000 | 9.6000 | 9.4000 | 9.5500 |
| A-raw-random | formal_fit | budget_not_converged | A | 9.5010 | 9.6010 | 9.4010 | 9.5510 |
| A-ms-consensus | formal_fit | converged | A | 9.5020 | 9.6020 | 9.4020 | 9.5520 |
| A-ms-random | formal_fit | budget_not_converged | A | 9.5030 | 9.6030 | 9.4030 | 9.5530 |
| B-raw-consensus | formal_fit | converged | A | 9.5040 | 9.6040 | 9.4040 | 9.5540 |
| B-raw-random | formal_fit | budget_not_converged | A | 9.5050 | 9.6050 | 9.4050 | 9.5550 |
| B-ms-consensus | formal_fit | converged | A | 9.5060 | 9.6060 | 9.4060 | 9.5560 |
| B-ms-random | formal_fit | budget_not_converged | A | 9.5070 | 9.6070 | 9.4070 | 9.5570 |
| C-raw-consensus | formal_fit | converged | A | 9.5080 | 9.6080 | 9.4080 | 9.5580 |
| C-raw-random | formal_fit | budget_not_converged | A | 9.5090 | 9.6090 | 9.4090 | 9.5590 |
| C-ms-consensus | formal_fit | converged | A | 9.5100 | 9.6100 | 9.4100 | 9.5600 |
| C-ms-random | formal_fit | budget_not_converged | A | 9.5110 | 9.6110 | 9.4110 | 9.5610 |
| initial-consensus | initial_control | converged | NA | 9.5120 | 9.6120 | 9.4120 | 9.5620 |
| initial-random | initial_control | budget_not_converged | NA | 9.5130 | 9.6130 | 9.4130 | 9.5630 |
| baseline-046-real-extension-G-full-J | baseline | converged | NA | 9.5140 | 9.6140 | 9.4140 | 9.5640 |

- 冻结预算：每 fit `1502` FG（全 1Mb full grid），实际 `outerFG` 与冻结值不一致的 fit：无。
- 046 baseline 累计 `1988` FG 含 5Mb/2Mb 低分辨率 stage；本轮为 1502 全 1Mb full-grid FG，**成本不同**，不得宣称与旧 baseline 成本相等。

## 2. 端点选择（打开 reference 之前冻结）

| loss × solver | 选中 source | rule | margin | 复算一致 |
| --- | --- | --- | --- | --- |
| A-raw | consensus | min own count | 1.000e-10 | yes |

| loss | display endpoint | 复算 argmin | 并列集合 |
| --- | --- | --- | --- |
| A | A-raw-consensus | A-raw-consensus | A-raw-consensus |

- 选择只消费 endpoint 自身 count 与 terminal；三个 loss 的数值不可跨 loss 直接排名。图 2 的 5 panel 由该冻结 display 选择决定，评价不改动选择。

## 3. R2 结构一致性（每 chr 四 Pearson/Spearman，fixed-20 macro）

| dataset | P matched | P cross | P contrast | P min margin | P defined | S matched | S cross | S contrast | S min margin | S defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | 0.2000 | 0.0000 | 0.2000 | 0.0100 | 20 | 0.1900 | -0.0100 | 0.2000 | 0.0100 | 20 |
| A-raw-random | 0.2100 | 0.0100 | 0.2000 | 0.0100 | 20 | 0.2000 | 8.674e-18 | 0.2000 | 0.0100 | 20 |
| A-ms-consensus | 0.2200 | 0.0200 | 0.2000 | 0.0100 | 20 | 0.2100 | 0.0100 | 0.2000 | 0.0100 | 20 |
| A-ms-random | 0.2300 | 0.0300 | 0.2000 | 0.0100 | 20 | 0.2200 | 0.0200 | 0.2000 | 0.0100 | 20 |
| B-raw-consensus | 0.2400 | 0.0400 | 0.2000 | 0.0100 | 20 | 0.2300 | 0.0300 | 0.2000 | 0.0100 | 20 |
| B-raw-random | 0.2500 | 0.0500 | 0.2000 | 0.0100 | 20 | 0.2400 | 0.0400 | 0.2000 | 0.0100 | 20 |
| B-ms-consensus | 0.2600 | 0.0600 | 0.2000 | 0.0100 | 20 | 0.2500 | 0.0500 | 0.2000 | 0.0100 | 20 |
| B-ms-random | 0.2700 | 0.0700 | 0.2000 | 0.0100 | 20 | 0.2600 | 0.0600 | 0.2000 | 0.0100 | 20 |
| C-raw-consensus | 0.2800 | 0.0800 | 0.2000 | 0.0100 | 20 | 0.2700 | 0.0700 | 0.2000 | 0.0100 | 20 |
| C-raw-random | 0.2900 | 0.0900 | 0.2000 | 0.0100 | 20 | 0.2800 | 0.0800 | 0.2000 | 0.0100 | 20 |
| C-ms-consensus | 0.3000 | 0.1000 | 0.2000 | 0.0100 | 20 | 0.2900 | 0.0900 | 0.2000 | 0.0100 | 20 |
| C-ms-random | 0.3100 | 0.1100 | 0.2000 | 0.0100 | 20 | 0.3000 | 0.1000 | 0.2000 | 0.0100 | 20 |
| initial-consensus | 0.3200 | 0.1200 | 0.2000 | 0.0100 | 20 | 0.3100 | 0.1100 | 0.2000 | 0.0100 | 20 |
| initial-random | 0.3300 | 0.1300 | 0.2000 | NA | 20 | 0.3200 | 0.1200 | 0.2000 | NA | 20 |
| baseline-046-real-extension-G-full-J | 0.3400 | 0.1400 | 0.2000 | 0.0100 | 20 | 0.3300 | 0.1300 | 0.2000 | 0.0100 | 20 |

- 主列口径：20 chr 全部定义才给值（任一 chr 缺失即 NA），不使用 defined-only 替代；`matched - cross` 是规范不变读出，`u=0` 对照见第 7 节。

## 4. 染色体间排序四距（pooled sorted four-copy distances）

| dataset | pearson | spearman | locus pairs | 四距分母 | 非有限值数 | order0 P | order1 P | order2 P | order3 P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | 0.6000 | 0.6500 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| A-raw-random | 0.6010 | 0.6510 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| A-ms-consensus | 0.6020 | 0.6520 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| A-ms-random | 0.6030 | 0.6530 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| B-raw-consensus | 0.6040 | 0.6540 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| B-raw-random | 0.6050 | 0.6550 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| B-ms-consensus | 0.6060 | 0.6560 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| B-ms-random | 0.6070 | 0.6570 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| C-raw-consensus | 0.6080 | 0.6580 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| C-raw-random | 0.6090 | 0.6590 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| C-ms-consensus | 0.6100 | 0.6600 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| C-ms-random | 0.6110 | 0.6610 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| initial-consensus | 0.6120 | 0.6620 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| initial-random | 0.6130 | 0.6630 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |
| baseline-046-real-extension-G-full-J | 0.6140 | 0.6640 | 2835152 | 11340608 | 0 | 0.5000 | 0.5100 | 0.5200 | 0.5300 |

- 支持集固定为 046 old21 common mask 的 2447 个 valid loci（4894 beads）；缺失记 NA，不缩小分母、不丢 chr。

## 5. 空间读出

| dataset | center P(190) | center S(190) | center stress | single scale | per-chr 19 距离 macro S | top3 邻居均值 | 标签置换 p | copy P(760) | copy S(760) | copy stress | proper det | proper norm RMSD | reflect norm RMSD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | 0.5000 | 0.5000 | 0.9000 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| A-raw-random | 0.5050 | 0.5050 | 0.8950 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| A-ms-consensus | 0.5100 | 0.5100 | 0.8900 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| A-ms-random | 0.5150 | 0.5150 | 0.8850 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| B-raw-consensus | 0.5200 | 0.5200 | 0.8800 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| B-raw-random | 0.5250 | 0.5250 | 0.8750 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| B-ms-consensus | 0.5300 | 0.5300 | 0.8700 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| B-ms-random | 0.5350 | 0.5350 | 0.8650 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| C-raw-consensus | 0.5400 | 0.5400 | 0.8600 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| C-raw-random | 0.5450 | 0.5450 | 0.8550 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| C-ms-consensus | 0.5500 | 0.5500 | 0.8500 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| C-ms-random | 0.5550 | 0.5550 | 0.8450 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| initial-consensus | 0.5600 | 0.5600 | 0.8400 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| initial-random | 0.5650 | 0.5650 | 0.8350 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |
| baseline-046-real-extension-G-full-J | 0.5700 | 0.5700 | 0.8300 | 1.0000 | 0.4000 | 1.5000 | 0.0010 | 0.4500 | 0.4600 | 0.8500 | 1.0000 | 0.7000 | 0.6000 |

- 20 merged chr 中心（每 copy 用共通 valid bins 均值后两 copy 等权）；单一尺度 `a=Σ(pred·ref)/Σ(pred²)`，`stress=sqrt(Σ(a·pred−ref)²/Σref²)`；每 chr 19 条距离 profile 用 min_pairs=2 的 helper。
- 190 距离整 chr 标签置换 null（seed 461100，9999 draws，dyadic 依赖，不是独立 pair 检验）：见上表 p 值。
- copy center（40 点/760 跨 chr 距离）对应由每 chr whole-chr signed-Pearson `derive_rho` 规则固定，同 chr homolog pair 不入距离向量；tie/undefined chr：A-raw-consensus=无; A-raw-random=无; A-ms-consensus=无; A-ms-random=chrX; B-raw-consensus=无; B-raw-random=无; B-ms-consensus=无; B-ms-random=无; C-raw-consensus=无; C-raw-random=无; C-ms-consensus=无; C-ms-random=无; initial-consensus=无; initial-random=无; baseline-046-real-extension-G-full-J=无。
- 20 center 相似度用单个 global Procrustes（proper 与允许反射两版，报 det）；反射不改变距离类指标，只改变 aligned RMSD。

## 6. null：u=0 与 16 个 random-u

null 设计：8 个来源（6 个按 count 选中的 loss×solver endpoint + 2 个 initial）× 17 个变体（u0 + seeds 450500..450515）= **136 个新文件**；046 同 candidate SHA 复用 baseline null **17 个**；合计 **153 个 null draw**（`evaluation.json.null_draw_count`=153）。

| source | null kind | draws | P matched mean | P matched min | P matched max | P cross mean | P contrast mean | P contrast range | inter P mean | inter P range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-raw-consensus | u_zero | 1 | 0.3000 | 0.2500 | 0.3500 | 0.3000 | 0.3000 | 0.1000 | 0.3000 | 0.1000 |
| A-raw-consensus | random_u | 16 | 0.3000 | 0.2500 | 0.3500 | 0.3000 | 0.3000 | 0.1000 | 0.3000 | 0.1000 |
| initial-consensus | u_zero | 1 | 0.3000 | 0.2500 | 0.3500 | 0.3000 | 0.3000 | 0.1000 | 0.3000 | 0.1000 |
| initial-consensus | random_u | 16 | 0.3000 | 0.2500 | 0.3500 | 0.3000 | 0.3000 | 0.1000 | 0.3000 | 0.1000 |

- `u=0` 的 margin 类指标为 NA（tie 规则下每 chr margin 未定义），符合 PLAN §8 的预期；
- 16 个 random-u 是同一细胞内的技术性置换，**不是生物重复**，只用于给出同格对照的 mean/range。

## 7. paired bootstrap（seed 450301，10000 draws，20 chr 配对）

| comparison | left | right | metric | mean diff | CI2.5 | CI97.5 | left wins | right wins | ties | defined chr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| endpoint-minus-baseline | A-raw-consensus | baseline-046-real-extension-G-full-J | pearson:matched | 0.0100 | -0.0100 | 0.0300 | 12 | 8 | 0 | 20 |
| endpoint-minus-baseline | A-raw-consensus | baseline-046-real-extension-G-full-J | pearson:contrast | 0.0000 | -0.0200 | 0.0200 | 10 | 10 | 0 | 20 |
| endpoint-minus-baseline | A-raw-consensus | baseline-046-real-extension-G-full-J | spearman:matched | 0.0100 | -0.0100 | 0.0300 | 12 | 8 | 0 | 20 |
| endpoint-minus-baseline | A-raw-consensus | baseline-046-real-extension-G-full-J | spearman:contrast | 0.0000 | -0.0200 | 0.0200 | 10 | 10 | 0 | 20 |

- 全部 20 chr 定义时才给 fixed-20 估计；胜出数分母为 20。比较包括每个 endpoint − baseline、同 source 的 B−A / C−A 与 ms−raw。

## 8. 验证与回归

| 检查 | 结果 |
| --- | --- |
| baseline_regression | PASS |
| null_draws_evaluated | 153 |
| reference_sha256_verified | yes |
| smoke | no |

- baseline 回归：与 046 既有指标表在 atol=1e-12 下核对 168 个数值，mismatch=0（status=PASS）。
- mask：common 2447 valid loci = 4894 beads

## 9. 限制与不可声称项

- 本轮只做 R2 与空间读出；**不做 R1/R3**，不声称 L2（整条染色体一致拷贝身份可恢复）。
- 结构尺度未校准：使用 rank/单一 scale/Procrustes 口径；copy 内相关高不等于同源拷贝恢复。
- `copyA`/`copyB` 是标签规范自由度；跨拷贝指标按几何最佳互换（tie<=1e-12 记 unresolved），并给出 `u=0` 与 `random-u` 对照。
- sorted inter 四距含高背景，random-u 保留中心而非 packing null；不把 raw r 当恢复率，也不把「与 random-u 差异小」写成「不可区分」。
- 生物重复 n=1；20 chr 是同一细胞内的关联测量，bootstrap 只描述细胞内技术/结构变异。
- 046 baseline 为事后采纳的工作对照，且 FG 预算与分辨率组合与本轮不同。
- 几何效果不直接等于 phase/L2 恢复；本轮不读 phase 列。

## 10. 产物清单

| 路径 | 内容 |
| --- | --- |
| results/evaluation.json | 全部 dataset 指标、null draw、选择与支持集 |
| results/r2_per_chromosome.tsv | 逐 chr 四 rho 与 matched/cross/contrast/margin |
| results/r2_summary.tsv | fixed-20 macro 与 defined-only 描述列 |
| results/endpoint_common_g.tsv | 15 行 own 目标值与共同原 G count/fullJ（不重算） |
| results/inter_summary.tsv | pooled 四距与 4 个 order statistic |
| results/spatial_summary.tsv | 中心/stress/置换 null/copy center/Procrustes |
| results/null_per_draw.tsv | 153 个 null draw 的逐 draw 指标 |
| results/null_summary.tsv | 按 source×kind 的 mean/range |
| results/paired_bootstrap.json / .tsv | 28 组配对比较的 CI 与胜出数 |
| results/validation.json | mask/paths/回归/gate 一致性 |
| results/terminal.json | 端点与评价终态 |
| plots/049_figure1_core_metrics.png | 核心指标比较（12 fit + baseline + 2 initial） |
| plots/049_figure2_whole_genome.png | ref + baseline + 3 loss display endpoint 5 panel |
| gates/pre_reference_gate.json | 封存 gate（含首次打开 reference 时间） |
