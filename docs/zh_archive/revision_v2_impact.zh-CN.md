# 024 Revision v2：固定参考映射审计

## 修订原因

v1 的 `swapped` 分支使用 `(a,b,c,d)=(rho_A_pat,rho_A_mat,rho_B_pat,rho_B_mat)`。这交换了两列参考，不是固定的 `ref1=mat/ref2=pat` 比较，尽管它保留了无序 margin 集合以及 aggregate matched/cross/contrast 代数。

v2 分支只交换候选行：

```text
(a,b,c,d) = (rho_B_mat, rho_B_pat, rho_A_mat, rho_A_pat)
```

direct 方向保持 `(rho_A_mat,rho_A_pat,rho_B_mat,rho_B_pat)`。因此 `margin_ref1=a-b` 始终是 mat-reference margin，`margin_ref2=d-c` 始终是 pat-reference margin。

## 锚定回归检查

新测试使用原始 `(rho_A_mat,rho_A_pat,rho_B_mat,rho_B_pat)=(0.1,0.8,0.7,0.2)`。由于 `cross > direct`，预期的固定参考行是 `(a,b,c,d)=(0.7,0.2,0.1,0.8)`，其中 `margin_ref1=0.5`、`margin_ref2=0.7`。测试通过。

所有 100 个有限、非平局的双拷贝行还通过了以下检查：

- 候选标签交换保持 `matched_ref1`、`matched_ref2`、`margin_ref1` 和 `margin_ref2` 不变；
- 参考标签交换保持 `matched/cross/contrast/minmargin`，并交换两个命名的 matched/margin 字段；
- `contrast=(margin_ref1+margin_ref2)/2` 以及 matched/cross 恒等式成立；
- 几何平局保留原始 rho，把固定参考的逐参考字段设为 n/a，从逐参考汇总中排除，并设 `contrast=0`。

真实 024 数据有 0 个几何平局，因此平局政策不会改变任何已报告的真实行数值。

## 影响清单

未改变：

- 三个锁定的 022 source 文件及其哈希；
- 原始 rho 值、配对分母、共同掩码、有限行计数；
- direct/cross 方向决定和方向计数；
- aggregate `matched`、`cross`、`contrast` 和 `minmargin` 值；
- margin 模式计数（`both_positive`、`one_negative`、`both_negative`、`margin_tie`、`geometry_tie`）；
- 没有编辑任何 022 文件。

重新计算或更正：

- cross-oriented 行逐染色体的 `a,b,c,d`；
- `matched_ref1/ref2`，其中 ref1 为 mat、ref2 为 pat；
- `margin_ref1/ref2` 及相应的逐参考汇总统计；
- `matched_ref1`、`matched_ref2`、`margin_ref1` 和 `margin_ref2` 的所有配对比较行；
- 逐参考绝对优势计数和贡献份额；
- `double_margin_scatter` 坐标及其 PNG/PDF；
- README 解释，包括主双拷贝条件中 ref2/pat 具有更大的描述性绝对贡献这一结论，而不是没有锚定的近似 50:50 主张。

## 哈希迁移

完整的 v1 哈希快照保存在 [`revision_v1_sha256_manifest.txt`](revision_v1_sha256_manifest.txt)。以下路径均相对于工作区根目录。

| 工件 | v1 SHA256 | v2 SHA256 |
|---|---|---|
| `test_res/024-20260913_133109-r2-allele-signal-derived/config.json` | `6effc586b508ff0c7ac793ca77d1f46cc107572ad4f9dcea153998080bb1a59b` | `68087a877bab7d6b349a06f12d7b53bece68898332e6e54b149666fa2651cee2` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/scripts/analyze_allele_signal.py` | `365231707d5d00bed9b724d0150e75f85ce37ad83114e9deda44b2dd1cb0535a` | `7cdb686c87797b67c53afb329eed28cbe25511cfaa5e2aa7cf579a6fb098879a` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/README.md` | `36949ee819a83b37075343a22f0c3acb4681d30626fe9f2c89c1501d779d5f77` | `c787db7141c9588541cafbbed6b09f63f7f1f7c4193eeecd7a41a56f340ef6aa` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/logs/terminal.log` | `0a65b4519ce4c5fb463de2037bb61dd5c76a83ae35e0f83ca9339c7b83069b81` | `bbe0d3c8f19e49e1d87af96e85fc0af80d601ed88fe1e6b30bae72d54b3c48b3` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_per_chromosome.tsv` | `21b8dd7345480b5fbe70cb77958c3253a083aed4f9363ff80a4221e11374bb98` | `7f48667672a5fb9be3269e5f0923ed122dc1c940f8b2d9a3dc20ced46ac649d4` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_paired_comparisons.tsv` | `3ab6f44981974cb78fa4a8221defdad55f2290d1c759bb8a78c46a710e86312f` | `e61e8ea5616acd82668382844bcaf931d26ee35fbcafe15ecbe76e90709428d0` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_results.json` | `0ebf2f11aeb2ea1b4c3f01450f0c100492d9db7df456bbf8117b9c39bf031979` | `39c75e2ca33d3b5ee751c601450884ca4e5034ed504b94d81cd2c6f78bddbd22` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_summary.json` | `6990f44911078893b154fefbdd8529f51b674e33d7e5254ba5ab1a3ccf7bffab` | `0516ff3a6bd7a1ff96805ed3edf4ab0848ff3d2bf254a7dfea0f9b11ef836403` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/provenance/input_manifest.json` | `15fd7f3452a05978a9fdf36bdbfd54c38a6fc6c5471524361e2c8adeaebf77d9` | `3123d9f7582ff220a48f57a9b795a368696eacd0a7768296d855191ead48c953` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/contrast_pairs.png` | `0c2f66828643984e353c161cd3aee3910a6bebec519ef6dd49c08636b83a07ef` | `0c2f66828643984e353c161cd3aee3910a6bebec519ef6dd49c08636b83a07ef` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/double_margin_scatter.png` | `cc33fc575d3b425aa6a1c6b071cf8f4c87bea4069dc5277520601e20686ee680` | `6060b74e62220513caebf6dbeb6ff350ba8f9497b276c262095d99d791a6e568` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/matched_vs_cross.png` | `2b91532e8a3096ba6d8bb441e380df0b463a91f77ff6ff89911f217d698a89e9` | `2b91532e8a3096ba6d8bb441e380df0b463a91f77ff6ff89911f217d698a89e9` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/contrast_pairs.pdf` | `5c4888783e017b860d741446ee9d354fe02628dd62aee84fe1edffa2a8344d08` | `61ce12b6a9f3df87affc01e80e4c3923304741a1dcd556a30f4b3a9aced2a967` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/double_margin_scatter.pdf` | `fafb0b3a23bed2166920fb9c13549b431511bbae26cdbc028473110df487b9d1` | `bf33f4c0e19bfaffa9bb6ce295a17e7bf5b5673ad2ce12a08514ae58adbfd433` |
| `test_res/024-20260913_133109-r2-allele-signal-derived/plots/matched_vs_cross.pdf` | `805634310e37370ae162dcff6b2bac4105c2b21655a9716980be43e1f933a411` | `4214825b23b97b685c99920d4424fd0ce750961038033fad457588b3ca5755ec` |
