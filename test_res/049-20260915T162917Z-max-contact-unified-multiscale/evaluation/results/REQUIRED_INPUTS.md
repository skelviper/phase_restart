# 049 评价侧所需输入契约（pre-reference 阶段）

本文件由评价执行方在 gate 之前写出，说明 `evaluation/source/eval049_prep.py` 实际消费的字段。
字段名以父侧冻结 schema 为准；下表列出接受的别名，缺失必需字段会**硬失败并打印可用 key 列表**，
不会静默变成 `None`。

## 1. 端点坐标（12 fit）

- 路径：`049/coords/{fit_id}/1Mb.npz`，`fit_id = {A|B|C}-{raw|ms}-{consensus|random}`
- 必需 key：`coordinates`，shape `(2, 2645, 3)`，float64，严格在单位球内
- 期望 key（有则核对）：`raw_y`、`theta`（`6*2645+1`）、`p`、`q`
- 同目录 `1Mb.3dg`：必须存在，且回读必须与 `coordinates` 逐元素相等或 `<=1e-12`（`READBACK_ATOL`）

## 2. `results/endpoint_manifest_pre_reference.json`

必需顶层：`fits`（恰好 12 条，fit_id 覆盖 3 loss × 2 solver × 2 source）；建议同时给
`initials`（consensus/random）与 `baseline`。

每条 fit 必需字段（别名见括号）：

| 字段 | 别名 | 说明 |
| --- | --- | --- |
| `fit_id` | `id` | 必须与目录名一致 |
| `loss` | `loss_id` | A/B/C，且与 fit_id 一致 |
| `solver` | `solver_id` | raw/ms |
| `source` | `blind_source`, `source_id` | consensus/random |
| `coords_npz_path` | `npz_path` | 绝对或相对 run 目录 |
| `coords_npz_sha256` | `npz_sha256` | 文件字节 SHA256 |
| `three_dg_path` | `tdg_path`, `3dg_path` | |
| `three_dg_sha256` | `tdg_sha256`, `3dg_sha256` | |
| `terminal` | `status`, `terminal_state` | `converged` / `budget_not_converged` / `not_converged` / `failure` |
| `outerFG` | `outer_fg`, `outer_fg_actual`, `accepted_fg` | 实际 outer FG |
| `count_nll_by_loss` | `count_nll` | `{A,B,C}` → 数值或含 `own_count` 的 dict |
| `fullJ` | `fullJ_value`, `full_j` | |
| `p` / `q` | `p_final` / `q_final` | |

可选但会被读取：`coordinate_array_sha256`（有则核对）、`kind`、`fit_wall_seconds`、
`canonical_gradient_max_abs`（也可放在逐 fit JSON 中）。

## 3. 逐 fit 终态 JSON

- `049/results/fits/{fit_id}.json`（首选）与 `049/stages/{fit_id}/1Mb.json`
- 抽取并交叉核对：`terminal`、`outer_fg`、`full_j`、`fit_wall_seconds`、
  `canonical_gradient_max_abs`、`count_nll_by_loss`、`p`、`q`、`coords_npz_sha256`、`three_dg_sha256`
- 与 manifest 数值不一致即硬失败；两个文件都不存在会记录 `missing` 并注明「按 manifest 字段处理」

## 4. `results/selection_pre_reference.json`

- 必需：`reference_opened` 不得为 `true`
- 必需：`per_loss.{A,B,C}.source_selection.{raw,ms}`，含 `selected_source`
  （别名 `selected`/`winner`/`source`）、可选 `rule`/`candidates`/`margin`
- 必需：`per_loss.{A,B,C}.display_endpoint`，含 `solver`、`source`、可选 `count` 与 `coordsSHA`
- 评价侧复算规则：每个 `loss × solver` 内取 own count 最小者，差 `<=1e-9` 取 consensus；
  display 为每 loss 4 个 endpoint 中 own count 最小者。复算结果与文件不一致即硬失败
  （`selection_checks`）。loss 之间不做数值排名。

## 5. initial 与 baseline（只读复用）

- `045/coords/initial/real_consensus_1Mb.npz` / `real_random_1Mb.npz`
- 血统以 `045/inputs/formal_manifest.json` 为准：`source_sha256` = `e766...`（consensus）/
  `9a48...`（random，014 approved blind root）；并核对 `sha256`、`coordinate_sha256`、
  `raw_y_sha256`、`p_init=0.75`、`no_optimization=true`、`source_root` 含 014
- baseline：`046/coords/real-extension-G-full-J/1Mb.npz`（SHA `116e906e...`，coordinate array
  `855e3f8a...`）+ 046 同 candidate SHA 的既有 17 个 null 文件（逐个 hash 复核，并本地重算
  `u0` 与 2 个 random-u seed 做 parity）

## 6. gate 流程

1. `eval049_prep.py`：核对全部输入 → 生成 8 个来源（6 个按 count 选中的 endpoint + 2 个
   initial）各 `u0 + 16 random-u`，共 136 个 null npz → 写出
   `evaluation/gates/pre_reference_gate.json`（`reference_opened=false`）。
2. 父侧确认「所有正式端点已封存」后，`eval049_evaluate.py --authorized-by-parent "<原文>"`
   才打开 `data/P9016.1m.3dg.gz` 并记录首次打开 UTC；mask snapshot 同样只在 gate 之后加载。
3. `eval049_plots.py`：两张 PNG（仅 gate 之后）。
