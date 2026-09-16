# 协议文字澄清（不改变冻结规则）

1. 冻结 train threshold literal 仍为 `14757395258967642112`。它等于
   `int(float(0.8) * 2**64)`；精确有理数 `floor((4/5) * 2**64)` 是
   `14757395258967641292`。两者相差 `820 / 2**64`，对约 80% 比例无可见影响。本轮不得改用
   后者，也不得因此改变 split。
2. `config.json` 的 `fold_domain` 是 byte string 的可读表示。实际 domain bytes 是 ASCII
   `P9016-contact-fold-v1` 后接一个且仅一个 NUL byte `0x00`；不能编码成反斜杠与字符 `0`。
   fold 工程门必须包含固定编码 hex 例和端点反向后 hash 相同的测试。
3. 四个实验3 endpoint 及其全部 splice 共同使用 055 frozen legacy support；不得按 condition 或
   endpoint 各自建 mask。结构 primary 为 Pearson、20 条染色体等权。1 Mb 共同支持必须有恰好
   157,529 个 within-chromosome pairs；若减少，必须显式报告并暂停评价审查。
4. 同一 record fold 在 5/2/1 Mb 各层均从原始 train records 重新聚合。G-original 的 e 使用该层
   全部 train endpoints（diag 贡献两端），G-offdiag-e 使用该层 offdiag train endpoints；不能将
   1 Mb offdiag 集贯穿粗层。held-out primary 只在 1 Mb 计算。
5. 辅助上限64的单位是实际 full eligible-pair kernel sweep；forward/back 分开，任何额外 full-pair
   regularizer kernel也单列。训练不得调用每 stage 预算外 GPU init/readback FG：直接复用同一
   `run_budgeted_lbfgs`，首次 solver evaluate 在训练 FG cap 内；初始化只做 CPU finite/hash/geometry
   检查。只有四个最终1 Mb endpoint各做一次必要单-forward value readback；test四次与32个 splice
   均单-forward。实际 auxiliary ledger 按 sweep 权重逐项累计，不以 logical calls 替代。
