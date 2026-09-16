# 已取代的准备目录

该准备目录在任何真实 variant 阶段启动前已被取代。

请使用以下哈希一致的替代目录：

`docs/audits/multires-variant-preflight-20260914_053514/`

原因：该目录写入后，controller 收到了一次保持语义不变的空行规范化，因此原始 source hash 不再与当前 controller 字节一致。该失败尝试作为审计证据保留；其 `prepared_not_launched` 意图没有启动 native FDG，也没有启动任何 C1/C2-map/C2-free/C3 阶段。
