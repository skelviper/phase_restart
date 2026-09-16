# 多分辨率变体预备

状态：`prepared_not_launched`。本目录只包含 C1/C2-map/C2-free/C3 的 24 个 stage 计划与 reference-free preflight；没有启动真实变体拟合，也没有读取 phase、reference 或 R2/evaluation 文件。

每个 variant/source 独立执行 `5m 300/930 -> 2m 200/630 -> 1m 240/750`，首层从同一盲 source physical start 开始，后续只传递该 variant 自己的 physical state 与 raw `q`。C2-free 的跨层传递与写出不调用 sphere inverse、clip、center/rescale。

真实启动必须使用 `launch_command.txt`，并在固定 033 `C0_gate.json` 为 passed 后由父侧显式放行。

专项测试与终端摘要见 `test_evidence.json` 和 `terminal_evidence.json`：7/7 通过，24 个真实 stage 仍为 0/24。
