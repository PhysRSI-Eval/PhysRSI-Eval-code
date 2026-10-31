# 方法适配表 v1（插充电器，唯一状态；84 个状态，其中 83 个有 VS 结果）

| 接手步 | 抬起时 v4 | VS 成功 | rg3f 成功 | Astra 成功 | 都不成功 |
|---|---|---|---|---|---|
| t<=100 | lift_abstain | 6 | 0 | 0 | 5 |
| t<=100 | v4_ok | 22 | 7 | 1 | 18 |
| t>100 | lift_abstain | 5 | 0 | 0 | 4 |
| t>100 | v4_ok | 7 | 0 | 1 | 8 |

"成功"取第一个成功的方法，按 VS → rg3f → Astra 的顺序判。VS 结果优先取 C1 线 o7 的真值上限（TRUTH-CEILING-combined-o7.json，83 个唯一状态，用真值关键点），没有 o7 时取 o1–o5 开发轮次。逐行数据见 fitmap.json。
**v1 更新（10-30 12:0xZ）**：抬起时 v4 弃权的 20 个状态里，VS 执行器用真值关键点插好了 11 个。VS 不需要在手位姿，所以绕开了这个 D2 缺口。
部署版还取决于 KPNet 握持物关键点线（原先 0/20 的读法见 FITMAP-v0.md）。
deposit_coin 的路由：CAD 模板线索不适合薄圆盘，coin 沿用已见任务上的 E1。
takeover 标签（ROUTER-BIG 的 takeover 头）：以每个唯一状态的 best 方法为准（fitmap.json），VS 上限 40/83。
