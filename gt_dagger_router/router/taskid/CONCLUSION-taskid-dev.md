# 任务识别 router（router v4 第 1 步，开发级；2026-10-30）

数据：artifacts/native 中带逐步画面的 Astra 回合，覆盖全部 42 个任务，共 4688 个（最少的是 general_pickup，3 个）。逐任务清单见 taskid/episode_index.json。
模型：DINOv3-B/16 对头部和两路腕部画面取 [CLS, 平均 patch]，再接 MLP 做 42 类分类。

| 时刻 t | 回合数 | 准确率 | 平衡准确率 |
|---|---|---|---|
| 0 | 4688 | 1.0000 | 1.0000 |
| 10 | 4152 | 0.9998 | 0.9940 |
| 20 | 4112 | 1.0000 | 1.0000 |
| 40 | 4067 | 0.9978 | 0.9818 |
| 80 | 3842 | 0.9974 | 0.9759 |

分组方式（测量修正）：
- 第一版按上下文分组。其中 16 个任务只有一个上下文，整组被留出，等于没见过的任务，准确率为 0。这是测量问题，不是 router 的错误。
- v2 改为：任务有 ≥5 个上下文时按上下文分组；有 ≥2 个布局时按 (上下文, 布局) 分组；否则按回合划分，此时测试回合和训练回合出自同一场景。

**跨上下文**（新种子 / 布局）检验的任务共 7 个，全部正确：build_tower, deposit_coin, play_Xylophone, plug_in_charger, pour_by_language, pour_liquid_into_cup, push_T。
**只能做场景内检验**（同一种子 / 布局的其他回合）的任务有 35 个，准确率也 ≈1.0，但这只说明它能认出见过的场景。
正式测试会换种子和布局，所以这些任务还需要更多上下文的数据。

混淆极少：t ≥ 40 后，偶有 deposit_coin ↔ plug_in_charger、classify_objects ↔ classify_objects_by_language、stack_blocks ↔ stack_blocks_by_language，各 1 次。
按语言区分的任务对（*_by_language）在后面的时刻偶有混淆；画面看不到语言指令，部署时可以用指令文本区分（是否允许另议）。
## 新种子 / 新布局检验（10-30 10:2xZ，零额度；队列 main-rt 20-；RESULT-taskid-fresh.json）
- **环境的种子是固定清单。** 用清单外的种子做 reset，会报 "Seed N not found in seed list"。
  - 34 个任务只有 1 个公开种子（布局 0）。评测场景就是存档里的场景，所以对固定基准来说，场景内检验就是该做的检验。
  - 只有 build_tower（55 个公开种子）和 classify_objects_by_language（55 个）有存档里没出现过的种子。
- **结果**：
  - 新种子的开局帧：build_tower 10/10，classify_objects_by_language 10/10 认对，和 classify_objects 没有混淆。
  - 留出布局 3/8/13（7 个多布局任务，298 个回合）：准确率 1.00，没有混淆。
- **读法**：开局一帧就能可靠识别任务。预期中的语言变体任务对，在开局帧里没有混淆；t ≥ 40 后偶有 1 次混淆，到时可以用指令文本区分（是否使用由用户决定）。
