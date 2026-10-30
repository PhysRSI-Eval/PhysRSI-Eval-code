# RSI harness v2 — release candidate（能力线，2026-10-28 22:xxZ；零额度，离线，未武装，未上线）

规格：experiments/harness-design-20261027/HARNESS-v2-DRAFT.md。底座仍是原有的 RSI harness（技能库为空、Astra 跨 K 轮自行迭代）；本包只增加工具、事实和经验包。

## 内容
| 层 | 文件 | 内容 |
|---|---|---|
| L1 工具注册表 | rsi_v2/registry.py, rsi_v2/contract.py, skills/<tool>/SKILL.md | 统一返回 {value, uncertainty, abstain, steps_used, note}，带契约检查；违反契约的结果不交给 Astra。注册的都是过了预登记关卡的工具：locate_mm v2（outlets, keyhole, toolbox_slots, tube_holes, mahjong_tile）、inhand_pose v3（只限充电器）、check_reach v2、predict_outcome v3（W7 PASS）、grasp_compatible v1.1（只开 info 模式，并写明适用范围：仅限第一阶段结束、手臂空闲的状态；不是通用工具）。每个工具配一张英文 SKILL 卡，写明证据和适用范围 |
| L2 状态块 | rsi_v2/monitors.py | 由 v1.7b / b3 的 monitor_hub 泛化而来：hub 里不写任何任务规则，过了第 1 关的监视器以插件注册（现有 kong_p2）。按步号在决策点送达；只给事实："确认"（相机读数）、"已发出"（Astra 自己的闭合次数）、已用/剩余步数 |
| L3 显示规则 | rsi_v2/surfacing.py | 工具结果只作为 Astra 自己调用的回复；主动出现的只有事实，并经过建议措辞过滤器；执行器模式只在显式调用时运行（依据：plug M3 3/52 对 24/48、coin c3 4/44、b3 TSTAVE 第一阶段 0/5 对 3/6）；check_hold 不主动显示；每 25 步最多一个块 |
| L4 经验包 | rsi_v2/experience.py | RoboSkill 式 schema rsi-exp-v1（动作账本含工具调用和"下一步是否用了"、关键帧、binding、程序索引）。验证器只查结构和引用；经验和通用事实里不允许出现数值字面量，必须改用 binding 中的名字，每个名字都要有重算方法（我们的工具、本体、CAD 或机器人常数）；给一次修复机会，之后冻结 |
| 粘合 | rsi_v2/harness.py | 每轮：工具调用 → L3 守卫 → L1；动作后 → L2 + L3（进入精细阶段时给一次"工具可用"的事实）；回合结束 → L4 |

## 离线冒烟（smoke/，结果 out/SMOKE-harness-v2.json、out/SMOKE-L4.json）
- 4 个任务的已存 Astra 回合（make_kong、plug_in_charger、play_Xylophone、insert_tubes），每 5 步一个决策点，共 396 个决策点，读存档帧和本体；没有仿真，没有在线。
- 工具调用全部符合契约（0 违反）：
  - mahjong_tile 在两臂回 home 时 0.10 mm 残差，否则弃权并写明前提；
  - grasp_compatible info 给出候选（附适用范围说明）；
  - outlets 在 budget_steps=0 时动臂前就弃权；
  - inhand_pose 因成员分歧弃权；
  - predict_outcome、check_reach 正常返回；
  - tube_holes 因 SAM3 粗定位没找到而弃权。
- 增加的 token（按字符数 / 4 估计，本机没有分词器）：
  - 每轮平均 0–3.5，单轮最多约 90（一个状态块）；
  - 对照存档中回复文本的中位约 900–950 token / 轮，全回合占比低于 0.4%；
  - 工具列表约 1000 token / 次请求（可缓存）。
- L3：主动文字中的建议措辞 0 处；植入的建议句被过滤掉；执行器自动调用被拦，显式调用放行；check_hold 自动显示被拦。
- L4：合格包判 valid；坏包（硬编码坐标、不存在的工具、越界步段、无帧关键帧）报出 5 条错误，一次修复后判 valid，不修复判 invalid_no_repair。

## 冒烟中的发现（已处理）
- **xylophone_keys 已从注册表移除**：它的第 1 关测试只用了 layout 0。在已存 Astra 帧上，layout 0 的残差为 0.4 mm，其余 17 个布局为 20–40 mm（smoke/xylo_layout_check.py）。和 push_T P-v1 是同一个教训。
- tube_holes、toolbox_slots、keyhole 的测试集没有记录 layout 字段，布局覆盖尚未核查。mahjong_tile 只在 layout 0 上验证过（make_kong 只有这个布局）。

## 武装前还缺
1. 接到 release 构建（参照 v1.8 / v1.7b 的 build 脚本）。服务部署：在手/插孔服务和 SAM3。
2. 核查 tube_holes、toolbox_slots、keyhole 的布局覆盖。
3. 用户批准。check_hold 和执行器模式保持关闭。

## 10-29 12:xxZ 更新（卡点线接手 harness 层；零额度，未武装）
- L1 新增：
  - insert_charger：rg3f 冻结链，执行器，只在显式调用时运行；抬起时 v4 弃权就不动。
  - segment_small：SSEG 的 PASS 组，即 coin 头部和腕部、mahjong 头部和腕部。
  - predict_outcome 改为 listed=False：仍注册，但不列给 Astra。
  - register_kp_tool()：给 KPNet/KPT 用的注册口，合同写在 registry.py。
- L3：新增 rsi_v2/policy.py，包括 GateV2（冻结在 policies/gate_v2.json，tools/fit_gate_v2_final.py 生成）和 RouterV0 接口；harness.py 接线，有 log、offer、auto 三种模式，auto 需要 auto_approved。surfacing.py 的执行器列表加入 insert_charger，并加了 POLICY_TOKEN。
- 冒烟：smoke/smoke_l1l3_1029.py → out/SMOKE-l1l3-1029.json，全部通过（gate 重算 p 的差 ≤5e-5）。
- 改前备份：bak-20261029T1139/。
- 系统总图：reports/progress/SYSTEM-MAP-20261029.md。

## 10-29 14:xxZ 更新（卡点线）
- L1 新增 check_held_small：只限硬币，SSEG 抬起后检查；有英文 SKILL 卡，功能冒烟见 out/SMOKE-check-held-small-1029.json。
- L3 新增 RouterV0（policies/router_v0/，tools/fit_router_v0_final.py）：只允许 log/offer。冒烟 out/SMOKE-router-1029.json，67/67 一致。
- gate_v2.json 加了 dedup_refit_required 标记，在重拟合之前只跑 log 模式。
- hv2e（f15bb80a）里还没有 router 和 check_held_small；要上线需要另建 release，并做零额度冒烟。
