# Plug in charger：三组 sim / real 对比

目标全部为 `plug_in_charger`。复用参考 K0（3 个独立 Context 回合）和 K1–K17（同一条链逐轮积累 workspace，原生成功即停）的流程。固定同模型、预算、任务指令、400 原生控制步、25 Hz，以及同组配对的布局/初态和相机输入。K0 不加载 workspace；第二组的种子从 K1 生效，不能把带经验的回合标为无经验 K0。

- **scratch**：原有基础工具，workspace 从空开始。sim / real 使用同一个调度、工具和 revision 代码。
- **experience**：与 scratch 相同的工具权限，只增加成功来源的 memory / workspace。先实际积累真机成功并做 real→real；然后以单独运行名做 sim→real。同域和跨域不要合成一个成绩。
- **tools**：基础工具 + 当前 hv2f4 的显式 `insert_charger` 冻结执行链；workspace 从空开始。先由 actor 抓取、抬起，再按需要调用工具；工具没有自动接管。包含 v4 在手估计、LOC1 插孔定位、兼容重抓、转运和 R1 插入。现有其他研究工具保留在 `sim_real/`，不是全部默认授予。

本次为代码移植与协议验证，**没有执行新的 Isaac 实验或真机实验**。真机 SDK 接入由现场同事负责，见 [REAL_TODO.md](REAL_TODO.md)。模板明确缺少标定时会拒绝开始，不用仿真标定伪装开箱可用的真机。

## 安装和入口

```bash
cd /lizekai
# 仓库的新分支包含本目录及全部小型代码；大文件见 assets.json
 git clone -b plug-charger-three-arm-sim-real-20261103 https://github.com/PhysRSI-Eval/PhysRSI-Eval-code.git
cd PhysRSI-Eval-code
python3 -m venv .venv
. .venv/bin/activate
pip install -e . pytest
python -m pytest tests/ -q
```

从仓库根运行。编辑 `configs/plug_in_charger/sim.json` 中 RoboDojo 安装地址和 Python 地址，默认工作根为 `/lizekai`。复用原 `scripts/setup.sh` 安装仿真依赖；模型账号配置仍用 `configs/systems/gpt-6-astra.json`。不改变账号授权或预算。

```bash
# 第一组：只读预检，再运行 K0 + K1..K17
python -m physrsi_eval.plug_experiment check --config configs/plug_in_charger/sim.json --group scratch --backend sim
python -m physrsi_eval.plug_experiment run --config configs/plug_in_charger/sim.json --group scratch --backend sim --name layout0-r1

# 第二组的仿真经验版本（先做真机自身经验迁移，再做此对照）
python -m physrsi_eval.plug_experiment run --config configs/plug_in_charger/sim.json --group experience --backend sim --source-domain sim --bundle experience/plug_in_charger/sim-success-v1 --name layout0-from-sim

# 第三组：先部署 assets.json / services README 指定的推理服务
python -m physrsi_eval.plug_experiment run --config configs/plug_in_charger/sim.json --group tools --backend sim --name layout0-tools
```

真机接入完成后，复制 `real.todo.json` 为站点配置并填写测量值。以上各组命令改为 `--backend real --config /lizekai/RSIws/real/deployment.json` 即使用同一流程。第二组先用 `--source-domain real --bundle /lizekai/RSIws/real/success-bundle`；后续 sim→real 用上面的 sim bundle，另起 `--name`。可用 `--phase k0` / `--phase chain` 分开执行，不改变定义。多个条件必须分别重置现场，不能同时控制同一台机器人。

输出默认 `/lizekai/RSIws/runs/plug-three-arm/<backend>/<group[-from-domain]>/<name>/`。实验配置、种子 hash 冻结；换标定、种子、模型或工具版本要换运行名。中断的 round 不自动重放：先查原 reset / ACK / pending 文件。原生成功回合即使 revision 未完成也不得为了提交 workspace 重跑物理回合。

## Workspace 和迁移

代码自带 `workspaces/real/plug_in_charger/` 空脚手架，每轮由 launcher 初始化。共享方法、任务经验、机器人标定、现场布局是不同内容：前两者可迁移性需要审核；后两者绑定机器人与 reset。`memory/environment/scene.json` 每次 reset 清空旧绑定，写入本轮 episode ID；从本轮 RGB / 本体重新发现现场，而非复用旧位置。

```bash
python -m physrsi_eval.workspace audit /lizekai/RSIws/runs/.../K03/snapshot
# 逐文件填写 classifications.json，值为 portable / episode_local / environment_bound / software_bound
python -m physrsi_eval.workspace export /lizekai/RSIws/runs/.../K03 /lizekai/RSIws/real/success-bundle --domain real --classifications classifications.json
```

导出要求已记录的原生成功结果、reset、域与 committed snapshot；不是新增任务成功判据。全部原文和来源 hash 保留在只读 `memory/library/`；只有明确审核为 portable 的文件进入活跃 skills/tools/policy/memory。原始坐标可供追溯，但不作为当前动作目标。绝对文件路径应改为相对 workspace 路径；机器人/场景绝对位姿必须重新测量。启发式 `audit` 只发现风险，不能证明文件可迁移。

已附四份仿真成功的原始经验包（含动作账本与定位返回），并附可迁移方法整理。它们是 **tool-assisted** 仿真成功，原始程序 workspace 未完整保存，因此这里是“成功经验重建的 workspace”，不能称为原始成功程序快照。原失败 K3 不放入成功库。没有伪造真机成功样本。积累真机完整快照后，优先替换为真实的成功 workspace 导出。

`workspace-loaded.json` 仅记录文件已提供；`client/workspace-reads.jsonl` 记录通过文件工具实际读取的路径/hash/reset。读取不等于物理采用，需结合动作 ACK 与结果分析；程序内部读取没有逐文件追踪。K0 没有 workspace。

## 工具移植边界

`physrsi_eval/plug_tools/SOURCES.json` 记录 hv2f4 源码和打包 hash。算法、阈值和工具内部最终目标控制方式保留；修改为包内导入、显式标定/URDF/桌面高度、统一 episode/ACK 接口。`insert_charger` 可发较远最终 EEF 目标，由原生 bounded-DLS 控制器限制每 tick 关节步长；基础 actor 仍受 5 cm / 0.35 rad 提案限制。不能把最终目标裁成直线来声称等价：原算法记载这种方式会在腕部奇异位姿停滞。

真机第三组必须实现和验证相应控制器，测量桌面、TCP、夹爪宽度、双臂基座、相机和目标 CAD；在手推理自身还有头部/腕部标定与裁剪假设。服务代码的仿真权重不代表在真实 RGB 上已经成功。工具返回诊断/弃权，最终成绩仍来自现场原生任务判定。
