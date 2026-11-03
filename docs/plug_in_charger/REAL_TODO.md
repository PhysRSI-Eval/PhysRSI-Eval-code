# 交给真机接入负责人的 TODO

硬件按用户提供记录：方舟无限 ARX X5 **2023 款**；腕部 RealSense **D345 ×2、D405 ×2**；头部 **Gemini D335L**。型号字符串先保留原话，现场核对 SDK 型号与序列号，不自动替换型号。

- [ ] 在 `/lizekai/RSIws/real/site_driver.py` 实现 `examples/real_robot/site_driver.todo.py` 的五个接口；启动不会自行 reset。连接、急停、碰撞/力限制、失联后的保持/停止由现场控制器实现。
- [ ] 为每个物理设备记录序列号、左右归属和 RGB/深度流。实验协议目前只展示 `cam_high / cam_left_wrist / cam_right_wrist` 三路 RGB。明确每侧选 D345 或 D405 的哪一路，所有组固定一致；其余两路可记录但不能只向某组额外展示。若要五路作为输入，另建全部条件一致的协议版本。
- [ ] 测量并冻结内参、畸变/去畸变、手眼、头部外参、双臂基座、TCP、夹爪开度和关节顺序。统一米/弧度、wxyz 单位四元数、link6 和共同场景坐标系，写 calibration_id。真实镜头不得使用 `sim-tool-profile.json` 的内外参。
- [ ] 实现同步 `observe()`：uint8 RGB、14D（每臂六个关节弧度 + 归一化夹爪测量值）、两个 link6 位置/四元数、当前/剩余控制步。公共 packet 不包含隐藏目标真值或成功判据中间值。
- [ ] `reset(layout_id)` 取得人工准备好的独立现场、唯一 episode_id、原始指令和 400 步预算；不要重放仿真 reset 轨迹。`act()` 在 25 Hz 下执行 1..5 tick，保存硬件命令 ID/ACK，返回确切 executed_steps/step_id/terminated/truncated/success。未知 ACK 不重发。
- [ ] 定义并冻结真机任务原生成功判定（插入/释放及复位语义与任务定义对齐），`finish()` 返回 valid/native_success/native_score/native_control_steps/reason。不能由工具估计或模型宣称替代；真机判定与仿真传感器差异需记录，不能声称像素级相同。
- [ ] 第三组：验证 bounded-DLS 控制器的关节限制/0.05 rad tick 等与 `reach_model.py` 一致；支持执行器连续开度和最终 EEF 目标后才在 metadata 声明 `executor_targets: bounded_dls_v1`。不匹配时实现新的 reach/controller 适配并重新验证。
- [ ] 第三组：重新部署在手/插孔/粗定位服务，校准真实镜头、裁剪、TCP 和 charger/power-strip CAD；记录 inference_calibration_id 与主 profile 一致，验证 real RGB 输入。不把默认仿真网络精度作为真机已验证精度。
- [ ] 填写 `configs/plug_in_charger/real.todo.json` 和 `real-tool-profile.todo.json`；先跑无机器人测试、只读 check，再由现场人员执行一条动作/ACK/终态冒烟。逐级核验之后开展真实对比。
- [ ] 第一阶段从空真实 workspace 积累，真实原生成功后导出 bundle 做 real→real；第二阶段才独立做 sim→real，保持 source_domain、源 reset/hash、目标 reset、读取回执及动作/结果可追溯。

启动（完成 TODO 后）：

```bash
PYTHONPATH=/lizekai/RSIws/real python -m physrsi_eval.real_server \
  --driver site_driver:make_driver --config /lizekai/RSIws/real/driver.json \
  --state-dir /lizekai/RSIws/real/plug_in_charger/service --port 26710
```

服务绑定 localhost，跨机器用 SSH 隧道。`driver.json` 的 `metadata` 必须包括 backend=real、task=plug_in_charger、max_episode_steps=400、robot/frame/eef_link/control_hz/calibration_id/calibration_source（real_measured）。后五项必须与实验 profile 相同。

`owner.lock` 保证一个 state-dir 的单控制者；站点还应在 SDK 层按机器人序列号锁定，不能通过换 state-dir 绕过。崩溃后的 `active.json` 或 `pending.json` 禁止自动复位/重放。现场核对硬件与完整 ACK、封存原回合后，再人工释放状态；禁止写定时脚本自动删除这些记录。
