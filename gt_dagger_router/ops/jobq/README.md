# 实验执行规则（用户 10-31：拉实验必须是 tmux 脚本，与子代理解耦，子代理只盯实验）

1. **所有实验只通过队列执行**：写成 `/lizekai/RSIws/ops/jobq/<主机>-<线>/pending/NN-名字.sh`，由该主机上 tmux 里的 `jobq.sh` 守护进程按文件名顺序执行。
   - 守护进程：c1-jobq-{aux,h1,h2}、kp-jobq-{h1,h2}、rt-jobq-main、rt-jobq-big。
   - 子代理**不得**在自己的会话里直接启动 python/sim/训练进程，也不得用 `&`、nohup 或手写的 tmux 绕过队列。
   - 唯一例外是 H200：它只能经中转执行，必须在 H200 的 tmux 里启动，脚本同样落盘为文件。
2. **子代理只做三件事**：读结果（out/、results/、DASHBOARD.md）、归因、把下一轮写成 .sh 放进 pending/。
   - 每条线始终保持 ≥2 轮在排队，保证子代理被额度、重启或 API 错误打断时，显卡照样有活干。
3. **停实验**：只能用 process group（setsid + kill -- -PGID）或已核实命令行的 PID，并在 STATE 里记录。
4. **看进度**：reports/progress/DASHBOARD.md 每 3 分钟自动刷新（tmux dash-1030），不需要问子代理。

## 并行（10-31 03:1xZ 起）
- jobq2.sh 用 mv 原子认领，同一队列可以起多个工人并行执行。前提是每个任务的 `--gpus` 或 CUDA 设备互不重叠。
- 只等待前置结果、不占 GPU 的汇总任务（例如 B1、B3）不要和 GPU 任务排在同一个单工人队列里，以免阻塞后面的任务。
- 现有并行队列：h1-c1（工人 w1、w2）、main-rt2（main 上 3 个工人，供路由器线把可并行的项放进去）。
- kp（工具线 ip4，10-31 03:1xZ）：h1-kp 3 个工人（kp-jobq2-h1-w1..3，item 21 结束后由 h1-kp/switch-to-jobq2.sh 自动切换），h2-kp 2 个工人（kp-jobq2-h2-w1/w2），h2-kp-wait 1 个工人（kp-jobq2-h2wait，只放等待 C1 标签的 DAgger 项，GPU7）。每个任务只用一张卡：h1 用 GPU5/6/7（GPU4 留给 21），h2 用 GPU5/6，DAgger 用 GPU7。
