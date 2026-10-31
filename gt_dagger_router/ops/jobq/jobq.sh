#!/bin/bash
# 每条线、每台主机一个顺序队列：pending/ 里的 *.sh 按文件名顺序逐个执行，跑完自动接下一个。
# 目的：一轮实验结束后不需要等子代理回来就能接着跑下一轮（子代理会被额度/重启打断）。
# 用法（在目标主机的 tmux 里）：bash jobq.sh <队列目录>；加任务＝往 <队列目录>/pending/ 放一个 NN-名字.sh。
Q=$1; mkdir -p $Q/pending $Q/running $Q/done $Q/logs
while true; do
  f=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$f" ]; then sleep 60; continue; fi
  b=$(basename $f); mv $f $Q/running/$b
  echo "$(date -u +%FT%TZ) start $b" >> $Q/jobq.log
  bash $Q/running/$b > $Q/logs/$b.log 2>&1; rc=$?
  mv $Q/running/$b $Q/done/$b; echo "$(date -u +%FT%TZ) done $b rc=$rc" >> $Q/jobq.log
done
