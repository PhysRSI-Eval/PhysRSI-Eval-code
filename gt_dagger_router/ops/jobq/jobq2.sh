#!/bin/bash
# jobq2：同 jobq.sh，但用 mv 作原子"认领"，同一个队列可以同时起多个守护进程并行执行（每个任务自己的 --gpus 必须互不重叠）。
# 用法：bash jobq2.sh <队列目录> [工人编号]
Q=$1; W=${2:-0}; mkdir -p $Q/pending $Q/running $Q/done $Q/logs
while true; do
  f=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$f" ]; then sleep 60; continue; fi
  b=$(basename $f)
  mv $f $Q/running/$b 2>/dev/null || { sleep 2; continue; }   # 别的工人先认领了
  echo "$(date -u +%FT%TZ) w$W start $b" >> $Q/jobq.log
  bash $Q/running/$b > $Q/logs/$b.log 2>&1; rc=$?
  mv $Q/running/$b $Q/done/$b; echo "$(date -u +%FT%TZ) w$W done $b rc=$rc" >> $Q/jobq.log
done
