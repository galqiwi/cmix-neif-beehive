#!/usr/bin/env bash
# v3 hot-spare of the YOLO-2 production phase 1 (v6 1924): v6 kit tree + cfg, 8 GPUs accum 1, auto-resume, stop_at from cfg.
# usage: run_prod.sh <tree dir> <cfg.json> [PORT]. Training is pinned away from CCDs 28-48 (the full enc/dec pair on cores 35/42).
T=$1; C=$(readlink -f $2); PORT=${3:-36000}; OUT=$(jq -r .out $C); mkdir -p $OUT; STOP=$(jq -r '.stop_at // empty' $C)
source /home/user/work/env.sh >/dev/null 2>&1; source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd $T; fails=0; last=-1
while true; do
  [ -f $OUT/STOP ] && { echo "$(date -u +%FT%T) STOP file" >> $OUT/launcher.log; exit 0; }
  n=$(ls $OUT/stdout.*.log 2>/dev/null | wc -l)
  echo "$(date -u +%FT%T) LAUNCH attempt $n cfg $(sha256sum $C | cut -c1-16) tree $T" >> $OUT/launcher.log
  taskset -c 0-27,49-111 torchrun --nproc_per_node 8 --master_port $PORT rx7q.py $C > $OUT/stdout.$n.log 2>&1; rc=$?
  echo "$(date -u +%FT%T) EXIT rc=$rc" >> $OUT/launcher.log
  ep=$(jq -r .epoch $OUT/state/state.json 2>/dev/null || echo -1)
  [ -n "$STOP" ] && [ "$ep" -ge "$STOP" ] 2>/dev/null && { mkdir -p $OUT/snap_e$STOP; cp $OUT/state/* $OUT/snap_e$STOP/; echo "$(date -u +%FT%T) SNAPSHOT e$STOP $(sha256sum $OUT/snap_e$STOP/model.tch | cut -c1-16)" >> $OUT/launcher.log; break; }
  grep -q '"done"' $OUT/result.json 2>/dev/null && break
  step=$(jq -r .step $OUT/state/state.json 2>/dev/null || echo -1)
  if [ "$step" = "$last" ]; then fails=$((fails+1)); else fails=1; last=$step; fi
  [ $fails -ge 3 ] && { echo "$(date -u +%FT%T) GIVING UP at step $step" >> $OUT/launcher.log; exit 1; }
  sleep 30
done
echo "$(date -u +%FT%T) PHASE DONE" >> $OUT/launcher.log
