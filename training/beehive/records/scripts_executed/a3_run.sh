#!/usr/bin/env bash
# v3: experiment E2 = Arm A3 (admin 2168, v6 2167/2169): canonical snap_e1344 + Arm A recipe with a 192-epoch cosine 1->0.025x,
# C1 c1_rate_v2 from update 0, rx7r 9696be34. init_only -> train (resumable, 3-failure rule) -> finish on a copy:
# fp32 1 ep + strict W2048 + C4 + gate (driver) -> W1536 export (42a5f84d) + gate + strict W1536. Posts results to hutter_research.
set -u
Y=/home/user/work/y2p; R=$Y/runs/A3; T=$Y/tree; L=$R/a3.log
I=[REDACTED]; K="[REDACTED]"
say(){ echo "$(date -u +%FT%T) $*" >> $L; }; send(){ for i in 1 2 3; do $I send $K --text "$1" >/dev/null && return 0; sleep 30; done; }
source /home/user/work/env.sh >/dev/null 2>&1; source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
G=0,1,2,3,4,5,6,7; TS="taskset -c 0-27,49-111"
# 1. shared init (once)
if [ ! -s $R/A3/qat/state/state.json ]; then
  say "INIT start"; ( cd $T && CUDA_VISIBLE_DEVICES=$G $TS torchrun --nproc_per_node 8 --master_port 30910 rx7r.py $R/init/cfg.json > $R/init/stdout.log 2>&1 ) || { say "INIT FAIL"; send "v3 A3: INIT FAILED, see $R/init/stdout.log"; exit 1; }
  mkdir -p $R/A3/qat && cp -a $R/init/qat/state $R/A3/qat/state; say "INIT done: $(grep -hE 'init_opt|cosine_exact|scale init|lift_subnormal' $R/init/stdout.log | tr '\n' ' ')"
fi
# 2. train
if ! jq -e .done $R/A3/qat/result.json >/dev/null 2>&1; then
  fails=0; last=-1
  while true; do
    [ -f $R/A3/qat/STOP ] && { say "STOP file - exiting"; exit 0; }
    n=$(ls $R/A3/qat/stdout.[0-9]*.log 2>/dev/null | wc -l); say "LAUNCH attempt $n (state step $(jq -r .step $R/A3/qat/state/state.json))"
    ( cd $T && CUDA_VISIBLE_DEVICES=$G $TS torchrun --nproc_per_node 8 --master_port 30900 rx7r.py $R/cfg_A3.json > $R/A3/qat/stdout.$n.log 2>&1 ); rc=$?
    jq -e .done $R/A3/qat/result.json >/dev/null 2>&1 && { say "QAT DONE"; break; }
    [ -f $R/A3/qat/STOP ] && { say "STOP honoured rc $rc"; exit 0; }
    step=$(jq -r .step $R/A3/qat/state/state.json 2>/dev/null || echo -1)
    if [ "$step" = "$last" ]; then fails=$((fails+1)); else fails=1; last=$step; fi
    say "EXIT rc $rc at step $step (fail $fails/3)"
    [ $fails -ge 3 ] && { say "GIVING UP"; send "v3 A3: GAVE UP after 3 failures at step $step. Diagnosing."; exit 1; }
    sleep 30
  done
fi
# 3. finish on a copy (driver stages fp32 + export at W2048), then W1536
B=$Y/runs/br_A_s1_e1344_narrow_a3
if ! grep -q "RESULT" $B/branch.log 2>/dev/null; then
  mkdir -p $B/qat $B/fisher; cp $R/A3/qat/model.tch $B/qat/model.tch; cp $R/fisher/alloc_1.000.npz $B/fisher/
  ( cd $Y && Y=$Y BC=$Y/cfg_yolo2_A_s1.json SNAP=$Y/runs/yolo2_A_s1/snap_e1344 NARROW=1 VAR=_narrow_a3 PLAN=2000 STAGES="fp32 export" bash $Y/v6_y2_branch_prod.sh A 1 1344 $G 30920 ) >> $L 2>&1
fi
O=$B/w1536; mkdir -p $O/ev; EXPAR=/home/user/work/expar; TFT=/home/user/work/tft; GATE=/home/user/research/v3/gate
if [ ! -s $O/c4/final_v4.c4 ]; then
  jq '.overrides.window_size=1536' $B/fp32/cfg.json > $O/cfg_w1536.json
  ( cd $EXPAR && CUDA_VISIBLE_DEVICES=0 python export_v3_ar.py $O/cfg_w1536.json $B/fp32/model.tch $O/raw > $O/export.log 2>&1 )
  ( cd $TFT && TREE=$TFT PYTHON=python bash $EXPAR/export4/export.sh $O/raw/weights-poc8m_w2048.bin $O/c4 > $O/c4.log 2>&1 ) || true
  [ -f $O/c4/weights-poc8m_w2048_pruned.c3 ] || cp $O/c4/weights-poc8m_w2048_v3bf16.c3 $O/c4/weights-poc8m_w2048_pruned.c3
  ( cd $TFT && WC_INT5M=1 PYTHONPATH=$TFT python $EXPAR/export4/convert_container_v4.py $O/c4/weights-poc8m_w2048_pruned.c3 $O/c4/final_v4.c4 >> $O/c4.log 2>&1 )
fi
GW=$($GATE/gate $O/c4/final_v4.c4 > $O/digest.txt 2>/dev/null && $TFT/.venv/bin/python $GATE/gate.py $O/c4/final_v4.c4 $O/digest.txt 2>&1 | tail -1)
jq --arg o $O/ev --arg i $B/fp32/model.tch '.name="A3_ev_W1536" | .out=$o | .init=$i | .init_strict=true | .epochs=0 | .eval=true | .overrides.window_size=1536' $B/fp32/cfg.json > $O/ev/cfg.json
( cd $T && CUDA_VISIBLE_DEVICES=$G torchrun --nproc_per_node 8 --master_port 30930 rx3a.py $O/ev/cfg.json > $O/ev/stdout.log 2>&1 )
H2=$(grep EVAL $B/fp32/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+' | grep -oE '[0-9.]+$')
H15=$(grep EVAL $O/ev/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+' | grep -oE '[0-9.]+$'); C=$(stat -c %s $O/c4/final_v4.c4)
EV=$(grep -h '^EVAL' $R/A3/qat/stdout.*.log | grep -oE '"bits_per_tok": [0-9.]+, "n_scored": [0-9]+, "epoch": [0-9]+' | sed -E 's/"bits_per_tok": ([0-9.]+), "n_scored": [0-9]+, "epoch": ([0-9]+)/q\2 \1/' | tr '\n' ' ')
DS=$(/home/user/work/venv/bin/python -c "h=$H15;c=$C;ce=822000*100*(h-1.2915)/1.2915;ds=ce+2*(c-3716667)+12013;dA=822000*100*(h-1.2740351)/1.2915+2*(c-3649198);print('S proxy %d B, dS vs shipped %+.0f B = %+.3f%% S | vs Arm A: %+.0f B (0.8x %+.0f / 1.2x %+.0f)'%(95914051+ds,ds,100*ds/95914051,dA,0.8*822000*100*(h-1.2740351)/1.2915+2*(c-3649198),1.2*822000*100*(h-1.2740351)/1.2915+2*(c-3649198)))")
say "RESULT W2048 $H2 W1536 $H15 C4 $C sha $(sha256sum $O/c4/final_v4.c4 | cut -c1-16) gate $GW | $DS"
send "v3 E2 ARM A3 DONE (parent canonical E1344, 192 QAT ep, cosine 1->0.025x, C1 all epochs, rx7r 9696be34). QAT-mode evals: $EV. FINISHED (fp32 1 ep + strict): W2048 $H2, W1536 $H15. C4 W1536 $C B (sha $(sha256sum $O/c4/final_v4.c4 | cut -c1-16)), gate: $GW. $DS. [MEASURED CE/C4, PROXY S]"
say "A3 ALL DONE"
