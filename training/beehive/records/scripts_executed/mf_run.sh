#!/usr/bin/env bash
# mf_run.sh -- v6 matched-finish experiments (admin 2084): arm A (full C1) then arm B (late C1), each 64 QAT epochs from the
# SAME shared post-init state (runs/MF/init/qat/state, already copied to A/qat/state and B/qat/state), then the production
# finish on a copy of each q64 state: fp32 1 ep @3e-6 + strict W2048 + C4 + gate (y2_branch.sh), W1536 export 42a5f84d +
# gate + strict W1536 (fin_w1536.sh). Posts a report per arm; paired packet afterwards (mf_packet.py).
set -u
Y=/home/user/work/y2p; M=$Y/runs/MF; T=$Y/tree
I=[REDACTED]; K="[REDACTED]"
send(){ for i in 1 2 3; do $I send $K --text "$1" >/dev/null && return 0; sleep 30; done; }
source /home/user/work/env.sh >/dev/null 2>&1; source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
for a in ${ARMS:-A B}; do
  R=$M/$a; L=$R/launcher.log
  if ! jq -e .done $R/qat/result.json >/dev/null 2>&1; then
    echo "$(date -u +%FT%T) ARM $a START" >> $L
    (setsid nohup bash $Y/mf/mf_backup.sh $a > $R/backup.out 2>&1 < /dev/null &)
    fails=0; last=-1
    while true; do
      n=$(ls $R/qat/stdout.[0-9]*.log 2>/dev/null | wc -l)
      echo "$(date -u +%FT%T) LAUNCH $a attempt $n (state step $(jq -r .step $R/qat/state/state.json 2>/dev/null))" >> $L
      ( cd $T && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 30700 rx7r.py $R/cfg.json > $R/qat/stdout.$n.log 2>&1 ); rc=$?
      jq -e .done $R/qat/result.json >/dev/null 2>&1 && { echo "$(date -u +%FT%T) ARM $a QAT DONE" >> $L; break; }
      [ -f $R/qat/STOP ] && { echo "$(date -u +%FT%T) STOP honoured (rc $rc)" >> $L; exit 0; }
      step=$(jq -r .step $R/qat/state/state.json 2>/dev/null || echo -1)
      if [ "$step" = "$last" ]; then fails=$((fails+1)); else fails=1; last=$step; fi
      echo "$(date -u +%FT%T) EXIT rc $rc at step $step (fail $fails/3)" >> $L
      [ $fails -ge 3 ] && { echo "$(date -u +%FT%T) GIVING UP arm $a" >> $L; send "v6 MF arm $a: GAVE UP after 3 failures at step $step. Diagnosing."; exit 1; }
      sleep 30
    done
    P=$(cat $R/backup.pid 2>/dev/null); [ -n "$P" ] && kill $P 2>/dev/null
    bash $Y/mf/mf_backup.sh $a once > $R/backup_final.out 2>&1
  fi
  # finish on a copy
  BR=$Y/runs/br_Y2_s1_e930_mf$a
  if [ ! -s $BR/w1536/fin.log ] || ! grep -q "STRICT W1536" $BR/w1536/fin.log; then
    mkdir -p $BR/qat $BR/snap; cp -a $R/qat/state/. $BR/snap/; cp $R/qat/model.tch $BR/qat/model.tch; cp -a $M/fisher $BR/fisher
    ( cd $Y && BC=$Y/cfg/Y2_s1_bf16.json SNAP=$BR/snap NARROW=1 VAR=_mf$a STAGES="fp32 export" bash y2_branch.sh Y2 1 930 0,1,2,3,4,5,6,7 30710 )
    EXPAR=/home/user/work/expar42 bash $Y/fin_w1536.sh $BR $BR/w1536 0,1,2,3,4,5,6,7 30720
  fi
  H2=$(grep -oE 'STRICT DEPLOYED "bits_per_tok": [0-9.]+' $BR/branch.log | tail -1 | grep -oE '[0-9.]+$')
  H15=$(grep -oE 'STRICT W1536 "bits_per_tok": [0-9.]+' $BR/w1536/fin.log | tail -1 | grep -oE '[0-9.]+$')
  C=$(stat -c %s $BR/w1536/c4/final_v4.c4 2>/dev/null); G=$(grep "W1536 C4" $BR/w1536/fin.log | tail -1 | grep -oE 'gate: .*')
  EV=$(grep -h '^EVAL' $R/qat/stdout.*.log | grep -oE '"bits_per_tok": [0-9.]+, "n_scored": [0-9]+, "epoch": [0-9]+' | sed -E 's/"bits_per_tok": ([0-9.]+), "n_scored": [0-9]+, "epoch": ([0-9]+)/q\2 \1/' | tr '\n' ' ')
  DS=$(python3 -c "h=$H15;c=$C;ds=822000*100*(h/1.2915-1)+2*(c-3716667)+12013;print('S proxy %d B, dS vs shipped %+.0f B = %+.3f%% S (1%% line -959141; Href 1.2915, Cref 3716667, other +12013 planning)'%(95914051+ds,ds,100*ds/95914051))")
  send "v6 MF ARM $a (parent v3 E930, 64 QAT ep, cosine 1->0.025x, C1 $( [ $a = A ] && echo 'all 64 ep' || echo 'last 21 ep (from update 25435)')) DONE. QAT-mode evals: $EV. FINISHED (fp32 1 ep + strict): W2048 $H2, W1536 $H15. C4 W1536 $C B, $G. $DS. [status: MEASURED CE/C4, PROXY S]"
  echo "$a $H2 $H15 $C" >> $M/results.txt
done
python3 $Y/mf/mf_packet.py > $M/packet.out 2>&1; echo "$(date -u +%FT%T) ALL DONE" >> $M/mf_run.log
