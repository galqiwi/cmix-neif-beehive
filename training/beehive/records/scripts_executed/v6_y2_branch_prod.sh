#!/usr/bin/env bash
# y2f_branch.sh ARM SEED SW GPUS PORT  -- YOLO-2 24h factorial QAT branch driver (v3, for v6 msg 1592; both nodes).
# v6 production generalisation: PLAN (default 2000), BC/SNAP overridable, QAT evals every QEVAL, milestones every QKEEP.
# From the phase-1 bf16 snapshot runs/<ARM>_s<SEED>/snap_e<SW> (rx7q 64-ep plan): Fisher 2-6-bit allocation (budget 1.00,
# class rate) on that state -> QAT (rx7q, init = branch model, init_opt = branch opt carried by name [A: init_opt_arch],
# cold scale init 256 steps, C1 lambda 2.7e-9 tau 0.25 from QAT step 0, fakequant + FastMuon, plan_tail {64, SW} so the
# LR trajectory is the SAME 64-ep plan, 64-SW epochs) -> fp32 1 ep @3e-6 + STRICT deployed eval (rx3 / rx3a for A) ->
# real id7 C4 via v7's export_v3_ar.py (one exporter for P and A) -> C++ decode-digest gate -> summary line with dS.
#   ARM P|A, SEED 1|2, SW 43 (fused) | 52 (late), GPUS e.g. 0,1,2,3, PORT base port (uses PORT..PORT+2).
# Env (defaults = both nodes' layout): Y=/home/user/work/y2f EXPAR=/home/user/work/expar TFT=/home/user/work/tft
#   GATE=/home/user/research/v3/gate  LAMBDA=2.7e-9  STAGES="fisher qat fp32 export"  (re-run any subset; each is idempotent)
set -u
ARM=$1; SEED=$2; SW=$3; GPUS=$4; PORT=$5
Y=${Y:-/home/user/work/y2p}; PLAN=${PLAN:-2000}; QEVAL=${QEVAL:-100}; QKEEP=${QKEEP:-100}; EXPAR=${EXPAR:-/home/user/work/expar}; TFT=${TFT:-/home/user/work/tft}; GATE=${GATE:-/home/user/research/v3/gate}
LAMBDA=${LAMBDA:-2.7e-9}; STAGES=${STAGES:-fisher qat fp32 export}; J=${JQ:-jq}
LEVELS=${LEVELS:-[1,3,7,15,31]}; BUDGET=${BUDGET:-1.0}; VAR=${VAR:-}   # narrow variant: LEVELS='[1,3,7]' VAR=_narrow (v3, v6 1792)
[ "${NARROW:-0}" = 1 ] && { LEVELS='[1,3,7]'; VAR=${VAR:-_narrow}; }   # v6 1793: NARROW=1 -> 2-4-bit-only allocation, tag suffix _narrow
T=$Y/tree; BC=${BC:-$Y/cfg/${ARM}_s${SEED}_bf16.json}; SNAP=${SNAP:-$Y/runs/${ARM}_s${SEED}/snap_e$SW}; TAG=${ARM}_s${SEED}_e$SW
TAG=$TAG$VAR; B=$Y/runs/br_$TAG; mkdir -p $B; L=$B/branch.log; G=$Y/runs/y2f_branches.log
say(){ echo "$(date -u +%FT%T) $TAG $*" | tee -a $L >> $G; }
NG=$(echo $GPUS | tr ',' '\n' | wc -l); G0=${GPUS%%,*}; ACC=$((8 / NG))
[ -s $SNAP/model.tch ] && [ -s $SNAP/opt.pt ] || { say "FAIL no snapshot $SNAP"; exit 1; }
HASARCH=$($J -r 'has("arch") and .arch != null' $BC)
source /home/user/work/env.sh >/dev/null 2>&1; source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
ALLOC=$B/fisher/$(printf "alloc_%.3f.npz" $BUDGET)
# common keys every stage inherits from the phase-1 cfg (geometry, arch, seed, ve_sentinel)
COMMON='{overrides, seed, ve_sentinel} + (if .arch then {arch} else {} end)'
st_fisher(){ mkdir -p $B/fisher
  $J --arg o $B/fisher --arg i $SNAP/model.tch --argjson lv "$LEVELS" --argjson bu $BUDGET "$COMMON"' + {name:"'$TAG'_fisher", out:$o, phase:"bf16", init:$i, batches:48, levels:$lv, budgets:[$bu], rate:"class"}' $BC > $B/fisher/cfg.json
  FA=fisher_alloc.py; [ "$HASARCH" = true ] && FA=fisher_alloc_a.py
  ( cd $T && CUDA_VISIBLE_DEVICES=$G0 torchrun --nproc_per_node 1 --master_port $PORT $FA $B/fisher/cfg.json > $B/fisher/stdout.log 2>&1 ) && [ -s $ALLOC ] || { say "FAIL fisher ($FA)"; return 1; }
  say "fisher ($FA): $(grep -E '^budget' $B/fisher/stdout.log | tail -1 | cut -c1-140) alloc $(sha256sum $ALLOC | cut -c1-12)"; }
st_qat(){ mkdir -p $B/qat
  $J --arg o $B/qat --arg i $SNAP/model.tch --arg io $SNAP/opt.pt --arg m $ALLOC --argjson ep $((PLAN - SW)) --argjson sw $SW --argjson lam $LAMBDA \
     --slurpfile c1 $Y/cfg/c1_ref_yolo15_c1.json \
     "$COMMON"' + {name:"'$TAG'_qat", out:$o, phase:"qat", epochs:$ep, plan_tail:{epochs:'"$PLAN"', from_epoch:$sw},
       init:$i, init_strict:false, init_opt:$io, init_opt_phase:"bf16", mixed:$m, scale_steps:256, fakequant:true,
       compile_qat:false, fastmuon:true, rate_term:($c1[0].rate_term + {lambda:$lam}), eval_every:'"$QEVAL"', state_every:1,
       keep_every:'"$QKEEP"'} + (if .arch then {init_opt_arch:.arch} else {} end)' $BC > $B/qat/cfg.json
  [ -n "${SMOKE_STOP:-}" ] && { $J ".stop_at = $SMOKE_STOP" $B/qat/cfg.json > $B/qat/cfg.tmp && mv $B/qat/cfg.tmp $B/qat/cfg.json; say "SMOKE: qat stop_at $SMOKE_STOP"; }
  fails=0; last=-1
  while true; do
    n=$(ls $B/qat/stdout.*.log 2>/dev/null | wc -l); say "qat launch attempt $n gpus $GPUS (accum $ACC)"
    ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $((PORT+1)) rx7q.py $B/qat/cfg.json > $B/qat/stdout.$n.log 2>&1 ); rc=$?
    $J -e .done $B/qat/result.json >/dev/null 2>&1 && break
    [ -n "${SMOKE_STOP:-}" ] && [ "$($J -r .epoch $B/qat/state/state.json 2>/dev/null || echo 0)" -ge "$SMOKE_STOP" ] 2>/dev/null && break
    step=$($J -r .step $B/qat/state/state.json 2>/dev/null || echo -1)
    if [ "$step" = "$last" ]; then fails=$((fails+1)); else fails=1; last=$step; fi
    [ $fails -ge 3 ] && { say "FAIL qat rc=$rc, 3 failures at step $step"; return 1; }
    sleep 30; done
  say "qat done: QAT-mode $(grep -h EVAL $B/qat/stdout.*.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+') | $(grep -h -m1 -oE 'init_opt: carried[^\"]*' $B/qat/stdout.*.log | head -1) | $(grep -h -m1 -oE 'QAT scale init: [0-9]+ activation-scale batches' $B/qat/stdout.*.log | head -1)"; }
st_fp32(){ mkdir -p $B/fp32
  Q=$B/qat/model.tch; [ -s $Q ] || Q=$B/qat/state/model.tch   # (smoke: a stopped QAT has only its state)
  $J --arg o $B/fp32 --arg q $Q --arg m $ALLOC "$COMMON"' + {name:"'$TAG'_fp32", out:$o, phase:"fp32", init:$q, init_strict:true, epochs:1, lr:3e-6, warmup_frac:0.0, final_mult:1e-8, eval:true, mixed:$m}' $BC > $B/fp32/cfg.json
  RX=rx3.py; [ "$HASARCH" = true ] && RX=rx3a.py
  ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $((PORT+2)) $RX $B/fp32/cfg.json > $B/fp32/stdout.log 2>&1 ) && [ -s $B/fp32/model.tch ] || { say "FAIL fp32 ($RX)"; return 1; }
  say "fp32 ($RX): STRICT DEPLOYED $(grep EVAL $B/fp32/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+')"; }
st_export(){ O=$B/c4; rm -rf $O; mkdir -p $O
  ( cd $EXPAR && CUDA_VISIBLE_DEVICES=$G0 python export_v3_ar.py $B/fp32/cfg.json $B/fp32/model.tch $O/raw > $O/export.log 2>&1 ) || { say "FAIL export (see $O/export.log)"; return 1; }
  ( cd $TFT && TREE=$TFT PYTHON=python bash $EXPAR/export4/export.sh $O/raw/weights-poc8m_w2048.bin $O/c4 > $O/c4.log 2>&1 ) || true
  [ -f $O/c4/weights-poc8m_w2048_pruned.c3 ] || cp $O/c4/weights-poc8m_w2048_v3bf16.c3 $O/c4/weights-poc8m_w2048_pruned.c3
  ( cd $TFT && WC_INT5M=1 PYTHONPATH=$TFT python $EXPAR/export4/convert_container_v4.py $O/c4/weights-poc8m_w2048_pruned.c3 $O/c4/final_v4.c4 >> $O/c4.log 2>&1 )
  [ -s $O/c4/final_v4.c4 ] || { say "FAIL container (see $O/c4.log)"; return 1; }
  g=$($GATE/gate $O/c4/final_v4.c4 > $O/digest.txt 2>/dev/null && $TFT/.venv/bin/python $GATE/gate.py $O/c4/final_v4.c4 $O/digest.txt 2>&1 | tail -1)
  CE=$(grep EVAL $B/fp32/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+' | grep -oE '[0-9.]+$'); C4=$(stat -c %s $O/c4/final_v4.c4)
  [ -n "$CE" ] || CE="float('nan')"; DS=$(python3 -c "k=822000.0;ce=$CE;c4=$C4;ds=k*100*(ce/1.2915-1)+2*(c4-3716667)+12013;print('dS vs shipped %+.0f B = %+.3f%% S (1%% line -959141)'%(ds,100*ds/95914051))" 2>/dev/null || /home/user/work/venv/bin/python -c "k=822000.0;ce=$CE;c4=$C4;ds=k*100*(ce/1.2915-1)+2*(c4-3716667)+12013;print('dS vs shipped %+.0f B = %+.3f%% S (1%% line -959141)'%(ds,100*ds/95914051))")
  say "RESULT strict $CE | C4 $C4 B sha $(sha256sum $O/c4/final_v4.c4 | cut -c1-16) | gate: $g | $DS (planning A-vs-P fees not included)"; }
[ -d $EXPAR ] && [ -x $GATE/gate ] || { echo "EXPAR ($EXPAR) or GATE ($GATE/gate) missing" | tee -a $L; exit 1; }
say "START stages [$STAGES] gpus $GPUS snap $(sha256sum $SNAP/model.tch | cut -c1-16) (epoch $($J -r .epoch $SNAP/state.json))"
for s in $STAGES; do st_$s || exit 1; done
say "BRANCH DONE"
