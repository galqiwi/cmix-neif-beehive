#!/usr/bin/env bash
# reproduce.sh: train the transformer from scratch and write the container this entry ships.
#
#   WORK=<empty dir> PKG=<package D dir> TFT_TREE=<assembled training tree> \
#     bash training/beehive/scripts/reproduce.sh [stage ...]
#
# Stages (default: all, in this order; each one is skipped when its output exists, so a stopped run is resumed by
# running the same command again):
#   phase1  bf16 from random init (seed 1), 2000-epoch plan, stopped at epoch 930        rx7q.py   8 GPUs
#   fisher  Fisher 2/3/4-bit row allocation at budget 1.00 ("NARROW") on the E930 model  fisher_alloc_a.py  1 GPU
#   init    QAT init: activation scales (256 batches), optimizer moments carried, C1   rx7r.py init_only
#   qat     64 QAT epochs, cosine LR 1 -> 0.025x, C1 rate term on every update           rx7r.py   8 GPUs
#   fp32    1 fp32 epoch at lr 3e-6 + strict deployed eval at window 2048                rx3a.py   8 GPUs
#   export  export at attention window 1536, container v4, C++/python gate             ../export
#   eval    strict deployed eval at window 1536                                          rx3a.py
#   install copy the container into the source tree and stamp its fingerprint           (then: bash build.sh)
#
# Environment: training/README.md section 1 (Python 3.11, torch 2.12.1 CUDA, fla 0.5.1, triton 3.7.1, numpy 2.4.6),
# TFT_TREE = the training tree assembled as in training/README.md section 1 (unchanged upstream + cmix-neif overlay),
# PKG = package D of training/README.md stages (a)-(c). GPUS (default 0..7): the shipped run used 8 GPUs, micro batch
# 131072 tokens, accumulation 1; other GPU counts change the reduction order and therefore the bits.
# The expected checksums of every intermediate are in ../records/SHA256_ARTEFACTS.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); ROOT=$(cd "$HERE/../.." && pwd); T=$HERE/tree; C=$HERE/cfg; E=$HERE/export
: "${WORK:?set WORK}"; : "${PKG:?set PKG}"; : "${TFT_TREE:?set TFT_TREE}"; export TFT_TREE PKG
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; NG=$(echo $GPUS | tr ',' '\n' | wc -l); G0=${GPUS%%,*}; PORT=${PORT:-30700}
PY=${PYTHON:-python}; J=${JQ:-jq}; CXX=${CXX:-clang++}
STAGES=${*:-phase1 fisher init qat fp32 export eval install}
source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
mkdir -p $WORK; L=$WORK/reproduce.log
say(){ echo "$(date -u +%FT%T) $*" | tee -a $L; }
P1=$WORK/phase1; SNAP=$P1/snap_e930; FI=$WORK/fisher; ALLOC=$FI/alloc_1.000.npz; IN=$WORK/init; Q=$WORK/qat
F=$WORK/fp32; X=$WORK/w1536

# resumable torchrun loop: <script> <cfg> <out dir with state/> <done test>
run_loop(){ local s=$1 cfg=$2 out=$3 fails=0 last=-1 n step
  while true; do
    n=$(ls $out/stdout.[0-9]*.log 2>/dev/null | wc -l || true)
    ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $PORT $s $cfg > $out/stdout.$n.log 2>&1 ) || true
    eval "$4" && return 0
    step=$($J -r .step $out/state/state.json 2>/dev/null || echo -1)
    if [ "$step" = "$last" ]; then fails=$((fails+1)); else fails=1; last=$step; fi
    [ $fails -ge 3 ] && { say "FAIL $s: 3 failures at step $step (see $out/stdout.*.log)"; exit 1; }
    sleep 30
  done; }

st_phase1(){ [ -s $SNAP/model.tch ] && return 0; mkdir -p $P1
  $J --arg o $P1 '.out=$o' $C/phase1_bf16.json > $P1/cfg.json
  run_loop rx7q.py $P1/cfg.json $P1 '[ "$($J -r .epoch $P1/state/state.json 2>/dev/null || echo 0)" -ge 930 ]'
  mkdir -p $SNAP; cp $P1/state/model.tch $P1/state/opt.pt $P1/state/state.json $SNAP/
  say "phase1 E930: model $(sha256sum $SNAP/model.tch | cut -c1-16)"; }
st_fisher(){ [ -s $ALLOC ] && return 0; mkdir -p $FI
  $J --arg o $FI --arg i $SNAP/model.tch '.name="beehive_fisher_e930" | .out=$o | .init=$i' $C/fisher.json > $FI/cfg.json
  ( cd $T && CUDA_VISIBLE_DEVICES=$G0 torchrun --nproc_per_node 1 --master_port $PORT fisher_alloc_a.py $FI/cfg.json > $FI/stdout.log 2>&1 )
  say "fisher: $(grep -E '^budget' $FI/stdout.log | tail -1) alloc $(sha256sum $ALLOC | cut -c1-16)"; }
qcfg(){ $J --arg o "$1" --arg n "$2" --arg i $SNAP/model.tch --arg io $SNAP/opt.pt --arg m $ALLOC \
          '.name=$n | .out=$o | .init=$i | .init_opt=$io | .mixed=$m' $C/qat.json; }
st_init(){ [ -s $IN/qat/state/state.json ] && return 0; mkdir -p $IN/qat
  qcfg $IN/qat beehive_init_e930 | $J '.init_only=true' > $IN/cfg.json
  ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $PORT rx7r.py $IN/cfg.json > $IN/stdout.log 2>&1 )
  say "init: model $(sha256sum $IN/qat/state/model.tch | cut -c1-16)"; }
st_qat(){ $J -e .done $Q/result.json >/dev/null 2>&1 && return 0; mkdir -p $Q
  [ -s $Q/state/state.json ] || cp -a $IN/qat/state $Q/state
  qcfg $Q beehive_qat_e930_fullC1 > $WORK/qat_cfg.json
  run_loop rx7r.py $WORK/qat_cfg.json $Q '$J -e .done $Q/result.json >/dev/null 2>&1'
  say "qat: $(grep -h '^EVAL' $Q/stdout.*.log | grep -oE '"bits_per_tok": [0-9.]+' | tr '\n' ' ') model $(sha256sum $Q/model.tch | cut -c1-16)"; }
st_fp32(){ [ -s $F/model.tch ] && return 0; mkdir -p $F
  $J --arg o $F --arg i $Q/model.tch --arg m $ALLOC '.name="beehive_fp32" | .out=$o | .init=$i | .mixed=$m' $C/fp32.json > $F/cfg.json
  ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $PORT rx3a.py $F/cfg.json > $F/stdout.log 2>&1 )
  say "fp32: STRICT W2048 $(grep EVAL $F/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+') model $(sha256sum $F/model.tch | cut -c1-16)"; }
st_export(){ mkdir -p $X
  if [ ! -s $X/c4/final_v4.c4 ]; then
    $J '.overrides.window_size=1536' $F/cfg.json > $X/cfg_w1536.json
    ( cd $E && CUDA_VISIBLE_DEVICES=$G0 $PY export_v3_ar.py $X/cfg_w1536.json $F/model.tch $X/raw > $X/export.log 2>&1 )
    ( cd $TFT_TREE && TREE=$TFT_TREE PYTHON=$PY bash $E/export4/export.sh $X/raw/weights-poc8m_w2048.bin $X/c4 > $X/c4.log 2>&1 ) || true
    [ -f $X/c4/weights-poc8m_w2048_pruned.c3 ] || cp $X/c4/weights-poc8m_w2048_v3bf16.c3 $X/c4/weights-poc8m_w2048_pruned.c3
    ( cd $TFT_TREE && WC_INT5M=1 PYTHONPATH=$TFT_TREE $PY $E/export4/convert_container_v4.py $X/c4/weights-poc8m_w2048_pruned.c3 $X/c4/final_v4.c4 >> $X/c4.log 2>&1 )
  fi
  [ -x $X/gate ] || $CXX -O2 -std=c++17 -mavx2 -mfma -I$ROOT/src/tf1 $E/gate/gate.cpp $ROOT/src/tf1/weights_io.cpp $ROOT/src/tf1/weights_io_compressed.cpp -o $X/gate
  $X/gate $X/c4/final_v4.c4 > $X/digest.txt; g=$($PY $E/gate/gate.py $X/c4/final_v4.c4 $X/digest.txt | tail -1)
  say "export W1536: C4 $(stat -c %s $X/c4/final_v4.c4) B sha $(sha256sum $X/c4/final_v4.c4 | cut -c1-16) | gate: $g"; }
st_eval(){ mkdir -p $X/ev
  $J --arg o $X/ev --arg i $F/model.tch '.name="beehive_eval_W1536" | .out=$o | .init=$i | .init_strict=true | .epochs=0 | .eval=true | .overrides.window_size=1536' $F/cfg.json > $X/ev/cfg.json
  ( cd $T && CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NG --master_port $PORT rx3a.py $X/ev/cfg.json > $X/ev/stdout.log 2>&1 )
  say "eval: STRICT W1536 $(grep EVAL $X/ev/stdout.log | tail -1 | grep -oE '"bits_per_tok": [0-9.]+')"; }
st_install(){ cp -f $X/c4/final_v4.c4 $ROOT/weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4
  FP=$($PY - $X/c4/final_v4.c4 <<'PYEOF'
import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], 'rb').read():
    h ^= b; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print("0x%016xULL" % h)
PYEOF
)
  sed -i "s|  return 0x[0-9a-f]*ULL;|  return $FP;|" $ROOT/src/transformer.h
  say "install: container copied, fingerprint $FP stamped in src/transformer.h (shipped: 0xf6bec91b6fe34355ULL). Next: bash build.sh"; }

for s in $STAGES; do say "stage $s"; st_$s; done
