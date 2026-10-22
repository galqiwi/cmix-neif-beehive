#!/usr/bin/env bash
# mf2_prep.sh -- Arm A2: Arm A recipe from canonical snap_e1344 (parent mirrored at runs/v3_snap_e1344/ex). Fisher + shared init.
set -u
Y=/home/user/work/y2p; M=$Y/runs/MF2; P=$Y/runs/MF2_parent/snap_e1344; T=$Y/tree
source /home/user/work/env.sh >/dev/null 2>&1; source $T/exp.env
export TFT_MICRO_BATCH_TOKENS=131072 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
mkdir -p $M/fisher $M/init $M/A
jq --arg o $M/fisher --arg i $P/model.tch '{overrides, seed, ve_sentinel} + (if .arch then {arch} else {} end) + {name:"MF2_fisher_e1344", out:$o, phase:"bf16", init:$i, batches:48, levels:[1,3,7], budgets:[1.0], rate:"class"}' $Y/cfg/Y2_s1_bf16.json > $M/fisher/cfg.json
(cd $T && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 30800 fisher_alloc_a.py $M/fisher/cfg.json > $M/fisher/stdout.log 2>&1) || { echo FISHER FAIL; exit 1; }
jq --arg m $M/fisher/alloc_1.000.npz --arg i $P/model.tch --arg io $P/opt.pt '.mixed=$m | .init=$i | .init_opt=$io' $Y/runs/MF/cfg_common.json > $M/cfg_common.json
jq --arg o $M/init/qat '.name="MF2_init_e1344" | .out=$o | .init_only=true' $M/cfg_common.json > $M/init/cfg.json
jq --arg o $M/A/qat '.name="MF2_A2_fullC1_e1344" | .out=$o' $M/cfg_common.json > $M/A/cfg.json
(cd $T && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 30810 rx7r.py $M/init/cfg.json > $M/init/stdout.log 2>&1) || { echo INIT FAIL; exit 1; }
mkdir -p $M/A/qat && cp -a $M/init/qat/state $M/A/qat/state
grep -E "^budget" $M/fisher/stdout.log | tail -1; grep -E "init_opt|INIT_ONLY|cosine_exact" $M/init/stdout.log
echo PREP DONE
