#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0"
gpu_num=1
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
simu_2skpr_feats_dir="/nfs/wangjiaming.wjm/EEND_ARK_DATA/data/simu_data" #feature output dictionary
simu_train_dataset=train
simu_valid_dataset=dev

exp_dir="."
lang=zh
token_type=char
type=sound
scp=wav.scp
speed_perturb="0.9 1.0 1.1"
stage=3
stop_stage=5

# feature configuration
nj=64

# exp tag
tag="exp1"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

diar_config=conf/train_diar_eend_ola_2spkr.yaml
model_dir="baseline_$(basename "${diar_config}" .yaml)_${tag}"

# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

# ASR Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: ASR Training"
    mkdir -p ${exp_dir}/exp/${model_dir}
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi 
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name diar \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --data_dir ${simu_2skpr_feats_dir} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats_2spkr.scp,speaker_labels_2spkr.json" \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi