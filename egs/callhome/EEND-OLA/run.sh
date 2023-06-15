#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="7"
gpu_num=1
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
simu_feats_dir="/nfs/wangjiaming.wjm/EEND_ARK_DATA/data/simu_data"
simu_feats_dir_chunk2000="/nfs/wangjiaming.wjm/EEND_ARK_DATA/data/simu_data_chunk2000"
simu_train_dataset=train
simu_valid_dataset=dev

# model average
simu_average_2spkr_start=91
simu_average_2spkr_end=100
simu_average_allspkr_start=16
simu_average_allspkr_end=25

exp_dir="."
lang=zh
token_type=char
input_size=345
type=sound
scp=wav.scp
speed_perturb="0.9 1.0 1.1"
stage=7
stop_stage=8

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

simu_2spkr_diar_config=conf/train_diar_eend_ola_2spkr.yaml
simu_allspkr_diar_config=conf/train_diar_eend_ola_allspkr.yaml
simu_allspkr_chunk2000_diar_config=conf/train_diar_eend_ola_allspkr.yaml
simu_2spkr_model_dir="baseline_$(basename "${simu_2spkr_diar_config}" .yaml)_${tag}"
simu_allspkr_model_dir="baseline_$(basename "${simu_allspkr_diar_config}" .yaml)_${tag}"
simu_allspkr_chunk2000_model_dir="baseline_$(basename "${simu_allspkr_chunk2000_diar_config}" .yaml)_${tag}"

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
    mkdir -p ${exp_dir}/exp/${simu_2spkr_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_2spkr_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_2spkr_model_dir}/ddp_init
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
                --input_size $input_size \
                --data_dir ${simu_feats_dir} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats_2spkr.scp,speaker_labels_2spkr.json" \
                --resume true \
                --output_dir ${exp_dir}/exp/${simu_2spkr_model_dir} \
                --config $simu_2spkr_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_2spkr_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Average model parameters
simu_2spkr_ave_id=avg${simu_average_2spkr_start}-${simu_average_2spkr_end}
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "averaging model parameters into $simu_2spkr_model_dir/$simu_2spkr_ave_id.pb"
    models=`eval echo ${exp_dir}/exp/${simu_2spkr_model_dir}/{$simu_average_2spkr_start..$simu_average_2spkr_end}epoch.pb`
    python local/model_averaging.py ${exp_dir}/exp/${simu_2spkr_model_dir}/$simu_2spkr_ave_id.pb $models
fi

# ASR Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: ASR Training"
    mkdir -p ${exp_dir}/exp/${simu_allspkr_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_allspkr_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_allspkr_model_dir}/ddp_init
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
                --input_size $input_size \
                --data_dir ${simu_feats_dir} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats.scp,speaker_labels.json" \
                --resume true \
                --init_param ${exp_dir}/exp/${simu_2spkr_model_dir}/$simu_2spkr_ave_id.pb \
                --output_dir ${exp_dir}/exp/${simu_allspkr_model_dir} \
                --config $simu_allspkr_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_allspkr_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Average model parameters
simu_allspkr_ave_id=avg${simu_average_allspkr_start}-${simu_average_allspkr_end}
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "averaging model parameters into $simu_allspkr_model_dir/$simu_allspkr_ave_id.pb"
    models=`eval echo ${exp_dir}/exp/${simu_allspkr_model_dir}/{$simu_average_allspkr_start..$simu_average_allspkr_end}epoch.pb`
    python local/model_averaging.py ${exp_dir}/exp/${simu_allspkr_model_dir}/$simu_allspkr_ave_id.pb $models
fi

# ASR Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: ASR Training"
    mkdir -p ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/ddp_init
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
                --input_size $input_size \
                --data_dir ${simu_feats_dir_chunk2000} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats.scp,speaker_labels.json" \
                --resume true \
                --init_param ${exp_dir}/exp/${simu_allspkr_model_dir}/$simu_allspkr_ave_id.pb \
                --output_dir ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir} \
                --config $simu_allspkr_chunk2000_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi