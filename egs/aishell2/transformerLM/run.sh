#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1"
gpu_num=2
count=1

# general configuration
feats_dir="../DATA" #feature output dictionary
exp_dir="."
lang=zh
token_type=char

# data
tr_dir=/nfs/wangjiaming.wjm/asr_data/aishell2/AISHELL-2/iOS/data
dev_tst_dir=/nfs/wangjiaming.wjm/asr_data/aishell2/AISHELL-DEV-TEST-SET

train_set=train
valid_set=dev_ios
test_sets="dev_ios test_ios"

lm_token_list=

## path to AISHELL2 trans
lm_test_text=

lm_config=conf/train_lm_transformer.yaml
tag=exp1
model_dir="baseline_$(basename "${lm_config}" .yaml)_${lang}_${token_type}_${tag}"

inference_lm=valid.loss.ave.pb       # Language model path for decoding.

stage=0
stop_stage=2

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # For training set
    local/prepare_data.sh ${tr_dir} ${feats_dir}/data/local/train ${feats_dir}/data/train || exit 1;
    # # For dev and test set
    for x in iOS; do
        local/prepare_data.sh ${dev_tst_dir}/${x}/dev ${feats_dir}/data/local/dev_${x,,} ${feats_dir}/data/dev_${x,,} || exit 1;
        local/prepare_data.sh ${dev_tst_dir}/${x}/test ${feats_dir}/data/local/test_${x,,} ${feats_dir}/data/test_${x,,} || exit 1;
    done
    # Normalize text to capital letters
    for x in train dev_ios test_ios; do
        mv ${feats_dir}/data/${x}/text ${feats_dir}/data/${x}/text.org
        paste -d " " <(cut -f 1 ${feats_dir}/data/${x}/text.org) <(cut -f 2- ${feats_dir}/data/${x}/text.org \
             | tr 'A-Z' 'a-z' | tr -d " ") \
            > ${feats_dir}/data/${x}/text
        utils/text2token.py -n 1 -s 1 ${feats_dir}/data/${x}/text > ${feats_dir}/data/${x}/text.org
        mv ${feats_dir}/data/${x}/text.org ${feats_dir}/data/${x}/text
    done
fi

token_list=${feats_dir}/data/${lang}_token_list/$token_type/tokens.txt
echo "dictionary: ${token_list}"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Dictionary Preparation"
    mkdir -p ${feats_dir}/data/${lang}_token_list/$token_type

    echo "make a dictionary"
    echo "<blank>" > ${token_list}
    echo "<s>" >> ${token_list}
    echo "</s>" >> ${token_list}
    utils/text2token.py -s 1 -n 1 --space "" ${feats_dir}/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0}' >> ${token_list}
    echo "<unk>" >> ${token_list}
fi


# LM Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: LM Training"
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
                --task_name lm \
                --gpu_id ${gpu_id} \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${lm_token_list}" \
                --data_dir ${feats_dir}/data \
                --train_set ${train_set} \
                --valid_set ${valid_set} \
                --data_file_names "text" \
                --filter_input false \
                --resume true \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config ${lm_config} \
                --ngpu ${gpu_num} \
                --num_worker_count ${count} \
                --multiprocessing_distributed true \
                --dist_init_method ${init_method} \
                --dist_world_size ${world_size} \
                --dist_rank ${rank} \
                --local_rank ${local_rank} 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } & 
      done
      wait
fi

# Testing Stage
gpu_num=1
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Calc perplexity: ${lm_test_text}"
    
    python ../../../funasr/bin/lm_inference.py \
        --output_dir "${lm_exp}/perplexity_test" \
        --ngpu "${gpu_num}" \
        --batch_size 1 \
        --train_config "${lm_exp}"/config.yaml \
        --model_file "${lm_exp}/${inference_lm}" \
        --data_path_and_name_and_type "${lm_test_text},text,text" \
        --num_workers 1 \
        --split_with_space false 
fi

