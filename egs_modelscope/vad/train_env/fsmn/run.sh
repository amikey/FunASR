#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0,1,2,3"
gpu_num=1
count=1
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
train_cmd=utils/run.pl
infer_cmd=utils/run.pl

# general configuration
feats_dir="DATA" #feature output dictionary
exp_dir="."
lang=zh
dumpdir=dump/fbank
feats_type=fbank
token_type=char
#dataset_type=large
dataset_type=small
scp=feats.scp
type=kaldi_ark
stage=1
stop_stage=1

# feature configuration
feats_dim=80
sample_frequency=16000
nj=10
speed_perturb="0.9,1.0,1.1"

# data
tr_dir=
dev_tst_dir=

# exp tag
tag="exp1"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=fbank/train
valid_set=fbank/valid

vad_config=conf/train_vad_fsmn.yaml
model_dir="baseline_$(basename "${vad_config}" .yaml)"


# you can set gpu num for decoding here
gpuid_list=$CUDA_VISIBLE_DEVICES  # set gpus for decoding, e.g., gpuid_list=2,3, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

echo "stage 3: Training"
mkdir -p ${exp_dir}/exp/${model_dir}
mkdir -p ${exp_dir}/exp/${model_dir}/log
INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
if [ -f $INIT_FILE ];then
    rm -f $INIT_FILE
fi
gpu_id=5

feat_train_dir=${feats_dir}/${dumpdir}/train; mkdir -p ${feat_train_dir}
feat_valid_dir=${feats_dir}/${dumpdir}/valid; mkdir -p ${feat_valid_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # compute lfr and cmvn for fbank
    #utils/apply_lfr_and_cmvn.sh --cmd "${train_cmd}" --nj ${nj} --lfr_m 5 --lfr_n 1 \
        #${train_set} res/vad.mvn ${exp_dir}/exp/make_fbank/train ${feat_train_dir}
    #utils/apply_lfr_and_cmvn.sh --cmd "${train_cmd}" --nj ${nj} --lfr_m 5 --lfr_n 1 \
        #${valid_set} res/vad.mvn ${exp_dir}/exp/make_fbank/valid ${feat_valid_dir}
    #local/make_scp.sh --cmd "${train_cmd}" --nj ${nj} ${train_set} ${train_set}/log \
    #    ${feat_train_dir}
    local/make_scp.sh --cmd "${train_cmd}" --nj 1 ${valid_set} ${valid_set}/log \
        ${feat_valid_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
python /nfs/ailsa.zly/tfbase/espnet_work/FunASR_20230520/funasr/bin/vad_train.py \
            --gpu_id $gpu_id \
            --dataset_type $dataset_type \
            --train_data_path_and_name_and_type "${feat_train_dir}/feats.scp,speech,$type" \
            --train_data_path_and_name_and_type "${feat_train_dir}/target.scp,text,$type" \
            --train_shape_file "${feat_train_dir}/speech_shape" \
            --train_shape_file "${feat_train_dir}/text_shape" \
            --valid_data_path_and_name_and_type "${feat_valid_dir}/feats.scp,speech,$type" \
            --valid_data_path_and_name_and_type "${feat_valid_dir}/target.scp,text,$type" \
            --valid_shape_file "${feat_valid_dir}/speech_shape" \
            --valid_shape_file "${feat_valid_dir}/text_shape" \
            --resume true \
            --output_dir ${exp_dir}/exp/${model_dir} \
            --config $vad_config \
            --ngpu $gpu_num \
            --num_worker_count $count \
            --multiprocessing_distributed true
fi
