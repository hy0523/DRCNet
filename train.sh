#!/bin/sh
PARTITION=Segmentation

GPU_ID=4,5
dataset=pascal # pascal coco
exp_name=split3

arch=BAM_final
net=resnet50 # vgg resnet50

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=2233 train.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log


#CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
#        --config=${config} \
#        --arch=${arch} \
#        2>&1 | tee ${result_dir}/train-$now.log
