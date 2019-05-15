#!/bin/bash

## pretrain 57 gpu 3
# multi view inter
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/multi-inter-subject-ninapro-db2-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
#        --batch-size 1000 --decay-all --dataset ninapro-db2 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#        --fusion-type 'multi_no_imu' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 12 \
#        --gpu 1 \
#        crossval --crossval-type 4-fold-inter-subject --fold 0
#done
# single view inter
for i in $(seq 0 0); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/single-inter-subject-ninapro-db2-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
        --batch-size 1000 --decay-all --dataset ninapro-db2 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'featuresigimg_v2' \
        --fusion-type 'single' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 12 \
        --gpu 3 \
        crossval --crossval-type 4-fold-inter-subject --fold 0
done
