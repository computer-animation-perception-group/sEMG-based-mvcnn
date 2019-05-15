#!/bin/bash

## pretrain 57 gpu 1
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/3FS-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
#    --batch-size 1000 --decay-all --dataset ninapro-db3 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --gpu 1 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
## train
#for i in $(seq 0 5); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/3FS-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
#    --params .cache/3FS-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db3 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 1 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done
# multi view intra
python -m sigr.train exp --log log --snapshot model \
    --root .cache/multi-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db3 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
    --fusion-type 'multi_no_imu' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 12 \
    --gpu 1 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

# train
for i in $(seq 0 5); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/multi-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
    --params .cache/multi-ninapro-db3-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
    --batch-size 1000 --decay-all --dataset ninapro-db3 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
    --fusion-type 'multi_no_imu' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 12 \
    --dropout 0.65 \
    --gpu 1 \
    crossval --crossval-type one-fold-intra-subject --fold $i
done
