#!/bin/bash

# single inter-subject 
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/single-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/4-fold-inter-subject-$i \
#        --batch-size 800 --decay-all --dataset biopatrec-db2 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --dropout 0.65 \
#        --num-semg-row 1 --num-semg-col 8 \
#        --gpu 1 \
#        crossval --crossval-type 4-fold-inter-subject --fold $i
#done

## sinlge inter-subject adabn
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 8 \
#    --gpu 0 \
#    --adabn \
#    --minibatch \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

#for i in $(seq 0 16); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#    --params .cache/single-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 800 --decay-all --dataset biopatrec-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusio-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 8 \
#    --gpu 0 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done

# multi inter-subject 
dropouts=(0.3 0.5 0.7);
wds=(0.00001 0.0001 0.001);

for j in ${dropouts[@]}; do
    for l in ${wds[@]}; do
        for i in $(seq 0 3); do
            python -m sigr.train exp --log log --snapshot model \
                --root .cache/multi-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/dropout$j-wd$l/4-fold-inter-subject-$i \
                --batch-size 800 --decay-all --dataset biopatrec-db2 \
                --num-filter 64 \
                --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
                --balance-gesture 1 \
                --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
                --fusion-type 'multi_no_imu' \
                --window 1 \
                --dropout $j \
                --wd $l \
                --num-semg-row 1 --num-semg-col 8 \
                --gpu 0 \
                crossval --crossval-type 4-fold-inter-subject --fold $i
        done
    done
done

## multi inter-subject adabn
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 8 \
#    --gpu 0 \
#    --adabn \
#    --minibatch \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
#for i in $(seq 0 16); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#    --params .cache/multi-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 800 --decay-all --dataset biopatrec-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusio-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 8 \
#    --gpu 0 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done
