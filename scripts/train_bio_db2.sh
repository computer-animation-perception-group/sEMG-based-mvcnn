## single view intra 57 gpu 0
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 8 \
#    --gpu 2 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

## pretrain
#for i in $(seq 0 16); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/pretrain/single-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/single-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 800 --decay-all --dataset biopatrec-db2 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 8 \
#        --gpu 2 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done
##
#for i in $(seq 0 16); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/afterpretrain-single-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/pretrain/single-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i/model-0028.params \
#        --batch-size 800 --decay-all --dataset biopatrec-db2 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 8 \
#        --gpu 2 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done
#
##for i in $(seq 0 3); do
##    python -m sigr.train exp --log log --snapshot model \
##        --root .cache/single-biopatrec-db2-3FSsigimg-win-300-stride-100/4-fold-inter-subject-$i \
##        --batch-size 800 --decay-all --dataset biopatrec-db2 \
##        --num-filter 64 \
##        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
##        --balance-gesture 1 \
##        --feature-name 'featuresigimg_v2' \
##        --fusion-type 'single' \
##        --window 1 \
##        --lr 0.1 \
##        --num-semg-row 1 --num-semg-col 8 \
##        --gpu 0 \
##        crossval --crossval-type 4-fold-inter-subject --fold $i
##done

#single view inter adabn
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
#    --adabn \
#    --gpu 2 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/single-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/true-4-fold-inter-subject-fold-$i \
        --params .cache/single-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
        --batch-size 800 --decay-all --dataset biopatrec-db2 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'featuresigimg_v2' \
        --fusion-type 'single' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 8 \
        --dropout 0.65 \
        --gpu 2 \
        crossval --crossval-type 4-fold-inter-subject --fold $i
done

# multi view inter with adabn 54-gpu-2
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
#    --adabn \
#    --gpu 2 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/multi-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/true-4-fold-inter-subject-fold-$i \
        --params .cache/multi-adabn-inter-subject-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
        --batch-size 800 --decay-all --dataset biopatrec-db2 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
        --fusion-type 'multi_no_imu' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 8 \
        --gpu 2 \
        crossval --crossval-type 4-fold-inter-subject --fold $i
done
