# single view intra
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db4 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --lr 0.1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --gpu 2 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

#for i in $(seq 0 7); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/single-biopatrec-db4-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/single-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 400 --decay-all --dataset biopatrec-db4 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 16 \
#        --dropout 0.5 \
#        --gpu 2 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done

## multi view intra
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/v1.1-multi-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db4 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --gpu 1 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
##
#for i in $(seq 0 7); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/v1.1-multi-biopatrec-db4-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/v1.1-multi-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 400 --decay-all --dataset biopatrec-db4 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#        --fusion-type 'multi_no_imu' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 16 \
#        --dropout 0.65 \
#        --gpu 1 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done


#single view inter adabn 49-gpu-1
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db4 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --adabn \
#    --gpu 1 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/single-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/true-4-fold-inter-subject-fold-$i \
#        --params .cache/single-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 400 --decay-all --dataset biopatrec-db4 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 16 \
#        --dropout 0.65 \
#        --gpu 1 \
#        crossval --crossval-type 4-fold-inter-subject --fold $i
#done

# multi view inter with adabn 
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db4 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --adabn \
#    --gpu 1 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/multi-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/true-4-fold-inter-subject-fold-$i \
        --params .cache/multi-adabn-inter-subject-biopatrec-db4-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
        --batch-size 400 --decay-all --dataset biopatrec-db4 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
        --fusion-type 'multi_no_imu' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 16 \
        --dropout 0.65 \
        --gpu 1 \
        crossval --crossval-type 4-fold-inter-subject --fold $i
done
