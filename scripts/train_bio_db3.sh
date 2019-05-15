# single view intra
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db3 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 4 \
#    --lr 0.25 \
#    --gpu 0 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
#for i in $(seq 0 7); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/single-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/single-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 300 --decay-all --dataset biopatrec-db3 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 4 \
#        --dropout 0.65 \
#        --gpu 0 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done
###
### multi view intra 54 gpu 0
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
#    --batch-size 800 --decay-all --dataset biopatrec-db3 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 4 \
#    --dropout 0.65 \
#    --lr 0.25 \
#    --gpu 0 \
#    --dropout 0.65 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
#for i in $(seq 0 7); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/multi-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/one-fold-intra-subject-fold-$i \
#        --params .cache/multi-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 300 --decay-all --dataset biopatrec-db3 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#        --fusion-type 'multi_no_imu' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 4 \
#        --gpu 0 \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done

# single view inter subject
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/single-inter-subject-dropout0.65-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/one-fold-inter-subject-fold-$i \
#        --batch-size 100 --decay-all --dataset biopatrec-db3 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 4 \
#        --gpu 0 \
#        --dropout 0.65 \
#        crossval --crossval-type 4-fold-inter-subject --fold $i
#done
#
## multi view inter subject
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/multi-inter-subject-biopatrec-db3-INTAN-3FSsigimg-win-300-stride-100/one-fold-inter-subject-fold-$i \
#        --batch-size 100 --decay-all --dataset biopatrec-db3 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#        --fusion-type 'multi_no_imu' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 4 \
#        --dropout 0.65 \
#        --gpu 0 \
#        crossval --crossval-type 4-fold-inter-subject --fold $i
#done

#single view inter adabn 49-gpu-1
python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
    --batch-size 800 --decay-all --dataset biopatrec-db3 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 4 \
    --adabn \
    --gpu 1 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/single-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/true-4-fold-inter-subject-fold-$i \
        --params .cache/single-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0028.params \
        --batch-size 400 --decay-all --dataset biopatrec-db3 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'featuresigimg_v2' \
        --fusion-type 'single' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 4 \
        --dropout 0.65 \
        --gpu 1 \
        crossval --crossval-type 4-fold-inter-subject --fold $i
done

# multi view inter with adabn 
python -m sigr.train exp --log log --snapshot model \
    --root .cache/multi-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/true-universal-one-fold-intra-subject \
    --batch-size 800 --decay-all --dataset biopatrec-db3 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
    --fusion-type 'multi_no_imu' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 4 \
    --dropout 0.65 \
    --adabn \
    --gpu 1 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/multi-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/true_v2-4-fold-inter-subject-fold-$i \
        --params .cache/multi-adabn-inter-subject-biopatrec-db3-3FSsigimg-win-300-stride-100/true-universal-one-fold-intra-subject/model-0028.params \
        --batch-size 400 --decay-all --dataset biopatrec-db3 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
        --fusion-type 'multi_no_imu' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 4 \
        --dropout 0.65 \
        --gpu 1 \
        crossval --crossval-type 4-fold-inter-subject --fold $i
done
