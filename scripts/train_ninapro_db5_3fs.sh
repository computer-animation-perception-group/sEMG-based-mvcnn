## 49 gpu 0

# single inter-subject adabn
python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --gpu 0 \
    --adabn \
    --minibatch \
    --preprocess 'ninapro-lowpass' \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0
#
## train
for i in $(seq 0 9); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/one-fold-intra-subject-$i \
    --params .cache/single-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject/model-0028.params \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --dropout 0.65 \
    --gpu 0 \
    crossval --crossval-type one-fold-intra-subject --fold $i
done

for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/4-fold-inter-subject-$i \
    --params .cache/single-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject/model-0028.params \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --dropout 0.65 \
    --gpu 0 \
    crossval --crossval-type 4-fold-inter-subject --fold $i
done

# single inter-subject no-adabn
for i in $(seq 0 3); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/4-fold-inter-subject-$i \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --dropout 0.65 \
    --gpu 0 \
    crossval --crossval-type 4-fold-inter-subject --fold $i
done

###########################################################

# 48 gpu 2
# multi inter-subject adabn
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject \
#    --batch-size 1000 --decay-all --dataset ninapro-db5 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --gpu 2 \
#    --adabn \
#    --minibatch \
#    --preprocess 'ninapro-lowpass' \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0
##
### train
#for i in $(seq 0 9); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/one-fold-intra-subject-$i \
#    --params .cache/multi-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db5 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --dropout 0.65 \
#    --gpu 2 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done
#
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/4-fold-inter-subject-$i \
#    --params .cache/multi-adabn-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db5 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --dropout 0.65 \
#    --gpu 2 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done
#
## multi inter-subject no-adabn
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-inter-subject-ninapro-db5-featuresigimg-win-40-stride-20/4-fold-inter-subject-$i \
#    --batch-size 1000 --decay-all --dataset ninapro-db5 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 16 \
#    --dropout 0.65 \
#    --gpu 2 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done
