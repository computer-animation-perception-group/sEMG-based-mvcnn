## pretrain 54 gpu1
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
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
#for i in $(seq 0 20); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
#    --params .cache/3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
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

# 54 gpu-0
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --gpu 0 \
#    crossval --crossval-type universal-one-fold-intra-subject --fold 0

# train
#for i in $(seq 0 20); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
#    --params .cache/multi-3FS-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 0 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done

#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 1 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done
#
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 1 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done

#######################################
# 50 gpu-0

# pretrain 
python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db7 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 12 \
		--adabn \
		--minibatch \
    --gpu 0 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

python -m sigr.train exp --log log --snapshot model \
    --root .cache/multi-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db7 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
    --fusion-type 'multi_no_imu' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 12 \
		--adabn \
		--minibatch \
    --gpu 0 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

## single view
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
#		--params .cache/single-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 0 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done
#
#for i in $(seq 0 20); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/single-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
#    --params .cache/single-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 0 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done
#
## multi view
#for i in $(seq 0 3); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/4-fold-inter-subject-$i \
#    --params .cache/multi-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 1 \
#    crossval --crossval-type 4-fold-inter-subject --fold $i
#done
#
#for i in $(seq 0 20); do
#    python -m sigr.train exp --log log --snapshot model \
#    --root .cache/multi-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/one-fold-intra-subject-$i \
#    --params .cache/multi-adabn-inter-subject-ninapro-db7-downsample20-featuresigimg-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
#    --batch-size 1000 --decay-all --dataset ninapro-db7 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1 --num-semg-col 12 \
#    --dropout 0.65 \
#    --gpu 0 \
#    crossval --crossval-type one-fold-intra-subject --fold $i
#done
