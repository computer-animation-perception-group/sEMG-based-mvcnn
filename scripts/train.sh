ver=1.0.1.0
python -m sigr.train exp --log log --snapshot model \
    --root .cache/ninapro-db2-intra-single-imu-downsample20-20-1/ninapro-db2-semg-imu-universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db2 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_imuactimg' \
    --window 1 \
    --num-semg-row 1  --num-semg-col 12 \
    --gpu 0 \
    --dropout 0.65 \
    --imu-preprocess 'downsample-20' \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(0 39); do
    python -m sigr.train exp --log log --snapshot model \
        --root .cache/ninapro-db2-intra-single-imu-downsample20-20-1/ninapro-db2-semg-imu-one-fold-intra-subject-fold-$i \
        --params .cache/ninapro-db2-intra-single-imu-downsample20-20-1/ninapro2-db2-semg-imu-universal-one-fold-intra-subject/model-0028.params \
        --batch-size 1000 --decay-all --dataset ninapro-db2 \
        --num-filter 64 \
        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
        --balance-gesture 1 \
        --feature-name 'featuresigimg-imuactimg' \
        --window 1 \
        --num-semg-row 1 --num-semg-col 12 \
        --gpu 0 \
        --dropout 0.65 \
        --imu-preprocess 'downsample-20' \
        crossval --crossval-type one-fold-intra-subject --fold $i
done
        

#ver=1.0.0.0
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/ninapro-db2-semg-downsample20-single-win-20-stride-1-one-fold-intra-subject \
#    --batch-size 10 --decay-all --dataset ninapro-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28\
#    --balance-gesture 1 \
#    --feature-name 'featuresigimg_v2' \
#    --fusion-type 'single' \
#    --window 1 \
#    --num-semg-row 1  --num-semg-col 12 \
#    --gpu 0 \
#    --preprocess '(ninapro-lowpass, downsample-20)' \
#    crossval --crossval-type one-fold-intra-subject --fold 0

#ver=1.0.0.0
#python -m sigr.train exp --log log --snapshot model \
#    --root .cache/ninapro-db2-multi-downsample20-20-1/one-fold-intra-subject-0 \
#    --batch-size 10 --decay-all --dataset ninapro-db2 \
#    --num-filter 64 \
#    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28\
#    --balance-gesture 1 \
#    --feature-name 'rawsemg_feature_multisource_multistream_sigimgv2' \
#    --fusion-type 'multi_no_imu' \
#    --window 1 \
#    --num-semg-row 1  --num-semg-col 12 \
#    --gpu 0 \
#    --preprocess '(ninapro-lowpass, downsample-20)' \
#    crossval --crossval-type one-fold-intra-subject --fold 0
