# Ninapro DB5 with imu Single
python -m sigr.train exp --log log --snapshot model \
    --root .cache/ninapro-db5-intra-single-imu-20-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --blance-gesture 1 \
    --feature-name 'featuresigimg_imuactimg' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \

