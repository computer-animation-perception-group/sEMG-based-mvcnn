
python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-no-imu-one-fold-intra-subject-flip-ninapro-db1-win-20-stride-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db1 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 10 \
    --gpu 0 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

for i in $(seq 0 26); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/single-no-imu-one-fold-intra-subject-flip-ninapro-db1-win-20-stride-1/one-fold-intra-subject-$i \
    --params .cache/single-no-imu-one-fold-intra-subject-flip-ninapro-db1-win-20-stride-1/universal-one-fold-intra-subject/model-0028.params \
    --batch-size 1000 --decay-all --dataset ninapro-db1 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 10 \
    --gpu 0 \
    crossval --crossval-type one-fold-intra-subject --fold $i
done

