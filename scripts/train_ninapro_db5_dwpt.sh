## pretrain 54 gpu 1
python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection/DWPTC-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwpt"]' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --gpu 1 \
    --preprocess 'ninapro-lowpass' \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

# train
for i in $(seq 0 9); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection/DWPTC-ninapro-db5-featuresigimg-win-40-stride-20/one-fold-intra-subject-$i \
    --params .cache/Feature_selection/DWPTC-ninapro-db5-featuresigimg-win-40-stride-20/universal-one-fold-intra-subject/model-0028.params \
    --batch-size 1000 --decay-all --dataset ninapro-db5 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwpt"]' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 16 \
    --dropout 0.65 \
    --gpu 1 \
    --preprocess 'ninapro-lowpass' \
    crossval --crossval-type one-fold-intra-subject --fold $i
done
