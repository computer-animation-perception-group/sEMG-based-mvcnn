## pretrain 48 gpu 0
ver = 1.0.0.1
python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_pretrain/lr0.25-DWTC-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 50 --lr-step 16 --lr-step 24 --lr-step 36 --snapshot-period 50 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwt"]' \
    --window 1 \
    --dropout 0.65 \
    --num-semg-row 1 --num-semg-col 8 \
    --lr 1 \
    --gpu 0 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

## train
for i in $(seq 0 16); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_pretrain/lr0.25-DWTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i \
    --params .cache/Feature_selection_pretrain/lr0.25-DWTC-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0050.params \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 50 --lr-step 16 --lr-step 24 --snapshot-period 50 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwt"]' \
    --window 1 \
    --dropout 0.65 \
    --num-semg-row 1 --num-semg-col 8 \
    --lr 0.01 \
    --gpu 0 \
    crossval --crossval-type temp-one-fold-intra-subject --fold $i
done

# truely train
for i in $(seq 0 16); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_AfterPretrain/lr0.25-DWTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i \
    --params .cache/Feature_selection_pretrain/lr0.25-DWTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i/model-0050.params \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwt"]' \
    --window 1 \
    --dropout 0.65 \
    --num-semg-row 1 --num-semg-col 8 \
    --gpu 0 \
    crossval --crossval-type one-fold-intra-subject --fold $i
done
