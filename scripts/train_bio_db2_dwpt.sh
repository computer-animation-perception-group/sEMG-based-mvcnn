# pretrain 49 gpu 1
python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_pretrain/lr0.25-DWPTC-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 50 --lr-step 16 --lr-step 24 --snapshot-period 50 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwpt"]' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 8 \
    --dropout 0.65 \
    --lr 0.25 \
    --gpu 1 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

# train
for i in $(seq 0 16); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_pretrain/lr0.25-DWPTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i \
    --params .cache/Feature_selection_pretrain/lr0.25-DWPTC-biopatrec-db2-3FSsigimg-win-300-stride-100/universal-one-fold-intra-subject/model-0050.params \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwpt"]' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 8 \
    --dropout 0.65 \
    --lr 0.25 \
    --gpu 1 \
    crossval --crossval-type temp-one-fold-intra-subject --fold $i
done
# true train
for i in $(seq 0 16); do
    python -m sigr.train exp --log log --snapshot model \
    --root .cache/Feature_selection_AfterPretrain/lr0.25-DWPTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i \
    --params .cache/Feature_selection_pretrain/lr0.25-DWPTC-biopatrec-db2-3FSsigimg-win-300-stride-100/one-fold-intra-subject-$i/model-0028.params \
    --batch-size 800 --decay-all --dataset biopatrec-db2 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --feature-list '["dwpt"]' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 8 \
    --dropout 0.65 \
    --lr 0.25 \
    --gpu 1 \
    crossval --crossval-type one-fold-intra-subject --fold $i
done
