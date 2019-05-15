# ninapro-db6

# single view intra
python -m sigr.train exp --log log --snapshot model \
    --root .cache/ninapro-db6-downsample20-single-intra-20-1/universal-one-fold-intra-subject \
    --batch-size 1000 --decay-all --dataset ninapro-db6 \
    --num-filter 64 \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --balance-gesture 1 \
    --feature-name 'featuresigimg_v2' \
    --fusion-type 'single' \
    --window 1 \
    --num-semg-row 1 --num-semg-col 14 \
    --gpu 0 \
    crossval --crossval-type universal-one-fold-intra-subject --fold 0

#for i in $(seq 0 9);do
#    python -m sigr.train exp --log log --snapshot model \
#        --root .cache/ninapro-db4-single-intra-20-1/one-fold-intra-subject-fold-$i \
#        --params .cache/ninapro-db4-single-intra-20-1/universal-one-fold-intra-subject/model-0028.params \
#        --batch-size 1000 --decay-all --dataset ninapro-db4 \
#        --num-filter 64 \
#        --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#        --balance-gesture 1 \
#        --feature-name 'featuresigimg_v2' \
#        --fusion-type 'single' \
#        --window 1 \
#        --num-semg-row 1 --num-semg-col 12 \
#        --gpu 0 \
#        --preprocess '(ninapro-lowpass, downsample-4)' \
#        crossval --crossval-type one-fold-intra-subject --fold $i
#done



