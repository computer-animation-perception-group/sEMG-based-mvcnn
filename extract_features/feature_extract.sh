# ninapro-db3
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/semg/data \
#    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/feature \
#    --window 20 --stride 1 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'ninapro-db3'
#
## ninapro-db4
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
#    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
#    --window 20 --stride 1 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'ninapro-db4'
#
## ninapro-db7
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/semg/data \
#    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/feature \
#    --window 20 --stride 1 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'ninapro-db7'
#
## biopatrec-db1
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db1/semg \
#    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db1/feature \
#    --window 100 --stride 50 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'biopatrec-db1'
#
## biopatrec-db3
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/semg/ADS \
#    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/feature/ADS \
#    --window 100 --stride 50 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'biopatrec-db3'
#
## biopatrec-db4
#python feature_extract_parallel.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/semg/TBC \
#    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/feature/TBC \
#    --window 100 --stride 50 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --~/.vimswap/feature_extract.sh.swpdataset 'biopatrec-db4'

python feature_extract.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db5/new-semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db5/feature \
    --window 40 --stride 20 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr \
    --flip \
    -s 1 10 \
    --dataset 'ninapro-db5'

#python3 feature_extract.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
#    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
#    --window 20 --stride 1 \
#    --downsample 20 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'ninapro-db4'

 
