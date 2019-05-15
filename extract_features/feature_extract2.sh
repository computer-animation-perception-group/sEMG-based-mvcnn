# Ninapro-db3 win=20 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/feature \
    --window 20 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db3'

# Ninapro-db4 win=20 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
    --window 20 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db4'


