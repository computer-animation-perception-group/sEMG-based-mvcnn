# ninapro-db3 win=5 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/feature \
    --window 5 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db3'

# ninapro-db3 win=10 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/feature \
    --window 10 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db3'

# ninapro-db3 win=15 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db3/feature \
    --window 15 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db3'

# ninapro-db4 win=5 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
    --window 5 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db4'

# ninapro-db4 win=10 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
    --window 10 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db4'

# ninapro-db4 win=15 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db4/feature \
    --window 15 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db4'

# ninapro-db7 win=5 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/feature \
    --window 5 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db7'

# ninapro-db7 win=10 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/feature \
    --window 10 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db7'

# ninapro-db7 win=15 stride=1
python3 feature_extract_parallel.py \
    -i $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/semg/data \
    -o $HOME/10.214.150.99/semg/dqf/data/ninapro-db7/feature \
    --window 15 --stride 1 \
    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
    --dataset 'ninapro-db7'


