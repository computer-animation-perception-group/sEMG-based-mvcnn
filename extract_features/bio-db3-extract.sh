#python feature_extract.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db2/semg \
#    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db2/feature \
#    --window 300 --stride 100 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr \
#    tdd_cor ssc zc iemg var tdd rms mdwt hemg cc hht mavslpphinyomark \
#    --dataset 'biopatrec-db1'

python feature_extract.py \
    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/semg/INTAN \
    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/feature/INTAN \
    --window 15 --stride 5 \
    -f iemg var wamp wl ssc zc \
    tdd_cor mav rms mdwt hemg15 \
    mavslpframewise arc mnf_MEDIAN_POWER \
    psr sampen cc rms hht58 arr29 mnf dwt \
    dwpt cwt \
    --downsample 20 \
    --flip \
    --dataset 'biopatrec-db3'

#python feature_extract.py \
#		-i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/semg/ADS \
#		-o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/feature/ADS \
#		--window 300 --stride 100 \
#		-f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#		--dataset 'biopatrec-db3'
#
#python feature_extract.py \
#		-i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/semg/ADSbias \
#		-o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/feature/ADSbias \
#		--window 300 --stride 100 \
#		-f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#		--dataset 'biopatrec-db3'
#
#python feature_extract.py \
#		-i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/semg/INTAN \
#		-o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db3/feature/INTAN \
#		--window 300 --stride 100 \
#		-f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#		--dataset 'biopatrec-db3'
#
#python feature_extract.py \
#		-i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/semg/TBC \
#		-o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/feature/TBC \
#		--window 300 --stride 100 \
#		-f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#		--dataset 'biopatrec-db4'
#
#python feature_extract.py \
#		-i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/semg/TMC \
#		-o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/feature/TMC \
#		--window 300 --stride 100 \
#		-f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#		--dataset 'biopatrec-db4'

#python feature_extract.py \
#    -i $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/semg/UMC \
#    -o $HOME/10.214.150.99/semg/dqf/data/biopatrec-db4/feature/UMC \
#    --window 300 --stride 100 \
#    -f dwpt dwt mav wl wamp mavslpframewise arc mnf_MEDIAN_POWER psr tdd_cor \
#    --dataset 'biopatrec-db4'




