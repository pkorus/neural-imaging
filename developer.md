## DCN Training

python3 train_dcn.py --dcn AutoencoderDCN --split 16000:800:1 --param_list config/dcn.csv --epochs 2501 --out data/raw/compression_scenarios_full39 --data data/raw/compression_data/

sudo sshfs -o allow_other pk91@prince.hpc.nyu.edu:/beegfs/pk91/data/raw/ raw-remote/

python3 train_dcn.py --dcn AutoencoderDCN --param_list config/dcn.csv --out ./data/raw/compression_scenarios_final/ --dry --fill dcn_final.csv

## Test dataset sizes

python3 train_dcn.py --dcn TwitterDCN --split 1000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_1000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 2000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_2000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 4000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_4000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 16000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_16000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 31000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_31000/ --data data/compression/



python3 train_dcn.py --dcn TwitterDCN --split 10000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 2 --out ./data/raw/compression_twitter_10000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 10000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 0 --out ./data/raw/compression_twitter_10000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 10000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 1 --out ./data/raw/compression_twitter_10000/ --data data/compression/
python3 train_dcn.py --dcn TwitterDCN --split 10000:1000:1 --param_list config/twitter.csv --epochs 2500 --group 3 --out ./data/raw/compression_twitter_10000/ --data data/compression/
