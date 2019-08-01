#!/usr/bin/bash

# If working on the server, load dependencies
if [ "$USER" = "pk91" ]; then
    module load cudnn/9.0v7.3.0.29
    source ../neural-imaging-pipeline/venv/bin/activate
fi

nip="DNet"
cam="D90"
cmd="python3 train_manipulation.py --end 10 --patch 128 --epochs=4001 --ds none --nip $nip --cam $cam"

if [ "$2" = "dry" ]; then
    cmd="echo $cmd"
fi

# Scope of parameters for exploration
# ln="--ln 0.2 --ln 0.1 --ln 0.05 --ln 0.01 --ln 0.005 --ln 0.001"
ln="--ln 0.2 --ln 0.1 --ln 0.05"
lc="--lc 1.0 --lc 0.5 --lc 0.1 --lc 0.05 --lc 0.01 --lc 0.005 --lc 0.001"

case "$1" in
    jpeg)
        # Fixed JPEG Experiments
        for jpeg in 10 20 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do
            $cmd --dir ./data/raw/m/jpeg/$jpeg --jpeg $jpeg
        done
        ;;
   jpeg+nip)
        # Fixed JPEG + Trainable NIP
        for jpeg in 30 40 50 60 70; do
            $cmd --dir ./data/raw/m/jpeg+nip/$jpeg --jpeg $jpeg --train nip $ln
        done
        ;;
    dcn)
        # Fixed DCN Experiments
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/m/dcn/$dcn --dcn $dcn
        done
        ;;
    dcn+nip)
        # DCN + Trainable NIP
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/m/dcn+nip/$dcn --dcn $dcn --train nip $ln
        done
       ;;
   dcn+)
        # Trainable DCN
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/m/dcn+/$dcn --dcn $dcn --train dcn $lc
        done
        ;;
   *)
        echo $"Usage: $0 {jpeg|dcn|dcn+nip|dcn+|jpeg+nip}"
        exit 1
esac
