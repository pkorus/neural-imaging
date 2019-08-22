#!/usr/bin/bash

# If working on the server, load dependencies
if [ "$USER" = "pk91" ]; then
    module load cudnn/9.0v7.3.0.29
    source ../neural-imaging-pipeline/venv/bin/activate
fi

if [ -n "$2" ]; then
  rep=$2
else
  rep=5
fi

if [ "$3" = "down" ]; then
  dir="8m-ds"
  channel="--patch 256"
else
  dir="8m-no-ds"
  channel="--patch 128 --ds none"
fi

epochs="2501"
nip="DNet"
cam="D90"
manip="sharpen,gaussian,jpeg,resample,awgn,gamma,median"
cmd="python3 train_manipulation.py --end $rep --epochs=$epochs $channel --nip $nip --cam $cam --manip $manip"

if [ "$2" = "dry" ]; then
    cmd="echo $cmd"
fi

# Scope of parameters for exploration
ln="--ln 0.5 --ln 0.25 --ln 0.1 --ln 0.075 --ln 0.05 --ln 0.025 --ln 0.01 --ln 0.005 --ln 0.001"
lc="--lc 1.0 --lc 0.5 --lc 0.1 --lc 0.05 --lc 0.01 --lc 0.005 --lc 0.001"

case "$1" in
    jpeg)
        # Fixed JPEG Experiments
        for jpeg in 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do
            $cmd --dir ./data/raw/$dir/jpeg/$jpeg --jpeg $jpeg
        done
        ;;
   jpeg+nip)
        # Fixed JPEG + Trainable NIP
        for jpeg in 30 40 50 60 70; do
            $cmd --dir ./data/raw/$dir/jpeg-nip+/$jpeg --jpeg $jpeg --train nip $ln
        done
        ;;
    dcn)
        # Fixed DCN Experiments
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/$dir/dcn/$dcn --dcn $dcn
        done
        ;;
    dcn+nip)
        # DCN + Trainable NIP
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/$dir/dcn-nip+/$dcn --dcn $dcn --train nip $ln
        done
       ;;
   dcn+)
        # Trainable DCN
        for dcn in 4k 8k 16k; do
            $cmd --dir ./data/raw/$dir/dcn+/$dcn --dcn $dcn --train dcn $lc
        done
        ;;
   *)
        echo $"Usage: $0 {jpeg|dcn|dcn+nip|dcn+|jpeg+nip} rep [down]"
        exit 1
esac
