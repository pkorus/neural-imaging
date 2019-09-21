#!/bin/bash
manip="--manip sharpen:0.5,gaussian:0.5,jpeg:90,resample:75,awgn:3,median:3"
./test_fan.py --dir ./data/m/7-rgb/jpeg/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_jpeg_clic.log
./test_fan.py --dir ./data/m/7-rgb/jpeg/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 $manip| tee ~/Dropbox/logs/rgb_jpeg_raw.log
./test_fan.py --dir ./data/m/7-rgb/jpeg/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 $manip| tee ~/Dropbox/logs/rgb_jpeg_kodak.log

./test_fan.py --dir ./data/m/7-raw/jpeg/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_jpeg_clic.log
./test_fan.py --dir ./data/m/7-raw/jpeg/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_jpeg_raw.log
./test_fan.py --dir ./data/m/7-raw/jpeg/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_jpeg_kodak.log

./test_fan.py --dir ./data/m/7-rgb/dcn/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn_clic.log
./test_fan.py --dir ./data/m/7-rgb/dcn/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn_raw.log
./test_fan.py --dir ./data/m/7-rgb/dcn/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn_kodak.log

./test_fan.py --dir ./data/m/7-raw/dcn/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn_clic.log
./test_fan.py --dir ./data/m/7-raw/dcn/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn_raw.log
./test_fan.py --dir ./data/m/7-raw/dcn/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn_kodak.log

./test_fan.py --dir ./data/m/7-rgb/dcn+/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn+_clic.log
./test_fan.py --dir ./data/m/7-rgb/dcn+/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn+_raw.log
./test_fan.py --dir ./data/m/7-rgb/dcn+/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 $manip | tee ~/Dropbox/logs/rgb_dcn+_kodak.log

./test_fan.py --dir ./data/m/7-raw/dcn+/ --re ".*" --data ./data/rgb/clic512  --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn+_clic.log
./test_fan.py --dir ./data/m/7-raw/dcn+/ --re ".*" --data ./data/rgb/raw512   --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn+_raw.log
./test_fan.py --dir ./data/m/7-raw/dcn+/ --re ".*" --data ./data/rgb/kodak512 --patch 64 --patches 10 | tee ~/Dropbox/logs/rgb_dcn+_kodak.log
