#!/bin/sh
srun --partition=M3T --kill-on-bad-exit=1 -n1 --gres=gpu:2 --cpus-per-task=5 --mpi=pmi2 --job-name=python --quotatype=auto python tools/run_net.py --cfg configs/Kinetics/xvit_B16_16x16_k400.yaml DATA.PATH_TO_DATA_DIR /mnt/lustreold/share_data/sunweixuan/video_data/kinect400/ NUM_GPUS 2
