#!/bin/sh
srun --partition=MMG --kill-on-bad-exit=1 -n1 --gres=gpu:8 --cpus-per-task=40 --ntasks-per-node=1 --mpi=pmi2 --job-name=python --quotatype=auto python tools/run_net.py --cfg configs/Kinetics/xvit_B16_16x16_k600.yaml DATA.PATH_TO_DATA_DIR /mnt/lustreold/share_data/sunweixuan/video_data/kinect600 NUM_GPUS 8
