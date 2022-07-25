

GPU=$1
CPU=5


spring.submit run -p MMG --quotatype=auto --job-name=python --ntasks-per-node 8 --gpu -n$GPU --cpus-per-task $CPU  \
"
python tools/slurmRunNet.py \
  --cfg configs/Kinetics/lxvitv2_B16_16x16_k400.yaml --port 22235 \
  DATA.PATH_TO_DATA_DIR /mnt/lustre/share_data/liuzexiang/Data/k400/ \
  NUM_GPUS $GPU TRAIN.BATCH_SIZE 32
"


# spring.submit arun -p ${PARTITION} --quotatype=auto --job-name=python -n 16 —gpu —gres=gpu:4 --cpus-per-task $CPU  \
# "
# python tools/run_net.py \
#   --cfg configs/Kinetics/xvit_B16_16x16_k400.yaml \
#   DATA.PATH_TO_DATA_DIR /mnt/lustreold/share_data/sunweixuan/video_data/kinect400/ \
#   NUM_GPUS $GPU
# "