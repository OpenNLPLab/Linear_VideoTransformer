

PARTITION=MMG
GPU=$1
CPU=1
g=$(($1<8?$1:8))

spring.submit run -p ${PARTITION} --quotatype=auto --job-name=python --ntasks-per-node $g  -n$1  --gpu --cpus-per-task $CPU  \
"
python tools/slurmRunNet.py \
  --cfg configs/SSv2/lxvitv2_B16_16x16_test.yaml --port 21232 \
  DATA.PATH_TO_DATA_DIR /mnt/lustre/share_data/liuzexiang/Data/ssv2/ \
  NUM_GPUS $GPU \
  TEST.BATCH_SIZE 1 XVIT.SPATIAL_SIZE 289 DATA.NUM_FRAMES 16 LINEAR_TYPE softmax "



# spring.submit arun -p ${PARTITION} --quotatype=auto --job-name=python -n 16 —gpu —gres=gpu:4 --cpus-per-task $CPU  \
# "
# python tools/run_net.py \
#   --cfg configs/Kinetics/xvit_B16_16x16_k400.yaml \
#   DATA.PATH_TO_DATA_DIR /mnt/lustreold/share_data/sunweixuan/video_data/kinect400/ \
#   NUM_GPUS $GPU
# "
