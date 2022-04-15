

PARTITION=$1
GPU=$2
CPU=$3


spring.submit arun -p ${PARTITION} --quotatype=auto --job-name=python --gres=gpu:$GPU --gpu --cpus-per-task $CPU  --ntasks-per-node=1 \
"
python tools/run_net.py \
  --cfg configs/Kinetics/xvit_B16_16x16_k400.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/lustreold/share_data/sunweixuan/video_data/kinect400/ \
  NUM_GPUS $GPU
"

