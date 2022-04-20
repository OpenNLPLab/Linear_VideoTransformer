dir=/mnt/lustre/share_data/qinzhen/video_transformer/wheel_file

pip install --user h5py
pip install --user $dir/torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
pip install --user fvcore
pip install --user $dir/torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl
pip install --user simplejson
pip install --user av
pip install --user iopath
pip install --user psutil
pip install --user opencv-python
pip install --user tensorboard
pip install --user pytorchvideo
pip install --user timm==0.4.12
pip install --user $dir/detectron2-0.6+cu101-cp38-cp38-linux_x86_64.whl
pip install --user sklearn

# 可选, 如果要跑例子, 需要本地存一个ckpt, 1024
# cp /mnt/lustre/share_data/qinzhen/video_transformer/jx_vit_base_patch16_224_in21k-e5005f0a.pth /mnt/lustre/$USER/.cache/torch/hub/checkpoints