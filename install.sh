dir=/mnt/lustreold/share_data/sunweixuan/wheel_file

pip install h5py
pip install $dir/torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
pip install fvcore
pip install $dir/torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl
pip install simplejson
pip install av
pip install iopath
pip install psutil
pip install opencv-python
pip install tensorboard
pip install pytorchvideo
pip install timm==0.4.12
pip install $dir/detectron2-0.6+cu101-cp38-cp38-linux_x86_64.whl
pip install sklearn

# 可选, 如果要跑例子, 需要本地存一个ckpt, 1986
# cp /mnt/lustre/share_data/sunweixuan/jx_vit_base_patch16_224_in21k-e5005f0a.pth /mnt/lustre/$USER/.cache/torch/hub/checkpoints