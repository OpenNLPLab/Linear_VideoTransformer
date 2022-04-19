dir=/mnt/lustreold/share_data/sunweixuan/wheel_file

pip install h5py
pip install $dir/torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
pip install fvcore
pip install $dir/torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl
pip install simplejson
# conda install av -c conda-forge
pip install av
pip install iopath
pip install psutil
pip install opencv-python
pip install tensorboard
pip install pytorchvideo
pip install timm==0.4.12
pip install $dir/detectron2-0.6+cu101-cp38-cp38-linux_x86_64.whl