环境要求：
1. 建议先试一下已有的环境，如果可以运行，可以省去以下的的安装新环境的麻烦。
2. 安装新环境：

- conda 创建新环境 Python = 3.8
- cuda: 10.1
- Numpy
- PyTorch = 1.8.1 (这里下载：https://download.pytorch.org/whl/cu101/torch-1.8.1%2Bcu101-cp38-cp38-linux_x86_64.whl)
- hdf5
- fvcore: pip install 'git+https://github.com/facebookresearch/fvcore'
- torchvision = 0.9.1 （这里下载：https://download.pytorch.org/whl/cu101/torchvision-0.9.1%2Bcu101-cp38-cp38-linux_x86_64.whl）
- simplejson: pip install simplejson
- GCC >= 4.9 
- PyAV: conda install av -c conda-forge
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: pip install -U iopath or conda install -c iopath iopath
- psutil: pip install psutil
- OpenCV: pip install opencv-python
- tensorboard: pip install tensorboard
- PyTorchVideo: pip install pytorchvideo
- timm = 0.4.12
- detectron2: (这里下载： https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/detectron2-0.6%2Bcu101-cp38-cp38-linux_x86_64.whl)
- (以上whl文件下载了传上集群再pip install)

数据路径：

1986集群：   /mnt/lustreold/share_data/sunweixuan/video_data(kinect400, kinect600, somethingsomething v2)
1024集群：  

运行：

bash train.sh M3T 2 10 

在train.sh里配置对应的的yaml和数据集路径。

