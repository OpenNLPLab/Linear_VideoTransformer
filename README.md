This repo is the official implementations of "Linear Video Transformer with Feature Fixation". 
# Installation

## Install pytorch

```
conda create -n video_transformer python=3.8

conda activate video_transformer

bash install.sh
```
# Data pre-process

## 1.download K400, K600 and SSv2 dataset
## 2.extract video frames
```
bash data/extract_frames.sh $PATH_TO_YOUR_DATA_ROOT $PATH_TO_YOUR_SAVR_DIR
```
## 3.save the extraced frames in h5 format
```
python data/process.py $PATH_TO_YOUR_EXTRACTED_FRAMES $PATH_TO_YOUR_H5_SAVR_DIR
```
# Train: change config and DataPath in the shell.

K400:
```
sh train_k400_linear.sh 
```

K600:
```
sh train_k600_linear.sh 
```

SSv2:
```
sh train_ssv2_linear.sh 
```

Charades:
```
sh train_charades_linear.sh 
```

