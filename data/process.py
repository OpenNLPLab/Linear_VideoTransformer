import json
import io
import os
import h5py
from PIL import Image
import pdb
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Process, Queue

root_dir = '/mnt/lustre/liuzexiang/Data/extracted_frames'
save_dir = '/mnt/lustre/liuzexiang/Data/seq_h5'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def convert_frames2h5(frames_dirs):
    for frames_dir in tqdm(frames_dirs):
        frames = os.listdir(root_dir+'/'+frames_dir)
        frames.sort()
        datas = []
        if len(frames) == 0:
            print(frames_dir, flush=True)
            continue
        for j, frame in enumerate(frames):
            frame_path = root_dir + '/' + frames_dir + '/' + frame
            with open(frame_path, 'rb') as f:
                data = f.read()
                datas.append(data)

        with h5py.File(f"{save_dir}/{frames_dir}.h5", "w") as f:
            f.create_dataset(frames_dir, data=np.array(datas))

        # data_ = h5py.File(f"{frames_dir}.hdf5", "r")[frames_dir]
        # raw_frame = data_[0]
        # img_image = np.asarray(Image.open(io.BytesIO(raw_frame)).convert("RGB"))
        # im = np.asarray(Image.open(frame_path))
        # assert im==img_image

def load_trainLabeljson():
    with open('something-something-v2-labels.json', 'r') as f:
        label_dict = json.load(f)
    with open('something-something-v2-train.json', 'r') as f:
        label_json = json.load(f)
    for video in tqdm(label_json):
        video_name = video["id"]
        template = video["template"]
        template = template.replace("[", "")
        template = template.replace("]", "")
        label = int(label_dict[template])

        with open('train_videofolder.txt', 'a') as f:
            f.write(video_name+' ' + ',' + ' ' + str(label)+'\n')
def load_valLabeljson():
    with open('something-something-v2-labels.json', 'r') as f:
        label_dict = json.load(f)
    with open('something-something-v2-validation.json', 'r') as f:
        label_json = json.load(f)
    for video in tqdm(label_json):
        video_name = video["id"]
        template = video["template"]
        template = template.replace("[", "")
        template = template.replace("]", "")
        label = int(label_dict[template])
        with open('val_videofolder.txt', 'a') as f:
            f.write(video_name+' ' + ',' + ' ' + str(label)+'\n')
def load_testLabeljson():
    with open('something-something-v2-labels.json', 'r') as f:
        label_dict = json.load(f)
    with open('something-something-v2-test.json', 'r') as f:
        label_json = json.load(f)
    pdb.set_trace()
    for video in tqdm(label_json):
        video_name = video["id"]
        template = video["template"]
        template = template.replace("[", "")
        template = template.replace("]", "")
        label = int(label_dict[template])
        with open('test_videofolder.txt', 'a') as f:
            f.write(video_name+' ' + ',' + ' ' + str(label)+'\n')
# load_trainLabeljson()
# load_valLabeljson()
# load_testLabeljson()


###  extract_frames
if __name__ == '__main__':
    frames_dirs = os.listdir(root_dir)
    frames_dirs.sort(key=lambda frame_dir: int(frame_dir))
    process_list = []
    start = 0
    end = len(frames_dirs)
    frames_dirs = frames_dirs[start:end]

    num_workers = 40
    split = len(frames_dirs) // num_workers

    for i in range(num_workers):
        if i == num_workers-1:
            videos_split = frames_dirs[i * split:]
        else:
            videos_split = frames_dirs[i*split:(i+1)*split]
        p = Process(target=convert_frames2h5, args=(videos_split,))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()






