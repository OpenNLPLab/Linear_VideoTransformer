import os
import pdb
import subprocess
from multiprocessing import Process,Queue
from tqdm import tqdm
video_root = '/mnt/lustre/share_data/sunweixuan/video_data/somethingv2'
save_root = '/mnt/lustre/liuzexiang/Data/extracted_frames'
videos = os.listdir(video_root)
videos.remove('arun_log')
videos.sort(key=lambda v: int(v.split('.')[0]))
videos_ = videos[:]
def process(videos_):
    for video in tqdm(videos_):
        basename = video.split('.')[0]
        save_dir = save_root + '/' + basename
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        video_path = video_root + '/' + video
        cmd = 'ffmpeg -loglevel quiet -i \"{}\" -r 30 -q:v 1 \"{}/{}_%6d.jpg\"'.format(video_path, save_dir, basename)
        subprocess.call(cmd, shell=True)

process_list = []
for i in range(20):
    split = len(videos_)//20
    videos_split = videos_[i*split:(i+1)*split]
    p = Process(target=process, args=(videos_split,))
    p.start()
    process_list.append(p)
for p in process_list:
    p.join()