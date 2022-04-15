import os
import fnmatch

SPLIT = 'train'
filename = '/mnt/lustre/sunweixuan/k400_{}.txt'.format(SPLIT)
src_folder = '/mnt/lustreold/share_data/sunweixuan/video_data/kinect400/{}/'.format(SPLIT)
replace_folder = '/mnt/lustreold/share_data/sunweixuan/video_data/kinect400/replacement/replacement_for_corrupted_k400'

lost_count = 0
total_count = 0

with open('{}.csv'.format(SPLIT),'w') as csv_file:
    with open(filename) as file:
        for line in file:
            total_count += 1
            print(total_count)
            line_list = line.split(' ')
            cur_class = line_list[0].split('/')[0]
            video_name = line_list[0].split('/')[1]
            # print(video_name)
            cur_class = int(line_list[1])
            if os.path.isfile(os.path.join(src_folder, video_name)):
                # print('video file found!', cur_class)
                csv_file.write(os.path.join(src_folder, video_name))
                csv_file.write(',')
                csv_file.write(str(cur_class))
                csv_file.write('\n')
            elif os.path.isfile(os.path.join(replace_folder, video_name)):
                csv_file.write(os.path.join(src_folder, video_name))
                csv_file.write(',')
                csv_file.write(str( cur_class))
                csv_file.write('\n')
                print('video file found in replacement!', cur_class)
            else:
                print(video_name)
                print('video file lost!', lost_count)
                lost_count +=1
                


# csv_file = open('{}.csv'.format(SPLIT),'w')
# orig_csv_file = open('/mnt/lustreold/share_data/sunweixuan/video_data/kinect400/annotations/{}.csv'.format(SPLIT))

# for line in orig_csv_file:
#     line_list = line.split(',')
#     print(line_list)
#     video_name = 



