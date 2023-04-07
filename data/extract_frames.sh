video_root=$1
save_root=$2
files=$(ls $video_root | sort -n)
for video in $files
do
  if [ $video == arun_log ]; then
    echo $video
    continue
  fi
  let i=i+1
  echo $i
  save_dir=$save_root/$(basename $video .webm)
  video_path=$video_root/$video
  filename=$(basename $video .webm)
  if [ ! -d $save_dir ]; then
    mkdir $save_dir
  fi
  ffmpeg -i "${video_path}" -r 30 -loglevel quiet -q:v 1 "$save_dir/${filename}_%6d.jpg"
  if [ $i -eq 10000 ]; then
    break
  fi
done

