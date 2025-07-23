python script/video_preprocess.py \
    --input /share/project/lzx/video_data/pexels/2 \
    --frames 17 \
    --size auto \
    --fps 17 \
    --max-gpu-clips 6 \
    --output /share/project/huangxu/video-data/pexel-clips-2-filtered \
    --gpu 0,1,2,3,4,5,6,7 \
    --flow_stats /share/project/lzx/video_data/pexels/pexels_stat.jsonl \
    --max-clip-per-video 10