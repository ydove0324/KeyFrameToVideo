python tool/video_preprocess.py \
    --input data/pexel_part2_6 \
    --frames 17 \
    --size auto \
    --fps 17 \
    --max-gpu-clips 6 \
    --output /share/project/huangxu/video-data/pexel-clips-part2_6_filtered \
    --gpu 0,1,2,3 \
    --flow_stats /share/project/lzx/video_data/pexels/pexels_stat.jsonl
   