# torchrun --nproc_per_node=8 --master_port=29501 src/validate.py \
#   --validation_file examples/training/sft/wan/overfit-test/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction/model_weights/009500/ \
#   --output_dir validation_results/video-reconstruction/video-09500
# torchrun --nproc_per_node=8 --master_port=29502 src/validate.py \
#   --validation_file data/pexel_part2_6_validate_videos/validation_3.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction/model_weights/010000/ \
#   --output_dir validation_results/video-reconstruction/pexel6-video-10000
