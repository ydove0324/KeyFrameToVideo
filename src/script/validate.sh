# torchrun --nproc_per_node=2 --master_port=29501 src/validate.py \
#   --validation_file examples/training/sft/wan/video-pretrain-2fps/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/013250/ \
#   --output_dir validation_results/video-pretrain-2fps/video-013250-no-finetuning
  # torchrun --nproc_per_node=8 --master_port=29501 src/validate.py \
  # --validation_file examples/training/sft/wan/overfit-test/validation.json \
  # --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/012000/ \
  # --output_dir validation_results/video-pretrain/video-012000
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file data/pexel_part2_6_validate_videos/validation_1.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/002500/ \
#   --output_dir validation_results/video-pretrain/pexel6-video-2500
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file /share/project/huangxu/workspace/KeyFrameToVideo/data/txt-img/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain-2fps/model_weights/016500/ \
#   --output_dir validation_results/video-pretrain/txt-img-016500-pexel-koala-uniform
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file /share/project/huangxu/workspace/KeyFrameToVideo/data/face-img/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/014500/ \
#   --output_dir validation_results/video-pretrain/face-img-014500

# torchrun --nproc_per_node=8 --master_port=29502 src/validate.py \
#   --validation_file data/midjourney-sample-image/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/015000/ \
#   --output_dir validation_results/video-pretrain/midjourney-sample-image-015000

# torchrun --nproc_per_node=8 --master_port=29502 src/validate.py \
#   --validation_file examples/training/sft/wan/video-pretrain-2fps/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain-first-frame-condition/model_weights/022250/ \
#   --output_dir validation_results/video-pretrain-first-frame-condition/video-022500-2fps

torchrun --nproc_per_node=8 --master_port=29502 src/validate.py \
  --validation_file data/video-test/validation1.json \
  --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain-first-frame-condition/model_weights/017750/ \
  --output_dir validation_results/video-pretrain-first-frame-condition/video-test-017750-1fps

  # torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
  # --validation_file data/video-test/validation1.json \
  # --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/014000/ \
  # --output_dir validation_results/video-pretrain/video-test-014000-key-frame-1-uniform
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file data/video-test/validation3.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-pretrain/model_weights/009000/ \
#   --output_dir validation_results/video-pretrain/video-test-009000-key-frame-3
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file /share/project/huangxu/workspace/KeyFrameToVideo/data/face-img/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction-exp2/model_weights/010000/ \
#   --output_dir validation_results/video-reconstruction-exp2/face-img-10000
# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file examples/training/sft/wan/video-reconstruction/validation.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction-exp2/model_weights/007000/  \
#   --output_dir validation_results/video-reconstruction-exp2/video-7000


# torchrun --nproc_per_node=2 --master_port=29502 src/validate.py \
#   --validation_file  data/pexel_part2_6_validate_videos/validation_1.json \
#   --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction-exp2/model_weights/007000/ \
#   --output_dir validation_results/video-reconstruction-exp2/pexel6-video-7000
