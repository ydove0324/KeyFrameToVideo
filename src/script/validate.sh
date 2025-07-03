torchrun --nproc_per_node=8 --master_port=29500 src/validate.py \
  --validation_file examples/training/sft/wan/pixabay_img/validation.json \
  --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-pixabay-img/model_weights/003500/ \
  --output_dir validation_results/3500