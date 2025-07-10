torchrun --nproc_per_node=8 --master_port=29500 src/validate.py \
  --validation_file examples/training/sft/wan/pixabay_img_pexel_stage2/validation.json \
  --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-pixabay-img-2/model_weights/002000/ \
  --output_dir validation_results/pixabay-img-2/image-002000
