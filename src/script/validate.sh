torchrun --nproc_per_node=8 --master_port=29500 src/validate.py \
  --validation_file examples/training/sft/wan/overfit-img-test/validation.json \
  --transformer_path /share/project/huangxu/model/wan-ibq-key-frame-overfit-img-test/model_weights/002000/ \
  --output_dir validation_results/overfit-img-test/image-02000
