from diffusers import WanTransformer3DModel
import torch
from safetensors.torch import load_file, save_file
import json
# 原始模型路径
original_model_path = "/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers"

# 新模型保存路径
new_pretrained_model_name_or_path = "/share/project/huangxu/model/Wan2.1-KeyFrame2V-1.3B-cross-attn"

# 1. 加载原始模型权重
original_state_dict = {}
for file in ["diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"]:
    original_state_dict.update(load_file(f"{original_model_path}/transformer/{file}"))

# 2. 创建新模型（in_channels=276）
config = json.load(open(f"{original_model_path}/transformer/config.json"))
config["in_channels"] = 276
config["added_kv_proj_dim"] = 1536
config["image_dim"] = 256
model = WanTransformer3DModel(**config)

# 3. 手动迁移权重
new_state_dict = model.state_dict()
for name, param in original_state_dict.items():
    if name in new_state_dict:
        if "patch_embedding.weight" in name:  # 针对 patch_embedding 层
            # 原始权重形状为 [1536, 16, 1, 2, 2]
            # 新权重形状为 [1536, 276, 1, 2, 2]
            new_param = new_state_dict[name]
            new_param[:, :16, ...].copy_(param)  # 复制前 16 通道
            new_param[:, 16:, ...].zero_()       # 剩余 256 通道置 0
        else:
            # 其他层直接复制
            new_state_dict[name].copy_(param)
for name, param in new_state_dict.items():
    if "attn2.add_k_proj" in name:
        new_param = new_state_dict[name]
        new_param.zero_()
    if "attn2.add_v_proj" in name:
        new_param = new_state_dict[name]
        new_param.zero_()
    if "condition_embedder.image_embedder.ff" in name:
        new_param = new_state_dict[name]
        new_param.zero_()

# 4. 更新新模型权重
model.load_state_dict(new_state_dict)

# 5. 打印模型结构以验证
print(model)

# 6. 保存新模型
import os
os.makedirs(new_pretrained_model_name_or_path, exist_ok=True)

# 保存权重
save_file(new_state_dict, f"{new_pretrained_model_name_or_path}/transformer/diffusion_pytorch_model.safetensors")

# 保存配置（更新 in_channels）
model.config.in_channels = 276
model.config.added_kv_proj_dim = 1536
model.config.image_dim = 256
model.save_pretrained(new_pretrained_model_name_or_path)

# # 7. 验证（可选）
# # 使用 dummy 输入测试前向传播
# dummy_input = torch.randn(1, 272, 16, 32, 32)  # 调整输入形状 (batch, channels, time, height, width)
# output = model(dummy_input)
# print("Output shape:", output.shape)